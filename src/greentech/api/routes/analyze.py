"""Routes pour l'analyse d'articles par IA.

Endpoint principal POST /analyze qui orchestre la chaine complete :
reception → nettoyage → classification IA → resume SaaS → stockage.

Trois modes d'entree sont supportes :
    - URL (champ `url` du JSON) — le contenu est extrait de la page.
    - Texte brut (champ `texte` du JSON) — le contenu est utilise tel quel.
    - Fichier (upload multipart sur `/analyze/file`) — le texte est extrait
      selon le format (.txt, .md, .pdf, .docx, .html).

Les analyses sont executees en arriere-plan pour ne pas bloquer
la requete HTTP. Un job_id permet de suivre l'avancement.

"""

from __future__ import annotations

import io
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from loguru import logger
from sqlalchemy import select

from greentech.api.dependencies import get_current_user
from greentech.api.schemas.analysis import (
    AnalysisInput,
    AnalysisJobCreated,
    AnalysisResult,
    AnalysisStatus,
)
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import AnalysisLog, Article, User

router = APIRouter(prefix="/analyze", tags=["Analyse IA"])

# Stockage en memoire des jobs d'analyse (remplacable par Redis en production)
_jobs: dict[uuid.UUID, AnalysisResult] = {}

# Contraintes d'upload : on refuse les fichiers > 10 Mo pour proteger le serveur
# (parsing PDF/DOCX en memoire). Les articles depassent rarement quelques dizaines
# de Ko, donc cette limite est largement suffisante.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# Extensions acceptees avec leur extracteur dedie. On se limite aux formats qui
# contiennent majoritairement du texte exploitable pour une analyse d'article.
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}


async def _extract_text_from_url(url: str) -> tuple[str, str]:
    """Extrait le titre et le contenu textuel d'une URL.

    Effectue une requete HTTP GET et extrait le texte brut du HTML
    en supprimant les balises. Methode simplifiee adaptee aux articles.

    Args:
        url: URL de l'article a extraire.

    Returns:
        Tuple (titre, contenu_texte).

    Raises:
        HTTPException: Si l'URL est inaccessible.
    """
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "GreenTech-Bot/1.0"},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        msg = f"Impossible d'acceder a l'URL : {e}"
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg)

    html = response.text

    # Extraction basique du titre
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    titre = title_match.group(1).strip() if title_match else url

    # Suppression des balises HTML pour obtenir le texte brut
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return titre, text


def _extract_text_from_html_bytes(raw: bytes) -> str:
    """Extrait le texte brut d'un document HTML stocke en memoire.

    Applique la meme strategie que `_extract_text_from_url` (suppression des
    balises script/style puis des autres balises), adaptee a un contenu deja
    telecharge pour eviter un aller-retour reseau.
    """
    html = raw.decode("utf-8", errors="replace")
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_text_from_pdf(raw: bytes) -> str:
    """Extrait le texte d'un PDF en memoire via pypdf.

    pypdf ne supporte pas les PDF scannes (images) : dans ce cas il retourne
    une chaine vide ou quasi vide. L'appelant doit verifier la longueur.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(page_text)
    return "\n\n".join(parts).strip()


def _extract_text_from_docx(raw: bytes) -> str:
    """Extrait le texte d'un DOCX en memoire via python-docx.

    Concatene tous les paragraphes non vides avec un saut de ligne.
    Les tableaux sont traites ligne par ligne, cellule par cellule.
    """
    from docx import Document

    document = Document(io.BytesIO(raw))
    parts: list[str] = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts).strip()


async def _extract_text_from_upload(upload: UploadFile) -> tuple[str, str]:
    """Lit un fichier uploade et extrait titre + contenu textuel.

    Dispatche vers l'extracteur approprie selon l'extension. Le nom de fichier
    (sans extension) sert de titre par defaut, ce qui est plus parlant pour
    l'utilisateur que la troncature des 100 premiers caracteres du contenu.

    Args:
        upload: Fichier fourni par FastAPI via UploadFile.

    Returns:
        Tuple (titre, contenu_texte).

    Raises:
        HTTPException: Si l'extension est refusee, si le fichier depasse la
            taille limite, ou si l'extraction echoue / retourne trop peu de texte.
    """
    filename = upload.filename or "document"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Extension '{extension}' non supportee. Formats acceptes : "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    raw = await upload.read()
    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fichier trop volumineux (max {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} Mo)",
        )
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Fichier vide",
        )

    try:
        if extension in {".txt", ".md"}:
            contenu = raw.decode("utf-8", errors="replace").strip()
        elif extension == ".pdf":
            contenu = _extract_text_from_pdf(raw)
        elif extension == ".docx":
            contenu = _extract_text_from_docx(raw)
        else:  # .html / .htm
            contenu = _extract_text_from_html_bytes(raw)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Extraction echouee pour {filename}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Impossible d'extraire le texte du fichier : {exc}",
        ) from exc

    if len(contenu) < 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Contenu extrait trop court (minimum 50 caracteres). "
                "Si c'est un PDF scanne, il n'est pas lisible sans OCR."
            ),
        )

    titre = Path(filename).stem or "Document uploade"
    return titre[:500], contenu


async def _run_analysis(
    job_id: uuid.UUID,
    url: str | None,
    texte: str | None,
    titre_override: str | None = None,
) -> None:
    """Execute la chaine complete d'analyse IA en arriere-plan.

    Orchestre : extraction → classification → resume → stockage.
    Met a jour le statut du job a chaque etape.

    Args:
        job_id: Identifiant du job d'analyse.
        url: URL de l'article (si fournie).
        texte: Texte brut (si fourni).
        titre_override: Titre a utiliser prioritairement (ex. nom de fichier uploade).
            Si None, un titre est deduit du contenu ou de l'URL.
    """
    _jobs[job_id] = AnalysisResult(
        job_id=job_id,
        statut=AnalysisStatus.EN_COURS,
    )

    try:
        # 1. Extraction du contenu
        if url:
            titre, contenu = await _extract_text_from_url(url)
        else:
            titre = texte[:100] + "..." if texte and len(texte) > 100 else (texte or "")
            contenu = texte or ""

        if titre_override:
            titre = titre_override

        if len(contenu) < 50:
            _jobs[job_id] = AnalysisResult(
                job_id=job_id,
                statut=AnalysisStatus.ERREUR,
                erreur="Contenu insuffisant pour l'analyse (minimum 50 caracteres)",
            )
            return

        # 2. Insertion en base (ou recuperation si URL deja connue)
        async with async_session_factory() as session:
            article_url = url or f"analyse-directe://{job_id}"

            stmt = select(Article).where(Article.url == article_url)
            result = await session.execute(stmt)
            article = result.scalar_one_or_none()

            if article is None:
                article = Article(
                    titre=titre[:500],
                    url=article_url,
                    contenu=contenu,
                )
                session.add(article)
                await session.commit()
                await session.refresh(article)

            id_article = article.id_article

        # 3. Classification IA
        est_green_it: bool | None = None
        score_confiance: float | None = None
        modele_nom: str | None = None
        temps_ms: int | None = None

        try:
            from greentech.ai.models.inference import classify_article

            prediction = await classify_article(id_article)
            est_green_it = prediction.est_green_it
            score_confiance = prediction.score_confiance
            modele_nom = prediction.modele
            temps_ms = prediction.temps_ms
        except FileNotFoundError:
            logger.warning("Modele de production non disponible, classification ignoree")
        except Exception as e:
            logger.error(f"Erreur classification : {e}")

        # 4. Resumes via HuggingFace SaaS (Qwen2.5-7B-Instruct)
        #    - Resume general : toujours genere si contenu suffisant
        #    - Resume ecologique : uniquement si article classe Green IT
        #    Les deux appels sont parallelises via asyncio.gather pour minimiser la latence.
        resume: str | None = None
        resume_ecologique: str | None = None

        import asyncio

        from greentech.ai.services.summarizer import (
            summarize_article,
            summarize_green_for_article,
        )

        taches = [summarize_article(id_article)]
        if est_green_it is True:
            taches.append(summarize_green_for_article(id_article))

        resultats_resumes = await asyncio.gather(*taches, return_exceptions=True)

        # Resume general
        general = resultats_resumes[0]
        if isinstance(general, Exception):
            logger.error(f"Erreur resume general : {general}")
        elif general.succes:
            resume = general.resume

        # Resume ecologique (si applicable)
        if est_green_it is True and len(resultats_resumes) > 1:
            green = resultats_resumes[1]
            if isinstance(green, Exception):
                logger.error(f"Erreur resume ecologique : {green}")
            elif green.succes:
                resume_ecologique = green.resume

        # 5. Log d'analyse
        async with async_session_factory() as session:
            log = AnalysisLog(
                id_article=id_article,
                nom_modele=modele_nom or "non-disponible",
                temps_inference_ms=temps_ms,
                prediction=est_green_it,
                confiance=score_confiance,
            )
            session.add(log)
            await session.commit()

        # 6. Mise a jour du job
        _jobs[job_id] = AnalysisResult(
            job_id=job_id,
            statut=AnalysisStatus.TERMINE,
            id_article=id_article,
            titre=titre[:500],
            est_green_it=est_green_it,
            score_confiance=score_confiance,
            resume=resume,
            resume_ecologique=resume_ecologique,
            modele_classification=modele_nom,
            temps_inference_ms=temps_ms,
            date_analyse=datetime.now(UTC),
        )

        logger.info(f"Analyse {job_id} terminee : article_id={id_article}")

    except Exception as e:
        logger.exception(f"Erreur analyse {job_id}")
        _jobs[job_id] = AnalysisResult(
            job_id=job_id,
            statut=AnalysisStatus.ERREUR,
            erreur=str(e),
        )


@router.post(
    "",
    response_model=AnalysisJobCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lancer une analyse IA",
)
async def create_analysis(
    data: AnalysisInput,
    background_tasks: BackgroundTasks,
    _current_user: User = Depends(get_current_user),
) -> AnalysisJobCreated:
    """Soumet une URL ou un texte pour analyse IA.

    Declenche en arriere-plan la chaine complete :
    extraction du contenu, classification Green IT par le modele,
    generation du resume via l'API HuggingFace, et stockage en base.

    Le job_id retourne permet de suivre l'avancement via GET /analyze/{job_id}.

    Args:
        data: URL ou texte a analyser (exactement un des deux).
        background_tasks: Gestionnaire de taches en arriere-plan FastAPI.
        _current_user: Utilisateur authentifie (verification du token).

    Returns:
        Identifiant du job d'analyse avec statut initial.
    """
    job_id = uuid.uuid4()

    _jobs[job_id] = AnalysisResult(
        job_id=job_id,
        statut=AnalysisStatus.EN_ATTENTE,
    )

    background_tasks.add_task(_run_analysis, job_id, data.url, data.texte)

    source = f"URL={data.url}" if data.url else f"texte ({len(data.texte or '')} chars)"
    logger.info(f"Analyse soumise {job_id} : {source}")

    return AnalysisJobCreated(job_id=job_id)


@router.post(
    "/file",
    response_model=AnalysisJobCreated,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Lancer une analyse IA sur un fichier uploade",
)
async def create_analysis_from_file(
    background_tasks: BackgroundTasks,
    fichier: UploadFile = File(..., description="Fichier a analyser"),
    _current_user: User = Depends(get_current_user),
) -> AnalysisJobCreated:
    """Accepte un fichier uploade, extrait son texte et lance l'analyse IA.

    Formats supportes : .txt, .md, .pdf, .docx, .html, .htm.
    La taille est plafonnee a 10 Mo. Les PDF scannes (images) ne sont pas
    lisibles sans OCR et seront rejetes avec une erreur 422.

    Le pipeline applique ensuite est strictement le meme que pour une URL ou
    un texte : classification via le modele Llama entraine, puis resume
    generaliste via l'API HuggingFace Serverless, avant stockage en base.

    Args:
        background_tasks: Gestionnaire de taches en arriere-plan FastAPI.
        fichier: Fichier uploade (multipart/form-data).
        _current_user: Utilisateur authentifie (verification du token JWT).

    Returns:
        Identifiant du job d'analyse pour suivre l'avancement via GET /analyze/{job_id}.

    Raises:
        HTTPException: 413 si fichier trop gros, 415 si format non supporte,
            422 si le contenu extrait est insuffisant ou le fichier illisible.
    """
    titre, contenu = await _extract_text_from_upload(fichier)

    job_id = uuid.uuid4()
    _jobs[job_id] = AnalysisResult(
        job_id=job_id,
        statut=AnalysisStatus.EN_ATTENTE,
    )

    background_tasks.add_task(
        _run_analysis,
        job_id,
        None,
        contenu,
        titre,
    )

    logger.info(
        f"Analyse fichier soumise {job_id} : "
        f"nom='{fichier.filename}' taille={len(contenu)} chars"
    )
    return AnalysisJobCreated(job_id=job_id)


@router.get(
    "/{job_id}",
    response_model=AnalysisResult,
    summary="Statut d'une analyse",
)
async def get_analysis_status(
    job_id: uuid.UUID,
    _current_user: User = Depends(get_current_user),
) -> AnalysisResult:
    """Retourne le statut et le resultat d'un job d'analyse.

    Args:
        job_id: Identifiant UUID du job retourne par POST /analyze.
        _current_user: Utilisateur authentifie.

    Returns:
        Statut courant et resultats (si termines).

    Raises:
        HTTPException: 404 si le job_id n'existe pas.
    """
    result = _jobs.get(job_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job d'analyse {job_id} introuvable",
        )

    return result
