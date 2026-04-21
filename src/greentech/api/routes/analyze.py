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


def _extract_article_from_html(html: str, *, url: str | None = None) -> tuple[str | None, str]:
    """Isole le titre et le corps d'article d'un document HTML via trafilatura.

    Trafilatura combine des heuristiques DOM et ``justext`` pour eliminer le
    boilerplate (menus, footers, cookies, widgets sociaux, scripts, balises
    de navigation). Specialise sur les articles de presse, il evite que le
    modele de classification et le resumeur ne voient que du texte d'interface
    au lieu du contenu reel. En cas d'echec (SPA vide, paywall, page tres
    atypique), on retombe sur une extraction regex minimale pour garantir
    qu'on ait au moins quelque chose a analyser.

    Args:
        html: Document HTML complet, deja decode en texte.
        url: URL d'origine, utilisee par trafilatura pour activer certaines
            heuristiques specifiques a un domaine (optionnel).

    Returns:
        Tuple (titre_ou_None, contenu_texte). Le titre peut etre None si
        aucun ``<title>`` ni metadonnee n'a pu etre detecte.
    """
    from trafilatura import bare_extraction

    titre: str | None = None
    texte: str = ""

    try:
        doc = bare_extraction(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            with_metadata=True,
        )
    except Exception as exc:
        logger.warning(f"Extraction trafilatura echouee ({url or 'html brut'}) : {exc}")
        doc = None

    # L'API trafilatura peut retourner un Document dataclass ou un dict selon
    # la version et les options. On gere les deux pour rester robuste a une
    # montee de version mineure.
    if doc is not None:
        if isinstance(doc, dict):
            titre_extrait = doc.get("title") or ""
            texte = (doc.get("text") or doc.get("raw_text") or "").strip()
        else:
            titre_extrait = getattr(doc, "title", "") or ""
            texte = (getattr(doc, "text", None) or getattr(doc, "raw_text", "") or "").strip()
        titre = titre_extrait.strip() or None

    if texte:
        return titre, texte

    # Fallback : trafilatura n'a rien trouve (page atypique, JS-rendered, ...).
    # On garde l'ancienne extraction regex comme filet de securite ; elle est
    # bruyante mais vaut mieux qu'un contenu vide qui ferait echouer le
    # pipeline en aval.
    logger.info(f"Fallback regex pour l'extraction HTML (url={url})")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match and not titre:
        titre = title_match.group(1).strip() or None

    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return titre, text


async def _extract_text_from_url(url: str) -> tuple[str, str]:
    """Telecharge une URL et en extrait le titre + le corps d'article.

    Utilise httpx pour la requete HTTP (controle fin du timeout, du
    User-Agent et de la politique de redirection), puis delegue l'extraction
    a ``_extract_article_from_html`` qui s'appuie sur trafilatura pour
    isoler le contenu principal. Le titre extrait est tronque a 500
    caracteres pour tenir dans la contrainte de la colonne ``articles.titre``.

    Args:
        url: URL de l'article a extraire.

    Returns:
        Tuple (titre, contenu_texte) ou le contenu est limite au corps
        d'article. Le titre retombe sur l'URL si aucune metadonnee n'est
        disponible, pour garantir un affichage lisible dans l'UI.

    Raises:
        HTTPException: 422 si l'URL est inaccessible ou si le serveur
            distant a retourne une erreur HTTP.
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
    titre, contenu = _extract_article_from_html(html, url=url)
    return (titre or url)[:500], contenu


def _extract_text_from_html_bytes(raw: bytes) -> str:
    """Extrait le corps d'article d'un HTML brut via trafilatura.

    Utilise lors de l'upload d'un fichier ``.html`` / ``.htm`` : la logique
    d'extraction est identique a celle d'une URL, mais sans aller-retour
    reseau. Le titre est ignore ici car l'appelant (``_extract_text_from_upload``)
    se sert du nom de fichier comme titre par defaut.

    Args:
        raw: Contenu brut du fichier HTML.

    Returns:
        Texte du corps d'article debarrasse du boilerplate.
    """
    html = raw.decode("utf-8", errors="replace")
    _, contenu = _extract_article_from_html(html)
    return contenu


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

    Orchestre l'enchainement strictement sequentiel :

    1. Extraction du contenu (trafilatura pour URL, pypdf/docx pour uploads)
    2. Insertion en base (ou recuperation si URL deja connue)
    3. Resume de classification via LLM (peuple ``articles.resume``)
    4. Classification Qwen3-4B + LoRA sur le resume
       (reproduit la feature d'entrainement ``titre + resume``)
    5. Resume ecologique si l'article est classifie Green IT
       (peuple ``articles.resume_ecologique``)
    6. Log d'analyse dans ``analysis_logs``
    7. Mise a jour du statut du job en memoire

    Le resume de classification (etape 3) est genere AVANT la classification
    (etape 4) car le classifieur Qwen3-4B + LoRA a ete entraine sur le
    resume, pas sur le contenu brut. Cet ordre garantit l'absence de
    derive de distribution entre entrainement et inference.

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

        # 3. Resume de classification (feature d'entree du classifieur)
        #    Le classifieur Qwen3-4B + LoRA a ete entraine sur `titre + resume`
        #    (et non sur le contenu brut) pour garantir une distribution
        #    uniforme entre toutes les sources (arXiv, TechCrunch, NewsData,
        #    uploads). On doit donc generer le resume AVANT la classification
        #    et utiliser le meme prompt centralise qu'a l'entrainement.
        from greentech.ai.services.summarizer import (
            summarize_article,
            summarize_green_for_article,
        )

        resume: str | None = None
        resume_ecologique: str | None = None

        try:
            resume_result = await summarize_article(id_article)
            if resume_result.succes:
                resume = resume_result.resume
            else:
                logger.warning(
                    f"Resume de classification echoue pour article {id_article} : "
                    f"{resume_result.erreur}"
                )
        except Exception as exc:
            logger.error(f"Erreur resume de classification : {exc}")

        # 4. Classification IA sur le resume (lecture de articles.resume)
        est_green_it: bool | None = None
        score_confiance: float | None = None
        modele_nom: str | None = None
        temps_ms: int | None = None

        if resume is None:
            # Sans resume, la classification ne peut pas s'executer (classify_article
            # leve ValueError si articles.resume est NULL). On preserve le job en
            # statut terminal mais sans prediction, plutot que de propager une erreur.
            logger.warning(
                f"Classification sautee pour article {id_article} : pas de resume disponible"
            )
        else:
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

        # 5. Resume ecologique (uniquement si Green IT confirme)
        #    Genere apres la classification, sur le contenu complet de l'article
        #    (et non sur le resume) car cette synthese est destinee a l'affichage
        #    utilisateur et peut se permettre un contexte plus riche que celui
        #    contraint par le budget tokens du classifieur.
        if est_green_it is True:
            try:
                green_result = await summarize_green_for_article(id_article)
                if green_result.succes:
                    resume_ecologique = green_result.resume
                else:
                    logger.warning(
                        f"Resume ecologique echoue pour article {id_article} : "
                        f"{green_result.erreur}"
                    )
            except Exception as exc:
                logger.error(f"Erreur resume ecologique : {exc}")

        # 6. Log d'analyse
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

        # 7. Mise a jour du job
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
        f"Analyse fichier soumise {job_id} : nom='{fichier.filename}' taille={len(contenu)} chars"
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
