"""Routes pour l'analyse d'articles par IA.

Endpoint principal POST /analyze qui orchestre la chaine complete :
reception → nettoyage → classification IA → resume SaaS → stockage.

Les analyses sont executees en arriere-plan pour ne pas bloquer
la requete HTTP. Un job_id permet de suivre l'avancement.

"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
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


async def _run_analysis(job_id: uuid.UUID, url: str | None, texte: str | None) -> None:
    """Execute la chaine complete d'analyse IA en arriere-plan.

    Orchestre : extraction → classification → resume → stockage.
    Met a jour le statut du job a chaque etape.

    Args:
        job_id: Identifiant du job d'analyse.
        url: URL de l'article (si fournie).
        texte: Texte brut (si fourni).
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

        # 4. Resume via HuggingFace (si contenu suffisant)
        resume: str | None = None
        try:
            from greentech.ai.services.summarizer import summarize_article

            summary_result = await summarize_article(id_article)
            if summary_result.succes:
                resume = summary_result.resume
        except Exception as e:
            logger.error(f"Erreur resume : {e}")

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
