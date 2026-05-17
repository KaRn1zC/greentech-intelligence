"""Taches Celery du backend GreenTech Intelligence.

Cette tache encapsule le pipeline complet d'analyse IA d'un article :

1. Extraction du contenu (URL, texte brut, ou fichier deja extrait)
2. Insertion en base (ou recuperation si URL deja connue)
3. Resume de classification via LLM (feature d'entree du classifieur)
4. Classification Qwen3-4B + LoRA TIES sur `titre + resume`
5. Resume ecologique si l'article est confirme Green IT
6. Log d'analyse dans la table ``analysis_logs``
7. Retour d'un dict serialisable JSON consomme par l'endpoint
   ``GET /analyze/{job_id}`` via le result backend Redis.

Le pipeline est volontairement strictement sequentiel : le classifieur
attend le resume de classification (qui sert de feature d'entree), et le
resume ecologique attend la classification (genere uniquement si Green IT).

Le pool ``solo`` du worker garantit qu'une seule tache classify_article
s'execute a la fois, ce qui evite l'OOM VRAM sur le GPU (Qwen3-4B occupe
~7.7 GB par instance chargee).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy import select

from greentech.api.celery_app import celery_app
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import AnalysisLog, Article


class _WorkerLoop:
    """Gestionnaire d'event loop persistant pour les taches Celery async.

    ``asyncio.run()`` ferme le loop a chaque appel. Cela invalide les
    connexions asyncpg cachees dans le pool SQLAlchemy (qui sont liees au
    loop). Au deuxieme appel d'``asyncio.run()``, le nouveau loop essaye
    d'utiliser ces connexions mortes et plante avec :
    ``AttributeError: 'NoneType' object has no attribute 'send'`` (proactor
    event loop ferme).

    En gardant un loop persistant par process worker et en le reutilisant
    via ``run_until_complete``, le pool de connexions reste valide entre
    les taches. C'est l'approche recommandee par la documentation Celery
    pour mixer async/await avec un broker synchrone.

    Le loop est lazy-init au premier appel, et recree automatiquement
    s'il a ete ferme (defense en profondeur en cas d'exception).
    """

    _loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def get(cls) -> asyncio.AbstractEventLoop:
        """Retourne le loop persistant, le cree s'il n'existe pas."""
        if cls._loop is None or cls._loop.is_closed():
            cls._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls._loop)
        return cls._loop


async def _run_analysis_pipeline(
    url: str | None,
    texte: str | None,
    titre_override: str | None,
    job_id: str,
) -> dict[str, Any]:
    """Execute la chaine complete d'analyse IA et retourne un dict serialisable.

    Cette fonction est volontairement decoupplee de Celery (pas de decorateur)
    pour rester testable en pur asyncio. La tache Celery ``classify_article``
    l'appelle via ``asyncio.run()``.

    Args:
        url: URL de l'article a analyser (mutuellement exclusif avec texte).
        texte: Texte brut a analyser (mutuellement exclusif avec url).
        titre_override: Titre force (typiquement nom de fichier uploade).
            Si None, le titre est deduit du contenu ou de l'URL.
        job_id: Identifiant de la tache Celery, utilise pour generer une URL
            unique quand l'analyse porte sur du texte brut (pas d'URL source).

    Returns:
        Dict serialisable JSON avec : ``id_article``, ``titre``, ``est_green_it``,
        ``score_confiance``, ``resume``, ``resume_ecologique``,
        ``modele_classification``, ``temps_inference_ms``, ``date_analyse``.

    Raises:
        ValueError: Si le contenu est trop court pour etre analyse (<50 chars).
        Exception: Toute erreur d'extraction ou d'inference est propagee a
            Celery qui marquera la tache en FAILURE et stockera l'erreur dans
            le result backend.
    """
    from greentech.api.routes.analyze import _extract_text_from_url

    # 1. Extraction du contenu
    if url:
        titre, contenu = await _extract_text_from_url(url)
    else:
        titre = texte[:100] + "..." if texte and len(texte) > 100 else (texte or "")
        contenu = texte or ""

    if titre_override:
        titre = titre_override

    if len(contenu) < 50:
        msg = "Contenu insuffisant pour l'analyse (minimum 50 caracteres)"
        raise ValueError(msg)

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

    logger.info(f"Analyse {job_id} terminee : article_id={id_article}")

    return {
        "id_article": id_article,
        "titre": titre[:500],
        "est_green_it": est_green_it,
        "score_confiance": score_confiance,
        "resume": resume,
        "resume_ecologique": resume_ecologique,
        "modele_classification": modele_nom,
        "temps_inference_ms": temps_ms,
        "date_analyse": datetime.now(UTC).isoformat(),
    }


@celery_app.task(
    bind=True,
    name="greentech.api.tasks.classify_article",
    autoretry_for=(),
    retry_kwargs={"max_retries": 0},
    acks_late=True,
)
def classify_article(
    self,
    url: str | None = None,
    texte: str | None = None,
    titre_override: str | None = None,
) -> dict[str, Any]:
    """Tache Celery d'analyse complete d'un article.

    Le worker recupere cette tache de la queue Redis, execute le pipeline
    async via ``asyncio.run()`` et stocke le resultat dans le result backend.
    L'API peut alors lire le statut/resultat via ``AsyncResult(task_id)``.

    Notes :

    * **acks_late=True** : la tache n'est ack'ee qu'apres execution complete.
      Si le worker crash en cours d'execution, la tache est re-enqueue.
    * **autoretry_for=()** : pas de retry automatique sur exception car nos
      erreurs sont generalement persistantes (URL invalide, contenu trop
      court) et un retry serait du gachis. Le client peut renvoyer la
      requete s'il pense que c'est transitoire.
    * **bind=True** : donne acces a ``self.request.id`` (job_id Celery) pour
      tracer dans les logs.

    Args:
        url: URL de l'article a analyser.
        texte: Texte brut a analyser (alternatif a url).
        titre_override: Titre force (typiquement nom de fichier uploade).

    Returns:
        Dict serialisable JSON (cf. ``_run_analysis_pipeline``).

    Raises:
        ValueError: Contenu trop court.
        Exception: Erreur d'extraction ou d'inference (marque la tache en
            FAILURE cote Celery, l'erreur est consultable via AsyncResult).
    """
    job_id = self.request.id
    source = f"URL={url}" if url else f"texte ({len(texte or '')} chars)"
    logger.info(f"Tache Celery demarree {job_id} : {source}")

    loop = _WorkerLoop.get()
    try:
        return loop.run_until_complete(
            _run_analysis_pipeline(
                url=url,
                texte=texte,
                titre_override=titre_override,
                job_id=job_id,
            )
        )
    except Exception as exc:
        logger.exception(f"Echec tache Celery {job_id} : {exc}")
        raise
