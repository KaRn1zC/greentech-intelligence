"""Service de résumé automatique via Hugging Face Serverless Inference API.

Connecte l'application à l'API d'inférence Hugging Face pour générer
des résumés d'articles technologiques. Le résumé est ensuite stocké
dans la colonne `resume` de la table `articles` en base PostgreSQL.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from huggingface_hub import AsyncInferenceClient
from loguru import logger
from sqlalchemy import select, update

from greentech.config import get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Limites du modèle BART-large-CNN (en caractères, approximation conservative)
MAX_INPUT_CHARS = 3000
MIN_INPUT_CHARS = 50
# Paramètres de génération
MAX_SUMMARY_LENGTH = 150
MIN_SUMMARY_LENGTH = 30


@dataclass(frozen=True)
class SummaryResult:
    """Résultat d'un appel de résumé sur un article.

    Attributes:
        id_article: Identifiant de l'article traité.
        resume: Texte du résumé généré.
        temps_ms: Temps d'inférence en millisecondes.
        modele: Nom du modèle utilisé.
        succes: Indique si le résumé a été généré avec succès.
        erreur: Message d'erreur éventuel.
    """

    id_article: int
    resume: str | None
    temps_ms: int
    modele: str
    succes: bool
    erreur: str | None = None


def _get_client() -> AsyncInferenceClient:
    """Instancie le client d'inférence Hugging Face.

    Returns:
        Client asynchrone configuré avec le token du projet.

    Raises:
        ValueError: Si le token Hugging Face n'est pas configuré.
    """
    settings = get_settings()
    if not settings.huggingface_token:
        msg = (
            "HUGGINGFACE_TOKEN non configuré. "
            "Ajoutez-le dans le fichier .env pour utiliser l'API d'inférence."
        )
        raise ValueError(msg)

    return AsyncInferenceClient(
        model=settings.huggingface_model_summarizer,
        token=settings.huggingface_token,
    )


def _truncate_text(text: str) -> str:
    """Tronque le texte d'entrée à la taille maximale du modèle.

    BART-large-CNN accepte environ 1024 tokens. On tronque en caractères
    avec une marge de sécurité, en coupant à la dernière phrase complète.

    Args:
        text: Texte brut de l'article.

    Returns:
        Texte tronqué prêt pour l'inférence.
    """
    if len(text) <= MAX_INPUT_CHARS:
        return text

    truncated = text[:MAX_INPUT_CHARS]
    # Couper à la dernière phrase complète (point, point d'exclamation, etc.)
    last_period = truncated.rfind(".")
    if last_period > MAX_INPUT_CHARS // 2:
        truncated = truncated[: last_period + 1]

    logger.debug(f"Texte tronqué : {len(text)} → {len(truncated)} caractères")
    return truncated


async def summarize_text(text: str) -> SummaryResult:
    """Génère un résumé pour un texte donné via l'API Hugging Face.

    Envoie le texte au modèle de résumé configuré (par défaut BART-large-CNN)
    et retourne le résumé généré avec les métriques de performance.

    Args:
        text: Texte de l'article à résumer.

    Returns:
        Résultat contenant le résumé et les métriques d'inférence.
    """
    settings = get_settings()
    modele = settings.huggingface_model_summarizer

    if not text or len(text.strip()) < MIN_INPUT_CHARS:
        logger.warning(f"Texte trop court pour le résumé ({len(text)} chars)")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=0,
            modele=modele,
            succes=False,
            erreur="Texte trop court pour générer un résumé",
        )

    truncated = _truncate_text(text.strip())
    client = _get_client()

    start = time.perf_counter()
    try:
        result = await client.summarization(
            truncated,
            parameters={
                "max_length": MAX_SUMMARY_LENGTH,
                "min_length": MIN_SUMMARY_LENGTH,
            },
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        summary = result.summary_text if hasattr(result, "summary_text") else str(result)
        logger.info(f"Résumé généré en {elapsed_ms}ms ({len(summary)} chars)")

        return SummaryResult(
            id_article=0,
            resume=summary,
            temps_ms=elapsed_ms,
            modele=modele,
            succes=True,
        )

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.error(f"Échec résumé ({elapsed_ms}ms) : {exc}")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=elapsed_ms,
            modele=modele,
            succes=False,
            erreur=str(exc),
        )


async def summarize_article(
    article_id: int, *, session: AsyncSession | None = None
) -> SummaryResult:
    """Résume un article depuis la base de données et stocke le résultat.

    Lit le contenu de l'article, appelle l'API de résumé, puis met à jour
    la colonne `resume` dans la table `articles`.

    Args:
        article_id: Identifiant de l'article en base.
        session: Session SQLAlchemy optionnelle (créée automatiquement si absente).

    Returns:
        Résultat du résumé avec les métriques.

    Raises:
        ValueError: Si l'article n'existe pas ou n'a pas de contenu.
    """
    own_session = session is None
    if own_session:
        session = async_session_factory()

    try:
        # Récupérer l'article
        stmt = select(Article).where(Article.id_article == article_id)
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()

        if article is None:
            msg = f"Article id={article_id} introuvable"
            raise ValueError(msg)

        if not article.contenu:
            logger.warning(f"Article id={article_id} sans contenu, résumé impossible")
            return SummaryResult(
                id_article=article_id,
                resume=None,
                temps_ms=0,
                modele=get_settings().huggingface_model_summarizer,
                succes=False,
                erreur="Article sans contenu",
            )

        # Appeler l'API de résumé
        summary_result = await summarize_text(article.contenu)
        summary_result = SummaryResult(
            id_article=article_id,
            resume=summary_result.resume,
            temps_ms=summary_result.temps_ms,
            modele=summary_result.modele,
            succes=summary_result.succes,
            erreur=summary_result.erreur,
        )

        # Stocker le résumé en base si succès
        if summary_result.succes and summary_result.resume:
            stmt_update = (
                update(Article)
                .where(Article.id_article == article_id)
                .values(resume=summary_result.resume)
            )
            await session.execute(stmt_update)
            await session.commit()
            logger.info(
                f"Article id={article_id} : résumé stocké en base "
                f"({len(summary_result.resume)} chars)"
            )

        return summary_result

    except Exception:
        if own_session:
            await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()


async def summarize_batch(
    *,
    limit: int = 50,
    force: bool = False,
) -> list[SummaryResult]:
    """Résume un lot d'articles qui n'ont pas encore de résumé.

    Sélectionne les articles sans résumé (colonne `resume` IS NULL)
    et les traite séquentiellement pour respecter les limites de l'API.

    Args:
        limit: Nombre maximum d'articles à traiter.
        force: Si True, re-résume aussi les articles avec un résumé existant.

    Returns:
        Liste des résultats de résumé pour chaque article traité.
    """
    async with async_session_factory() as session:
        # Sélectionner les articles à résumer
        stmt = select(Article).where(Article.contenu.isnot(None))
        if not force:
            stmt = stmt.where(Article.resume.is_(None))
        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        articles = result.scalars().all()

        if not articles:
            logger.info("Aucun article à résumer")
            return []

        logger.info(f"Résumé de {len(articles)} articles en cours...")
        results: list[SummaryResult] = []

        for article in articles:
            summary = await summarize_article(article.id_article, session=session)
            results.append(summary)

        # Bilan
        succes = sum(1 for r in results if r.succes)
        echecs = len(results) - succes
        temps_total = sum(r.temps_ms for r in results)

        logger.info(
            f"Batch terminé : {succes} résumés / {echecs} échecs "
            f"({temps_total}ms total, ~{temps_total // max(len(results), 1)}ms/article)"
        )

        return results


async def run_summarizer() -> None:
    """Point d'entrée principal pour le résumé en lot.

    Traite tous les articles sans résumé en base. Utilisé en ligne
    de commande via `uv run python -m greentech.ai.services.summarizer`.
    """
    logger.info("Démarrage du service de résumé automatique")
    settings = get_settings()
    logger.info(f"Modèle : {settings.huggingface_model_summarizer}")

    results = await summarize_batch()

    if results:
        succes = sum(1 for r in results if r.succes)
        logger.info(f"Résumé terminé : {succes}/{len(results)} articles traités")
    else:
        logger.info("Aucun article en attente de résumé")


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_summarizer())
