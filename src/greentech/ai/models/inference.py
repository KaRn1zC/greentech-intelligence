"""Module d'inférence pour le modèle de classification Green IT.

Charge le modèle gagnant (sélectionné après le benchmark Champion vs Challengers)
et fournit une interface simple pour classifier les articles en production.
Met à jour la base PostgreSQL avec les résultats de classification.

Le modèle de production est un lien symbolique ou une copie du vainqueur
du benchmark (DeBERTa, Qwen+LoRA, ou Llama+LoRA) dans models/production/.

Rédigé par KaRn1zC - 2026-04-10
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from loguru import logger
from sqlalchemy import select, update

from greentech.ai.models.classifier import PredictionResult, TrainingConfig
from greentech.ai.models.training import ChampionClassifier
from greentech.config import BASE_DIR
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

# Chemin par défaut du modèle de production
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "production"

# Instance globale du classifieur (lazy loading)
_classifier: ChampionClassifier | None = None


async def get_classifier(model_path: Path | None = None) -> ChampionClassifier:
    """Retourne le classifieur de production (singleton lazy-loaded).

    Charge le modèle depuis le dossier de production au premier appel,
    puis réutilise l'instance pour les appels suivants.

    Args:
        model_path: Chemin vers le modèle (défaut: models/production).

    Returns:
        Instance du classifieur prête pour l'inférence.

    Raises:
        FileNotFoundError: Si le modèle n'est pas trouvé.
    """
    global _classifier  # noqa: PLW0603

    if _classifier is not None:
        return _classifier

    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        msg = (
            f"Modèle de production introuvable : {path}. "
            "Lancez l'entraînement avec : uv run python -m greentech.ai.models.training"
        )
        raise FileNotFoundError(msg)

    config = TrainingConfig(nom_modele=str(path))
    _classifier = ChampionClassifier(config)
    _classifier.load(path)
    logger.info(f"Modèle de production chargé depuis {path}")
    return _classifier


async def classify_article(article_id: int) -> PredictionResult:
    """Classifie un article et stocke le résultat en base.

    Lit l'article depuis PostgreSQL, exécute l'inférence, puis met à jour
    les colonnes `est_green_it`, `score_confiance`, `modele_classification`
    et `date_analyse`.

    Args:
        article_id: Identifiant de l'article en base.

    Returns:
        Résultat de la classification.

    Raises:
        ValueError: Si l'article n'existe pas ou n'a pas de contenu.
    """
    classifier = await get_classifier()

    async with async_session_factory() as session:
        stmt = select(Article).where(Article.id_article == article_id)
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()

        if article is None:
            msg = f"Article id={article_id} introuvable"
            raise ValueError(msg)

        if not article.contenu:
            msg = f"Article id={article_id} sans contenu"
            raise ValueError(msg)

        # Inférence
        prediction = await classifier.predict(article.contenu)

        # Mise à jour en base
        from datetime import datetime

        stmt_update = (
            update(Article)
            .where(Article.id_article == article_id)
            .values(
                est_green_it=prediction.est_green_it,
                score_confiance=prediction.score_confiance,
                modele_classification=prediction.modele,
                date_analyse=datetime.now(UTC),
            )
        )
        await session.execute(stmt_update)
        await session.commit()

        logger.info(
            f"Article id={article_id} classifié : "
            f"{'Green IT' if prediction.est_green_it else 'Non Green IT'} "
            f"(confiance={prediction.score_confiance:.2%}, {prediction.temps_ms}ms)"
        )

        return prediction


async def classify_batch(*, limit: int = 100, force: bool = False) -> list[PredictionResult]:
    """Classifie un lot d'articles non encore analysés.

    Args:
        limit: Nombre maximum d'articles à traiter.
        force: Si True, re-classifie aussi les articles déjà analysés.

    Returns:
        Liste des résultats de classification.
    """
    async with async_session_factory() as session:
        stmt = select(Article.id_article).where(Article.contenu.isnot(None))
        if not force:
            stmt = stmt.where(Article.est_green_it.is_(None))
        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        article_ids = [row[0] for row in result.all()]

    if not article_ids:
        logger.info("Aucun article à classifier")
        return []

    logger.info(f"Classification de {len(article_ids)} articles...")
    results = []

    for article_id in article_ids:
        prediction = await classify_article(article_id)
        results.append(prediction)

    green = sum(1 for r in results if r.est_green_it)
    non_green = len(results) - green
    logger.info(f"Batch terminé : {green} Green IT / {non_green} Non Green IT")

    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(classify_batch())
