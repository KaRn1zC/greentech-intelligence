"""Exporte le golden dataset depuis la DB post-classification.

Ce script produit `data/golden_dataset.csv` a partir de l'etat final de la
base de donnees (colonne `articles.est_green_it`), apres que le pipeline
de classification hybride a tranche pour chaque article.

Le fichier genere sert de source de verite pour :

- l'entrainement Llama 3.2 3B + LoRA (`step_train` et `step_train_cv`)
- les benchmarks de promotion du modele (`step_benchmark`, `step_auto_promote`)
- le versioning DVC du dataset

Filtrage
--------

Seuls les articles ayant une decision finale (`est_green_it IS NOT NULL`)
sont exportes. Les articles encore en attente de verification LLM
(`modele_classification = 'keyword_filter'` avec `est_green_it IS NULL`)
sont ignores : ils seront inclus au prochain run du pipeline apres
l'etage 2.

Usage
-----

    uv run python scripts/export_golden_dataset.py

"""

from __future__ import annotations

import asyncio
import csv

from loguru import logger
from sqlalchemy import select

from greentech.config import BASE_DIR
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "golden_dataset.csv"

CSV_HEADERS = [
    "id_article",
    "titre",
    "url",
    "contenu_extrait",
    "source_nom",
    "date_publication",
    "label_green_it",
    "score_confiance",
    "modele_classification",
]

EXTRAIT_MAX_CHARS = 500


async def export_from_db() -> dict[str, int]:
    """Exporte les articles classifies depuis PostgreSQL vers le CSV golden.

    Returns:
        Statistiques {total, green_it, non_green_it}.
    """
    stats = {"total": 0, "green_it": 0, "non_green_it": 0}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        stmt = (
            select(Article, Source.nom)
            .outerjoin(Source, Article.id_source == Source.id_source)
            .where(Article.contenu.isnot(None))
            .where(Article.est_green_it.isnot(None))
            .order_by(Article.id_article)
        )
        result = await session.execute(stmt)
        rows = result.all()

    if not rows:
        logger.warning(
            "Aucun article classifie trouve en base - "
            "lancer le pre-filtre et le LLM judge avant l'export"
        )
        return stats

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for article, source_nom in rows:
            extrait = (article.contenu or "")[:EXTRAIT_MAX_CHARS]
            if len(article.contenu or "") > EXTRAIT_MAX_CHARS:
                extrait += "..."

            label = 1 if article.est_green_it else 0
            writer.writerow(
                {
                    "id_article": article.id_article,
                    "titre": article.titre,
                    "url": article.url,
                    "contenu_extrait": extrait,
                    "source_nom": source_nom or "Inconnue",
                    "date_publication": (
                        article.date_publication.isoformat()
                        if article.date_publication
                        else ""
                    ),
                    "label_green_it": label,
                    "score_confiance": (
                        f"{article.score_confiance:.3f}"
                        if article.score_confiance is not None
                        else ""
                    ),
                    "modele_classification": article.modele_classification or "",
                }
            )

            stats["total"] += 1
            if article.est_green_it:
                stats["green_it"] += 1
            else:
                stats["non_green_it"] += 1

    pct_green = (stats["green_it"] / stats["total"] * 100) if stats["total"] else 0
    logger.info(f"Export termine : {OUTPUT_FILE}")
    logger.info(f"  Total articles   : {stats['total']}")
    logger.info(f"  Green IT (label=1): {stats['green_it']} ({pct_green:.2f}%)")
    logger.info(f"  Non Green IT     : {stats['non_green_it']}")
    return stats


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(export_from_db())
