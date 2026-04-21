"""Exporte le golden dataset depuis la DB post-classification.

Ce script produit ``data/golden_dataset.csv`` a partir de l'etat final de
la base de donnees (colonne ``articles.est_green_it``), apres que le
pipeline de classification hybride a tranche pour chaque article.

Depuis la bascule du classifieur sur le resume de classification
(``articles.resume``) au lieu du contenu tronque, le CSV contient la
feature ``resume_classification`` : c'est elle qui est concatenee au
titre et envoyee au classifieur Qwen3-4B + LoRA pendant l'entrainement.
Les articles sans resume sont exclus car ils ne peuvent servir ni a
l'entrainement ni a l'evaluation.

Le fichier genere sert de source de verite pour :

- l'entrainement Qwen3-4B + LoRA (``step_train`` et ``step_train_cv``)
- les benchmarks de promotion du modele (``step_benchmark``,
  ``step_auto_promote``)
- le versioning DVC du dataset

Filtrage
--------

Deux criteres cumulatifs d'exclusion :

1. Articles sans decision (``est_green_it IS NULL``) : pas de label, donc
   inutilisables pour l'entrainement supervise.
2. Articles sans resume (``resume IS NULL``) : pas de feature d'entree,
   generer le resume via ``scripts/generate_classification_summaries.py``
   avant relance.

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
    # Nouvelle feature d'entrainement : le resume uniforme produit par le LLM
    # de classification. Remplace l'ancien `contenu_extrait` qui n'etait que
    # les 500 premiers caracteres du contenu brut et creait une incoherence
    # train/inference (le modele voyait 500 chars a l'entrainement mais
    # jusqu'a 2000 chars a l'inference).
    "resume_classification",
    "source_nom",
    "date_publication",
    # Langue detectee par le pipeline de cleaning (colonne articles.langue).
    # Ajoutee en avril 2026 pour supporter la stratification croisee
    # (langue x label) lors du K-fold et la back-translation ciblee
    # EN<->FR via opus-mt sur les positifs.
    "langue",
    "label_green_it",
    "score_confiance",
    "modele_classification",
]


async def export_from_db() -> dict[str, int]:
    """Exporte les articles classifies et resumes depuis PostgreSQL vers le CSV golden.

    Returns:
        Statistiques ``{total, green_it, non_green_it, exclus_sans_resume}``.
    """
    stats = {"total": 0, "green_it": 0, "non_green_it": 0, "exclus_sans_resume": 0}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        # On compte les articles classifies mais sans resume pour alerter
        # l'utilisateur, puis on les exclut de l'export.
        sans_resume_stmt = (
            select(Article.id_article)
            .where(Article.contenu.isnot(None))
            .where(Article.est_green_it.isnot(None))
            .where(Article.resume.is_(None))
        )
        sans_resume_result = await session.execute(sans_resume_stmt)
        stats["exclus_sans_resume"] = len(sans_resume_result.all())

        stmt = (
            select(Article, Source.nom)
            .outerjoin(Source, Article.id_source == Source.id_source)
            .where(Article.contenu.isnot(None))
            .where(Article.est_green_it.isnot(None))
            .where(Article.resume.isnot(None))
            .order_by(Article.id_article)
        )
        result = await session.execute(stmt)
        rows = result.all()

    if stats["exclus_sans_resume"] > 0:
        logger.warning(
            f"{stats['exclus_sans_resume']} article(s) classifie(s) mais sans resume "
            "exclu(s) de l'export. Lancer 'uv run python "
            "scripts/generate_classification_summaries.py' puis relancer l'export."
        )

    if not rows:
        logger.warning(
            "Aucun article a exporter - verifier que le pipeline de classification "
            "et de resume a bien tourne sur l'integralite du dataset"
        )
        return stats

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for article, source_nom in rows:
            label = 1 if article.est_green_it else 0
            writer.writerow(
                {
                    "id_article": article.id_article,
                    "titre": article.titre,
                    "url": article.url,
                    "resume_classification": article.resume,
                    "source_nom": source_nom or "Inconnue",
                    "date_publication": (
                        article.date_publication.isoformat()
                        if article.date_publication
                        else ""
                    ),
                    "langue": article.langue or "",
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
    logger.info(f"  Total articles exportes : {stats['total']}")
    logger.info(f"  Green IT (label=1)      : {stats['green_it']} ({pct_green:.2f}%)")
    logger.info(f"  Non Green IT            : {stats['non_green_it']}")
    logger.info(f"  Exclus (sans resume)    : {stats['exclus_sans_resume']}")
    return stats


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(export_from_db())
