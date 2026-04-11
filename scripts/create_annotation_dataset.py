"""Génère un fichier CSV d'annotation pour la création du Golden Dataset.

Extrait les articles depuis PostgreSQL et crée un CSV avec les colonnes
nécessaires à l'annotation manuelle (labeling Green IT / Non Green IT).
Le fichier est ensuite versionné via DVC.

"""

from __future__ import annotations

import asyncio
import csv

from loguru import logger
from sqlalchemy import func, select

from greentech.config import BASE_DIR, get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "articles_a_annoter.csv"

# Colonnes du CSV d'annotation
CSV_HEADERS = [
    "id_article",
    "titre",
    "url",
    "contenu_extrait",
    "source_nom",
    "date_publication",
    "label_green_it",
]

# Longueur max de l'extrait de contenu (pour faciliter la lecture lors de l'annotation)
EXTRAIT_MAX_CHARS = 500


async def export_articles_for_annotation() -> int:
    """Exporte les articles depuis PostgreSQL vers un CSV d'annotation.

    Sélectionne tous les articles avec du contenu et génère un fichier CSV
    avec une colonne vide `label_green_it` à remplir manuellement (0 ou 1).

    Returns:
        Nombre d'articles exportés.
    """
    settings = get_settings()
    logger.info(f"Connexion à {settings.postgres_host}:{settings.postgres_port}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        # Compter les articles disponibles
        count_stmt = select(func.count(Article.id_article)).where(Article.contenu.isnot(None))
        total = (await session.execute(count_stmt)).scalar_one()
        logger.info(f"{total} articles avec contenu trouvés en base")

        # Récupérer les articles avec leur source
        stmt = (
            select(Article, Source.nom)
            .outerjoin(Source, Article.id_source == Source.id_source)
            .where(Article.contenu.isnot(None))
            .order_by(Article.id_article)
        )
        result = await session.execute(stmt)
        rows = result.all()

        # Écrire le CSV
        with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()

            for article, source_nom in rows:
                extrait = (article.contenu or "")[:EXTRAIT_MAX_CHARS]
                if len(article.contenu or "") > EXTRAIT_MAX_CHARS:
                    extrait += "..."

                writer.writerow(
                    {
                        "id_article": article.id_article,
                        "titre": article.titre,
                        "url": article.url,
                        "contenu_extrait": extrait,
                        "source_nom": source_nom or "Inconnue",
                        "date_publication": (
                            article.date_publication.isoformat() if article.date_publication else ""
                        ),
                        "label_green_it": "",
                    }
                )

    logger.info(f"{len(rows)} articles exportés vers {OUTPUT_FILE}")
    logger.info(
        "Instruction : ouvrir le CSV, remplir la colonne 'label_green_it' "
        "avec 1 (Green IT) ou 0 (Non Green IT), puis sauvegarder."
    )
    return len(rows)


async def import_annotated_dataset() -> int:
    """Importe les labels du CSV annoté vers PostgreSQL.

    Lit le fichier CSV annoté et met à jour la colonne `est_green_it`
    des articles correspondants en base de données.

    Returns:
        Nombre d'articles mis à jour.
    """
    annotated_file = OUTPUT_DIR / "golden_dataset.csv"
    if not annotated_file.exists():
        logger.error(f"Fichier annoté introuvable : {annotated_file}")
        logger.info(
            "Étapes : 1) Exécuter ce script en mode export, "
            "2) Annoter le CSV, 3) Renommer en golden_dataset.csv"
        )
        return 0

    updated = 0
    async with async_session_factory() as session:
        with annotated_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = row.get("label_green_it", "").strip()
                if label not in ("0", "1"):
                    continue

                article_id = int(row["id_article"])
                is_green = label == "1"

                from sqlalchemy import update

                stmt = (
                    update(Article)
                    .where(Article.id_article == article_id)
                    .values(est_green_it=is_green)
                )
                result = await session.execute(stmt)
                if result.rowcount > 0:
                    updated += 1

            await session.commit()

    logger.info(f"{updated} articles mis à jour avec les labels Green IT")
    return updated


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "import":
        count = asyncio.run(import_annotated_dataset())
        logger.info(f"Import terminé : {count} labels appliqués")
    else:
        count = asyncio.run(export_articles_for_annotation())
        logger.info(f"Export terminé : {count} articles à annoter dans {OUTPUT_FILE}")
