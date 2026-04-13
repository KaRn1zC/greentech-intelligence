"""Module 3 : Ingestion de fichiers (dataset arXiv depuis Kaggle).

Lit un dataset JSON volumineux (arXiv Metadata, ~3.6 Go) ligne par ligne,
filtre les publications pertinentes (catégorie cs.AI) et upload
les données brutes vers MinIO raw-data.

"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# Catégories arXiv pertinentes pour le Green IT / IA
RELEVANT_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE"}

# Nombre maximum d'articles à ingérer par batch
BATCH_SIZE = 500

# Nombre maximum d'articles à traiter au total
MAX_ARTICLES = 5000


class FileIngester(BaseCollector):
    """Collecteur de données depuis un fichier JSON local.

    Lit le dataset arXiv (format JSON Lines) téléchargé depuis Kaggle,
    filtre les publications par catégorie et les upload vers MinIO.
    """

    def __init__(self) -> None:
        super().__init__(source_name="arxiv")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        *,
        file_path: str | Path | None = None,
        max_articles: int = MAX_ARTICLES,
        **kwargs: Any,
    ) -> CollectResult:
        """Ingère les articles depuis le fichier JSON local.

        Le fichier est lu ligne par ligne (JSON Lines) pour éviter
        de charger l'intégralité en mémoire (fichier de ~3.6 Go).

        Args:
            keywords: Mots-clés pour le filtrage additionnel des abstracts.
            file_path: Chemin vers le fichier JSON (arXiv metadata).
            max_articles: Nombre maximum d'articles à ingérer.
            **kwargs: Paramètres additionnels.

        Returns:
            Résultat de la collecte.
        """
        result = CollectResult(source_name=self.source_name)

        if file_path is None:
            msg = "Chemin du fichier arXiv non spécifié (argument file_path requis)"
            logger.error(msg)
            result.errors.append(msg)
            return result

        path = Path(file_path)
        if not path.exists():
            msg = f"Fichier introuvable : {path}"
            logger.error(msg)
            result.errors.append(msg)
            return result

        logger.info(f"Lecture du fichier : {path} (max {max_articles} articles)")

        try:
            articles = self._read_and_filter(path, keywords, max_articles)

            if not articles:
                logger.warning("Aucun article pertinent trouvé dans le fichier")
                return result

            # Upload par batchs vers MinIO
            for i in range(0, len(articles), BATCH_SIZE):
                batch = articles[i : i + BATCH_SIZE]
                batch_num = i // BATCH_SIZE + 1
                raw_path = generate_raw_path("file", f"arxiv_batch_{batch_num:03d}")

                payload = {
                    "source": "arxiv_dataset",
                    "batch_number": batch_num,
                    "articles_count": len(batch),
                    "articles": batch,
                }
                minio_path = await upload_json_to_minio(
                    payload,
                    bucket=self.settings.minio_bucket_raw,
                    object_name=raw_path,
                )
                result.raw_paths.append(minio_path)

                logger.info(f"Batch {batch_num} : {len(batch)} articles -> {minio_path}")

            result.articles_count = len(articles)
            logger.info(
                f"Ingestion terminée : {len(articles)} articles en {len(result.raw_paths)} batchs"
            )

        except Exception as e:
            error_msg = f"Erreur ingestion fichier : {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def _read_and_filter(
        self,
        file_path: Path,
        keywords: list[str],
        max_articles: int,
    ) -> list[dict[str, Any]]:
        """Lit le fichier JSON Lines et filtre les articles pertinents.

        Le fichier arXiv est au format JSON Lines (une entrée JSON par ligne),
        ce qui permet une lecture incrémentale sans charger tout en mémoire.

        Args:
            file_path: Chemin du fichier.
            keywords: Mots-clés pour filtrer les abstracts.
            max_articles: Limite du nombre d'articles.

        Returns:
            Articles filtrés et normalisés.
        """
        articles: list[dict[str, Any]] = []
        total_read = 0
        total_filtered = 0

        keywords_lower = [kw.lower() for kw in keywords] if keywords else []

        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if len(articles) >= max_articles:
                    break

                total_read += 1
                if total_read % 100_000 == 0:
                    logger.debug(
                        f"Progression : {total_read} lignes lues, {len(articles)} articles retenus"
                    )

                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Filtrage par catégorie arXiv
                categories = set(entry.get("categories", "").split())
                if not categories & RELEVANT_CATEGORIES:
                    total_filtered += 1
                    continue

                # Filtrage par mots-clés dans le titre/abstract
                if keywords_lower:
                    titre = (entry.get("title") or "").lower()
                    abstract = (entry.get("abstract") or "").lower()
                    texte = f"{titre} {abstract}"
                    if not any(kw in texte for kw in keywords_lower):
                        total_filtered += 1
                        continue

                # Normalisation
                articles.append(
                    {
                        "titre": entry.get("title", "").strip().replace("\n", " "),
                        "url": f"https://arxiv.org/abs/{entry.get('id', '')}",
                        "contenu": entry.get("abstract", "").strip().replace("\n", " "),
                        "auteur": entry.get("authors", ""),
                        "date_publication": entry.get("update_date"),
                        "source_nom": "arXiv Dataset",
                        "categories": entry.get("categories", ""),
                        "arxiv_id": entry.get("id"),
                    }
                )

        logger.info(
            f"Fichier lu : {total_read} lignes, {total_filtered} filtrées, {len(articles)} retenues"
        )
        return articles


async def run_file_ingestion(file_path: str | Path) -> CollectResult:
    """Point d'entrée principal pour l'ingestion de fichiers.

    Args:
        file_path: Chemin vers le fichier arXiv metadata JSON.

    Returns:
        Résultat de l'ingestion.
    """
    logger.info("=== Démarrage ingestion fichier (arXiv Dataset) ===")

    ingester = FileIngester()
    return await ingester.collect(keywords=[], file_path=file_path)


if __name__ == "__main__":
    import sys

    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)

    if len(sys.argv) < 2:
        logger.error("Usage : python -m greentech.data.collectors.file_ingester <chemin_fichier>")
        sys.exit(1)

    asyncio.run(run_file_ingestion(sys.argv[1]))
