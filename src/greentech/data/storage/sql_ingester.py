"""Script d'ingestion des données nettoyées dans PostgreSQL.

Lit les fichiers Parquet depuis MinIO clean-data/articles/,
mappe les noms de sources vers la table sources, et insère
les articles dans PostgreSQL avec gestion des doublons
(upsert via ON CONFLICT sur l'URL unique).

Pipeline complet : MinIO clean-data (Parquet) → PostgreSQL (articles)

"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime
from typing import Any

import pyarrow.parquet as pq
from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.config import get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.minio_client import get_minio_client
from greentech.data.storage.models import Article, Source

# Taille des batchs pour les insertions (compromis mémoire/performance)
BATCH_SIZE = 50

# Mapping des noms de sources du Parquet vers les noms en base PostgreSQL.
# Les collecteurs produisent des valeurs variées pour source_nom :
#   - API NewsData.io : le source_id de l'API (ex: "bbc_news", "techradar")
#   - Scraping : "TechCrunch Climate" (fixe)
#   - Fichier arXiv : "arXiv Dataset" (fixe)
SOURCE_NAME_MAPPING: dict[str, str] = {
    "newsdata": "NewsData.io",
    "newsdata.io": "NewsData.io",
    "TechCrunch Climate": "TechCrunch Climate",
    "techcrunch": "TechCrunch Climate",
    "arXiv Dataset": "arXiv Dataset",
    "arxiv": "arXiv Dataset",
    "arxiv_dataset": "arXiv Dataset",
}


def _resolve_source_name(raw_name: str | None) -> str:
    """Résout le nom canonique de la source depuis la valeur brute du Parquet.

    Le champ source_nom dans les données nettoyées peut contenir
    soit un nom canonique (ex: "TechCrunch Climate"), soit un identifiant
    spécifique à l'API (ex: "bbc_news" depuis NewsData.io).

    Args:
        raw_name: Valeur brute du champ source_nom dans le Parquet.

    Returns:
        Nom canonique de la source tel que défini dans la table sources.
    """
    if not raw_name:
        return "NewsData.io"

    # Correspondance exacte
    if raw_name in SOURCE_NAME_MAPPING:
        return SOURCE_NAME_MAPPING[raw_name]

    # Correspondance insensible à la casse
    raw_lower = raw_name.lower()
    for key, value in SOURCE_NAME_MAPPING.items():
        if key.lower() == raw_lower:
            return value

    # Heuristique : les source_id inconnus de l'API NewsData.io
    # sont des identifiants de médias (ex: "bbc_news", "techradar")
    # qui transitent tous par le pipeline NewsData.io
    if "techcrunch" in raw_lower:
        return "TechCrunch Climate"
    if "arxiv" in raw_lower:
        return "arXiv Dataset"

    # Par défaut, un source_nom inconnu provient de NewsData.io
    return "NewsData.io"


def _parse_iso_date(date_str: str | None) -> datetime | None:
    """Parse une date ISO 8601 en objet datetime timezone-aware.

    Gère les formats produits par le pipeline Spark :
    - "2026-01-15T10:30:00Z"
    - "2026-01-15T10:30:00+00:00"
    - "2026-01-15T00:00:00Z" (date seule convertie par Spark)

    Args:
        date_str: Chaîne de date au format ISO 8601.

    Returns:
        Objet datetime ou None si la date n'est pas parsable.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = date_str.strip()
    if not date_str:
        return None

    try:
        normalized = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except (ValueError, AttributeError):
        logger.debug(f"Date non parsable ignorée : '{date_str}'")
        return None


# ---------------------------------------------------------------------------
# Lecture depuis MinIO
# ---------------------------------------------------------------------------


def read_parquet_from_minio(bucket: str, prefix: str) -> list[dict[str, Any]]:
    """Lit les fichiers Parquet depuis un bucket MinIO et retourne les enregistrements.

    Spark sauvegarde les DataFrames en Parquet partitionné (part-*.parquet).
    Cette fonction télécharge chaque partition, la lit avec PyArrow
    et concatène les résultats en une liste de dictionnaires Python.

    Args:
        bucket: Nom du bucket MinIO (ex: "clean-data").
        prefix: Préfixe du chemin dans le bucket (ex: "articles").

    Returns:
        Liste de dictionnaires, un par article nettoyé.
    """
    client = get_minio_client()

    objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
    parquet_files = [obj for obj in objects if obj.object_name.endswith(".parquet")]

    if not parquet_files:
        logger.warning(f"Aucun fichier Parquet trouvé dans {bucket}/{prefix}")
        return []

    logger.info(f"{len(parquet_files)} fichier(s) Parquet trouvé(s) dans {bucket}/{prefix}")

    all_records: list[dict[str, Any]] = []

    for obj in parquet_files:
        response = client.get_object(bucket, obj.object_name)
        try:
            data = response.read()
        finally:
            response.close()
            response.release_conn()

        table = pq.read_table(io.BytesIO(data))
        columns = table.to_pydict()
        n_rows = table.num_rows

        for i in range(n_rows):
            row = {col: values[i] for col, values in columns.items()}
            all_records.append(row)

        logger.debug(f"  Partition {obj.object_name} : {n_rows} enregistrements")

    logger.info(f"Total articles lus depuis Parquet : {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Gestion des sources
# ---------------------------------------------------------------------------


async def _load_source_cache(session: AsyncSession) -> dict[str, int]:
    """Charge le mapping source_nom → id_source depuis PostgreSQL.

    Args:
        session: Session SQLAlchemy asynchrone.

    Returns:
        Dictionnaire {nom_source: id_source}.
    """
    result = await session.execute(select(Source.nom, Source.id_source))
    cache = {row[0]: row[1] for row in result.all()}
    logger.info(f"Cache sources chargé : {list(cache.keys())}")
    return cache


async def _ensure_source_exists(
    session: AsyncSession,
    source_cache: dict[str, int],
    source_nom: str,
) -> int | None:
    """Retourne l'id_source pour un nom donné, crée la source si nécessaire.

    Args:
        session: Session SQLAlchemy asynchrone.
        source_cache: Cache mutable {nom: id_source}.
        source_nom: Nom canonique de la source.

    Returns:
        Identifiant de la source en base.
    """
    if source_nom in source_cache:
        return source_cache[source_nom]

    # Créer automatiquement une source manquante
    new_source = Source(
        nom=source_nom,
        type="api",
        description="Source créée automatiquement lors de l'ingestion",
    )
    session.add(new_source)
    await session.flush()

    source_cache[source_nom] = new_source.id_source
    logger.info(f"Nouvelle source créée : '{source_nom}' (id={new_source.id_source})")
    return new_source.id_source


# ---------------------------------------------------------------------------
# Ingestion dans PostgreSQL
# ---------------------------------------------------------------------------


async def ingest_to_postgresql(records: list[dict[str, Any]]) -> dict[str, int]:
    """Insère les articles nettoyés dans PostgreSQL avec gestion des doublons.

    Utilise INSERT ... ON CONFLICT (url) DO NOTHING pour garantir
    l'idempotence : relancer l'ingestion ne crée pas de doublons.

    Les articles sont insérés par batchs pour optimiser les performances
    et limiter l'utilisation mémoire des transactions.

    Args:
        records: Liste de dictionnaires issus du Parquet nettoyé.

    Returns:
        Statistiques d'ingestion {inseres, ignores, erreurs}.
    """
    stats = {"inseres": 0, "ignores": 0, "erreurs": 0}
    sources_utilisees: set[str] = set()

    async with async_session_factory() as session:
        source_cache = await _load_source_cache(session)

        total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            for record in batch:
                try:
                    # Résolution de la source
                    raw_source = record.get("source_nom")
                    resolved_name = _resolve_source_name(raw_source)
                    id_source = await _ensure_source_exists(session, source_cache, resolved_name)
                    sources_utilisees.add(resolved_name)

                    # Préparation des valeurs de l'article
                    url = record.get("url") or ""
                    if not url:
                        stats["erreurs"] += 1
                        continue

                    titre = (record.get("titre") or "")[:500]
                    if not titre:
                        stats["erreurs"] += 1
                        continue

                    values: dict[str, Any] = {
                        "titre": titre,
                        "url": url,
                        "contenu": record.get("contenu"),
                        "auteur": (record.get("auteur") or "")[:200] or None,
                        "date_publication": _parse_iso_date(record.get("date_publication")),
                        "langue": (record.get("langue") or "en")[:10],
                        "id_source": id_source,
                        "chemin_donnees_brutes": "clean-data/articles",
                    }

                    # Upsert : INSERT ... ON CONFLICT (url) DO NOTHING
                    stmt = pg_insert(Article).values(**values)
                    stmt = stmt.on_conflict_do_nothing(index_elements=["url"])
                    result = await session.execute(stmt)

                    if result.rowcount > 0:
                        stats["inseres"] += 1
                    else:
                        stats["ignores"] += 1

                except Exception as e:
                    stats["erreurs"] += 1
                    logger.warning(f"Erreur insertion article : {e}")

            await session.commit()
            logger.info(
                f"Batch {batch_num}/{total_batches} traité "
                f"(insérés={stats['inseres']}, ignorés={stats['ignores']})"
            )

        # Mettre à jour la date de dernière collecte des sources utilisées
        for nom_source in sources_utilisees:
            await session.execute(
                update(Source).where(Source.nom == nom_source).values(derniere_collecte=func.now())
            )
        await session.commit()
        logger.info(f"Date de dernière collecte mise à jour pour : {sorted(sources_utilisees)}")

    return stats


# ---------------------------------------------------------------------------
# Vérification post-ingestion
# ---------------------------------------------------------------------------


async def verify_ingestion() -> dict[str, Any]:
    """Exécute des requêtes SQL de vérification après l'ingestion.

    Affiche les statistiques de la base pour confirmer que les données
    sont correctement insérées et structurées.

    Returns:
        Dictionnaire des statistiques de vérification.
    """
    async with async_session_factory() as session:
        # Total articles
        result = await session.execute(select(func.count(Article.id_article)))
        total = result.scalar() or 0

        # Articles par source (jointure sources ↔ articles)
        result = await session.execute(
            select(Source.nom, Source.type, func.count(Article.id_article))
            .outerjoin(Article, Article.id_source == Source.id_source)
            .group_by(Source.nom, Source.type)
            .order_by(func.count(Article.id_article).desc())
        )
        par_source = result.all()

        # Articles en attente d'analyse IA
        result = await session.execute(
            select(func.count(Article.id_article)).where(Article.est_green_it.is_(None))
        )
        en_attente = result.scalar() or 0

        # Articles avec date de publication
        result = await session.execute(
            select(func.count(Article.id_article)).where(Article.date_publication.isnot(None))
        )
        avec_date = result.scalar() or 0

    logger.info("=" * 50)
    logger.info("VÉRIFICATION POST-INGESTION")
    logger.info("=" * 50)
    logger.info(f"Total articles en base        : {total}")
    logger.info(f"Articles avec date publication : {avec_date}")
    logger.info(f"En attente d'analyse IA       : {en_attente}")
    logger.info("-" * 50)
    for nom, type_src, count in par_source:
        logger.info(f"  {nom} ({type_src}) : {count} articles")
    logger.info("=" * 50)

    return {
        "total": total,
        "en_attente_analyse": en_attente,
        "avec_date": avec_date,
        "par_source": {nom: count for nom, _, count in par_source},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def run_sql_ingestion() -> dict[str, int]:
    """Orchestre l'ingestion complète des données nettoyées vers PostgreSQL.

    Pipeline :
    1. Lecture des fichiers Parquet depuis MinIO clean-data/articles/
    2. Mapping des source_nom vers les sources en base
    3. Insertion avec upsert (ON CONFLICT sur URL)
    4. Mise à jour de la date de dernière collecte des sources
    5. Vérification post-ingestion

    Returns:
        Statistiques d'ingestion {inseres, ignores, erreurs}.
    """
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("INGESTION SQL : MinIO clean-data → PostgreSQL")
    logger.info("=" * 60)

    # 1. Lecture des données nettoyées depuis MinIO
    records = read_parquet_from_minio(settings.minio_bucket_clean, "articles")

    if not records:
        logger.warning("Aucun article nettoyé trouvé dans MinIO clean-data/articles/")
        logger.info("Vérifiez que le pipeline Spark (spark_cleaner.py) a été exécuté.")
        return {"inseres": 0, "ignores": 0, "erreurs": 0}

    # 2. Insertion dans PostgreSQL avec upsert
    stats = await ingest_to_postgresql(records)

    # 3. Vérification post-ingestion
    await verify_ingestion()

    logger.info("=" * 60)
    logger.info(
        f"RÉSULTAT FINAL : {stats['inseres']} insérés, "
        f"{stats['ignores']} ignorés (doublons), "
        f"{stats['erreurs']} erreurs"
    )
    logger.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_sql_ingestion())
