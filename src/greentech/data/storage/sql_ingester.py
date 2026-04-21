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
#   - Guardian : "theguardian.com" (fixe)
#   - Dev.to : "dev.to" (fixe)
#   - TechCrunch : "TechCrunch Climate" (fixe)
#   - arXiv Dataset : "arXiv Dataset" (Kaggle, type=file)
#   - arXiv API : "arxiv.org" (ajout B2.2)
#   - Crossref : "crossref.org" (ajout B2.2)
#   - 4 spiders statiques B2.3 : noms explicites configures dans chaque
#     spider (ex: "GreenIT.fr", "Green Software Foundation", ...)
#   - NewsData.io (legacy) : le source_id de l'API (ex: "bbc_news", "techradar")
SOURCE_NAME_MAPPING: dict[str, str] = {
    # Guardian Content API (source REST/JSON principale depuis avril 2026)
    "theguardian.com": "The Guardian",
    "theguardian": "The Guardian",
    "the guardian": "The Guardian",
    "guardian": "The Guardian",
    # Dev.to / Forem API (source REST/JSON complementaire)
    "dev.to": "Dev.to",
    "devto": "Dev.to",
    # TechCrunch (scraping hybride)
    "TechCrunch Climate": "TechCrunch Climate",
    "techcrunch": "TechCrunch Climate",
    "techcrunch.com": "TechCrunch Climate",
    # arXiv Dataset (file, dump Kaggle historique)
    "arXiv Dataset": "arXiv Dataset",
    "arxiv_dataset": "arXiv Dataset",
    # arXiv API (ajout B2.2, preprints live)
    "arXiv API": "arXiv API",
    "arxiv.org": "arXiv API",
    "arxiv_api": "arXiv API",
    # Crossref API (ajout B2.2, peer-reviewed)
    "Crossref": "Crossref",
    "crossref.org": "Crossref",
    "crossref": "Crossref",
    "api.crossref.org": "Crossref",
    # Spiders statiques B2.3 (4 sites Green IT)
    "GreenIT.fr": "GreenIT.fr",
    "greenit.fr": "GreenIT.fr",
    "Green Software Foundation": "Green Software Foundation",
    "greensoftware.foundation": "Green Software Foundation",
    "Sustainable Web Design": "Sustainable Web Design",
    "sustainablewebdesign.org": "Sustainable Web Design",
    "Climate Action Tech": "Climate Action Tech",
    "climateaction.tech": "Climate Action Tech",
    # NewsData.io (legacy, desactivee)
    "newsdata": "NewsData.io",
    "newsdata.io": "NewsData.io",
}


def _resolve_source_name(raw_name: str | None, url: str | None = None) -> str:
    """Resout le nom canonique de la source depuis source_nom ou URL.

    Applique en cascade plusieurs strategies de resolution :

    1. Correspondance exacte dans ``SOURCE_NAME_MAPPING``.
    2. Correspondance insensible a la casse.
    3. Heuristique par mot-cle dans ``raw_name`` (techcrunch, arxiv, ...).
    4. **Heuristique par URL** (filet de securite) : si l'URL de l'article
       pointe vers un domaine connu (theguardian.com, dev.to, techcrunch.com,
       arxiv.org), on retourne la source correspondante independamment
       de ``raw_name``.
    5. Fallback : NewsData.io (source historique de type 'api').

    La resolution par URL est la plus robuste car chaque collecteur stocke
    systematiquement l'URL d'origine de l'article, contrairement au champ
    ``source_nom`` qui peut etre un identifiant API specifique (ex: "bbc_news"
    pour NewsData) plutot que le nom de la plateforme source.

    Args:
        raw_name: Valeur brute du champ source_nom dans le Parquet.
        url: URL de l'article, utilisee comme filet de securite si
            ``raw_name`` est inconnu.

    Returns:
        Nom canonique de la source tel que defini dans la table ``sources``.
    """
    # 1. Correspondance exacte via source_nom
    if raw_name and raw_name in SOURCE_NAME_MAPPING:
        return SOURCE_NAME_MAPPING[raw_name]

    # 2. Correspondance insensible a la casse via source_nom
    if raw_name:
        raw_lower = raw_name.lower()
        for key, value in SOURCE_NAME_MAPPING.items():
            if key.lower() == raw_lower:
                return value

        # 3. Heuristique par mot-cle dans source_nom
        if "theguardian" in raw_lower or "guardian" in raw_lower:
            return "The Guardian"
        if "dev.to" in raw_lower or raw_lower == "devto":
            return "Dev.to"
        if "techcrunch" in raw_lower:
            return "TechCrunch Climate"
        # Important : ordre d'evaluation arxiv - l'API vit a "arxiv.org"
        # alors que le dataset Kaggle historique porte le nom "arXiv Dataset".
        # On renvoie "arXiv API" pour tout ce qui vient du live API et
        # "arXiv Dataset" uniquement pour l'ancien dump file.
        if "arxiv_dataset" in raw_lower or raw_name == "arXiv Dataset":
            return "arXiv Dataset"
        if "arxiv" in raw_lower:
            return "arXiv API"
        if "crossref" in raw_lower:
            return "Crossref"
        if "greenit.fr" in raw_lower or raw_lower == "greenit":
            return "GreenIT.fr"
        if "greensoftware" in raw_lower or "green software" in raw_lower:
            return "Green Software Foundation"
        if "sustainablewebdesign" in raw_lower or "sustainable web" in raw_lower:
            return "Sustainable Web Design"
        if "climateaction" in raw_lower or "climate action" in raw_lower:
            return "Climate Action Tech"

    # 4. Heuristique par URL (filet de securite contre les source_nom
    # incoherents, frequents avec l'ancien pipeline NewsData.io)
    if url:
        url_lower = url.lower()
        if "theguardian.com" in url_lower:
            return "The Guardian"
        if "dev.to/" in url_lower:
            return "Dev.to"
        if "techcrunch.com" in url_lower:
            return "TechCrunch Climate"
        if "arxiv.org" in url_lower:
            # Meme distinction que plus haut : on ne sait pas si c'est
            # un article live API ou un Kaggle dump, mais dans le doute
            # on renvoie l'API (la plus courante en B2+).
            return "arXiv API"
        if "doi.org" in url_lower or "api.crossref.org" in url_lower:
            return "Crossref"
        if "greenit.fr" in url_lower:
            return "GreenIT.fr"
        if "greensoftware.foundation" in url_lower:
            return "Green Software Foundation"
        if "sustainablewebdesign.org" in url_lower:
            return "Sustainable Web Design"
        if "climateaction.tech" in url_lower:
            return "Climate Action Tech"

    # 5. Fallback : NewsData.io (source legacy par defaut)
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
                    # Preparation des valeurs de l'article (URL d'abord pour
                    # permettre la resolution de source par heuristique URL).
                    url = record.get("url") or ""
                    if not url:
                        stats["erreurs"] += 1
                        continue

                    titre = (record.get("titre") or "")[:500]
                    if not titre:
                        stats["erreurs"] += 1
                        continue

                    contenu = record.get("contenu") or ""

                    # Filet de securite : on refuse les articles NewsData.io
                    # dont le contenu a ete tronque au placeholder du free
                    # tier. Ces entrees sont inexploitables pour l'entrainement
                    # et pourrissent le ground truth si on les laisse passer.
                    if "ONLY AVAILABLE IN PAID PLANS" in contenu:
                        stats["ignores"] += 1
                        continue

                    # Resolution de la source : on passe l'URL en second
                    # parametre pour que le fallback URL (heuristique sur les
                    # domaines connus) puisse corriger un source_nom
                    # incoherent (ex: source_id NewsData applique par erreur
                    # a un article Guardian legacy).
                    raw_source = record.get("source_nom")
                    resolved_name = _resolve_source_name(raw_source, url=url)
                    id_source = await _ensure_source_exists(session, source_cache, resolved_name)
                    sources_utilisees.add(resolved_name)

                    values: dict[str, Any] = {
                        "titre": titre,
                        "url": url,
                        "contenu": contenu,
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
