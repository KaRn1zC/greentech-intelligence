"""Script de vérification de l'infrastructure de stockage (Étape 2.2).

Vérifie la connectivité et le bon fonctionnement de PostgreSQL et MinIO
après leur déploiement via Docker Compose.

Rédigé par KaRn1zC - 2026-03-10
"""
from __future__ import annotations

import asyncio
import sys

from loguru import logger

# ---------------------------------------------------------------------------
# Configuration Loguru
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)

# ---------------------------------------------------------------------------
# Configuration connexion
# ---------------------------------------------------------------------------
POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "greentech",
    "password": "greentech_dev_password",
    "database": "greentech_db",
}

MINIO_CONFIG = {
    "endpoint": "localhost:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin123",
    "secure": False,
}

EXPECTED_TABLES = {
    "search_config",
    "sources",
    "articles",
    "users",
    "daily_stats",
    "analysis_logs",
}

EXPECTED_BUCKETS = {"raw-data", "clean-data", "models", "mlflow"}


# ---------------------------------------------------------------------------
# Vérification PostgreSQL
# ---------------------------------------------------------------------------
async def verify_postgres() -> bool:
    """Vérifie la connexion et la structure de la base PostgreSQL.

    Returns:
        True si toutes les vérifications passent, False sinon.
    """
    import asyncpg

    logger.info("--- Vérification PostgreSQL ---")

    try:
        conn = await asyncpg.connect(**POSTGRES_CONFIG)
        logger.info("Connexion PostgreSQL réussie")
    except Exception as e:
        logger.error(f"Connexion PostgreSQL échouée : {e}")
        return False

    try:
        # Vérifier les tables
        rows = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        tables = {row["tablename"] for row in rows}
        missing = EXPECTED_TABLES - tables
        if missing:
            logger.error(f"Tables manquantes : {missing}")
            return False
        logger.info(f"Tables trouvées : {len(tables)}/6 ({', '.join(sorted(tables))})")

        # Vérifier les vues
        views = await conn.fetch(
            "SELECT viewname FROM pg_views WHERE schemaname = 'public'"
        )
        view_names = [v["viewname"] for v in views]
        logger.info(f"Vues trouvées : {', '.join(view_names)}")

        # Vérifier les données de test
        source_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
        config_count = await conn.fetchval("SELECT COUNT(*) FROM search_config")
        logger.info(f"Données initiales : {source_count} sources, {config_count} configs")

        # Vérifier les triggers
        triggers = await conn.fetch(
            "SELECT trigger_name, event_object_table "
            "FROM information_schema.triggers "
            "WHERE trigger_schema = 'public'"
        )
        logger.info(f"Triggers actifs : {len(triggers)}")
        for t in triggers:
            logger.info(f"  - {t['trigger_name']} sur {t['event_object_table']}")

        # Vérifier l'extension uuid-ossp
        ext = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'uuid-ossp'"
        )
        if ext:
            logger.info("Extension uuid-ossp activée")
        else:
            logger.warning("Extension uuid-ossp manquante")

    finally:
        await conn.close()

    logger.info("PostgreSQL : OK")
    return True


# ---------------------------------------------------------------------------
# Vérification MinIO
# ---------------------------------------------------------------------------
def verify_minio() -> bool:
    """Vérifie la connexion et les buckets MinIO.

    Returns:
        True si toutes les vérifications passent, False sinon.
    """
    from minio import Minio

    logger.info("--- Vérification MinIO ---")

    try:
        client = Minio(
            MINIO_CONFIG["endpoint"],
            access_key=MINIO_CONFIG["access_key"],
            secret_key=MINIO_CONFIG["secret_key"],
            secure=MINIO_CONFIG["secure"],
        )

        # Lister les buckets
        buckets = {b.name for b in client.list_buckets()}
        logger.info(f"Buckets trouvés : {', '.join(sorted(buckets))}")

        missing = EXPECTED_BUCKETS - buckets
        if missing:
            logger.error(f"Buckets manquants : {missing}")
            return False

        # Vérifier l'accès en écriture sur raw-data (test avec un petit fichier)
        import io

        test_data = b"test-verification-infrastructure"
        client.put_object(
            "raw-data",
            "_test_verify.txt",
            io.BytesIO(test_data),
            len(test_data),
        )
        logger.info("Écriture test dans raw-data : OK")

        # Relire et supprimer
        response = client.get_object("raw-data", "_test_verify.txt")
        content = response.read()
        response.close()
        response.release_conn()
        assert content == test_data, "Contenu lu différent du contenu écrit"
        logger.info("Lecture test depuis raw-data : OK")

        client.remove_object("raw-data", "_test_verify.txt")
        logger.info("Suppression test : OK")

    except Exception as e:
        logger.error(f"Vérification MinIO échouée : {e}")
        return False

    logger.info("MinIO : OK")
    return True


# ---------------------------------------------------------------------------
# Vérification SQLAlchemy async
# ---------------------------------------------------------------------------
async def verify_sqlalchemy() -> bool:
    """Vérifie la connexion via SQLAlchemy 2.0 async (comme le fera l'application).

    Returns:
        True si la connexion fonctionne, False sinon.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    logger.info("--- Vérification SQLAlchemy async ---")

    cfg = POSTGRES_CONFIG
    url = (
        f"postgresql+asyncpg://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )

    try:
        engine = create_async_engine(url, echo=False)
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM articles"))
            count = result.scalar()
            logger.info(f"SQLAlchemy async : connexion OK (articles={count})")

        await engine.dispose()

    except Exception as e:
        logger.error(f"SQLAlchemy async échoué : {e}")
        return False

    logger.info("SQLAlchemy async : OK")
    return True


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
async def main() -> None:
    """Lance toutes les vérifications d'infrastructure."""
    logger.info("=" * 60)
    logger.info("Vérification de l'infrastructure GreenTech Intelligence")
    logger.info("=" * 60)

    results = {
        "PostgreSQL": await verify_postgres(),
        "MinIO": verify_minio(),
        "SQLAlchemy async": await verify_sqlalchemy(),
    }

    logger.info("=" * 60)
    logger.info("RÉSULTAT FINAL")
    logger.info("=" * 60)

    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("Infrastructure opérationnelle !")
    else:
        logger.error("Des vérifications ont échoué.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
