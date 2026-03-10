"""Gestion de la connexion asynchrone à PostgreSQL via SQLAlchemy 2.0.

Fournit un engine async et une factory de sessions pour toute l'application.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from greentech.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    echo=_settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Fournit une session de base de données asynchrone.

    À utiliser comme dépendance FastAPI ou dans un context manager async.

    Yields:
        Session SQLAlchemy asynchrone, automatiquement fermée en sortie.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_connection() -> bool:
    """Vérifie que la connexion à PostgreSQL fonctionne.

    Returns:
        True si la connexion est opérationnelle.
    """
    from sqlalchemy import text

    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Connexion PostgreSQL opérationnelle")
        return True
    except Exception as e:
        logger.error(f"Connexion PostgreSQL échouée : {e}")
        return False
