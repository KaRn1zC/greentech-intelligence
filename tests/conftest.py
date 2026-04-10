"""Fixtures partagees pour les tests du projet GreenTech Intelligence.

Fournit un client HTTP de test, une base de donnees en memoire
et des fixtures d'authentification reutilisables.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import JSON, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from greentech.api.dependencies import get_db
from greentech.api.main import app
from greentech.api.security.auth import create_access_token, hash_password
from greentech.data.storage.models import Base, DailyStats, User


# === Compatibilite SQLite : remplacer JSONB par JSON ===

# Le modele DailyStats utilise JSONB (PostgreSQL). Pour les tests SQLite,
# on remplace dynamiquement le type de la colonne par JSON standard.
DailyStats.__table__.c.articles_par_source.type = JSON()


# === Base de donnees de test (SQLite async en memoire) ===

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


# Activer le support des foreign keys dans SQLite
@event.listens_for(test_engine.sync_engine, "connect")
def set_sqlite_fk_pragma(dbapi_connection, _connection_record):
    """Active les contraintes de cles etrangeres dans SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Fournit une session de test utilisant la DB en memoire."""
    async with test_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Remplacer la dependance DB dans l'app
app.dependency_overrides[get_db] = override_get_db


# === Fixtures ===


@pytest.fixture(autouse=True)
async def setup_database():
    """Cree et nettoie les tables avant/apres chaque test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Client HTTP de test pour les requetes sur l'API."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def test_user() -> User:
    """Cree un utilisateur de test en base."""
    user = User(
        id_utilisateur=uuid.uuid4(),
        email="test@greentech.dev",
        mot_de_passe_hash=hash_password("TestPassword123"),
        est_actif=True,
    )
    async with test_session_factory() as session:
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user


@pytest.fixture
async def auth_headers(test_user: User) -> dict[str, str]:
    """Headers d'authentification avec un token JWT valide."""
    token = create_access_token(data={"sub": str(test_user.id_utilisateur)})
    return {"Authorization": f"Bearer {token}"}
