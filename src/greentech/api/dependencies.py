"""Dependances FastAPI injectees dans les routes.

Fournit les factories de session de base de donnees et
l'extraction de l'utilisateur courant depuis le token JWT.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.api.security.auth import decode_access_token
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import User

# Schema OAuth2 : l'URL de login est /auth/login
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Fournit une session de base de donnees asynchrone.

    Utilisee comme dependance FastAPI pour injecter la session DB
    dans chaque endpoint qui en a besoin.

    Yields:
        Session SQLAlchemy asynchrone.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extrait l'utilisateur courant depuis le token JWT.

    Decode le token, recupere l'utilisateur en base et verifie
    qu'il est actif. Renvoie une erreur 401 si le token est invalide
    ou si l'utilisateur n'existe pas/est desactive.

    Args:
        token: Token JWT extrait du header Authorization.
        db: Session de base de donnees.

    Returns:
        Instance User de l'utilisateur authentifie.

    Raises:
        HTTPException: 401 si le token est invalide ou l'utilisateur inexistant.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expire",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id_str: str | None = payload.get("sub")
    if user_id_str is None:
        raise credentials_exception

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    stmt = select(User).where(User.id_utilisateur == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        logger.warning(f"Token valide mais utilisateur introuvable : {user_id}")
        raise credentials_exception

    if not user.est_actif:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte desactive",
        )

    return user
