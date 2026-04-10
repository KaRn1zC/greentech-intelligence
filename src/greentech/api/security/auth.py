"""Gestion de l'authentification JWT et du hashing des mots de passe.

Fournit les fonctions de creation/verification des tokens JWT
et de hashing/verification des mots de passe via bcrypt.
Respecte les recommandations OWASP pour la securite des tokens.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt
from loguru import logger

from greentech.config import get_settings

_settings = get_settings()


def hash_password(password: str) -> str:
    """Hashe un mot de passe en clair avec bcrypt.

    Args:
        password: Mot de passe en clair.

    Returns:
        Hash bcrypt du mot de passe.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifie un mot de passe contre son hash bcrypt.

    Args:
        plain_password: Mot de passe en clair a verifier.
        hashed_password: Hash bcrypt stocke en base.

    Returns:
        True si le mot de passe correspond au hash.
    """
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def create_access_token(
    data: dict[str, Any],
    *,
    expires_delta: timedelta | None = None,
) -> str:
    """Cree un jeton JWT signe.

    Args:
        data: Donnees a encoder dans le token (ex: {"sub": "user@email.com"}).
        expires_delta: Duree de validite du token.

    Returns:
        Token JWT encode sous forme de string.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=_settings.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})

    return jwt.encode(
        to_encode,
        _settings.jwt_secret_key,
        algorithm=_settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode et verifie un jeton JWT.

    Args:
        token: Token JWT a decoder.

    Returns:
        Payload du token si valide, None sinon.
    """
    try:
        payload = jwt.decode(
            token,
            _settings.jwt_secret_key,
            algorithms=[_settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        logger.debug(f"Token JWT invalide : {e}")
        return None
