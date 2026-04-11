"""Routes d'authentification (inscription, connexion, profil).

Endpoints pour la gestion des comptes utilisateurs et l'emission
de tokens JWT. Respecte les recommandations OWASP :
- Hashing bcrypt pour les mots de passe
- Reponses generiques pour eviter l'enumeration des comptes
- Validation stricte des entrees via Pydantic

"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.api.dependencies import get_current_user, get_db
from greentech.api.schemas.user import (
    MessageResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from greentech.api.security.auth import create_access_token, hash_password, verify_password
from greentech.data.storage.models import User

router = APIRouter(prefix="/auth", tags=["Authentification"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Creer un compte utilisateur",
)
async def register(
    data: UserRegister,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Cree un nouveau compte utilisateur.

    Verifie que l'email n'est pas deja utilise, hashe le mot de passe
    et insere l'utilisateur en base.

    Args:
        data: Donnees d'inscription validees par Pydantic.
        db: Session de base de donnees.

    Returns:
        Utilisateur cree.

    Raises:
        HTTPException: 409 si l'email est deja pris.
    """
    # Verifier l'unicite de l'email
    stmt = select(User).where(User.email == data.email)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un compte avec cet email existe deja",
        )

    user = User(
        email=data.email,
        mot_de_passe_hash=hash_password(data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info(f"Nouveau compte cree : {data.email}")
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Connexion utilisateur (JWT)",
)
async def login(
    data: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authentifie un utilisateur et retourne un token JWT.

    Verifie les identifiants et genere un token d'acces.
    Le message d'erreur est volontairement generique pour
    empecher l'enumeration des comptes (OWASP A07).

    Args:
        data: Identifiants de connexion.
        db: Session de base de donnees.

    Returns:
        Token JWT d'acces.

    Raises:
        HTTPException: 401 si les identifiants sont incorrects.
    """
    # Message generique pour eviter l'enumeration
    bad_credentials = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Email ou mot de passe incorrect",
        headers={"WWW-Authenticate": "Bearer"},
    )

    stmt = select(User).where(User.email == data.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise bad_credentials

    if not verify_password(data.password, user.mot_de_passe_hash):
        raise bad_credentials

    if not user.est_actif:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Compte desactive",
        )

    token = create_access_token(data={"sub": str(user.id_utilisateur)})
    logger.info(f"Connexion reussie : {data.email}")

    return TokenResponse(access_token=token)


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Deconnexion",
)
async def logout(
    _current_user: User = Depends(get_current_user),
) -> MessageResponse:
    """Deconnexion de l'utilisateur.

    Avec des tokens JWT stateless, la deconnexion est geree cote client
    en supprimant le token. Cet endpoint valide que le token est encore actif.

    Returns:
        Message de confirmation.
    """
    return MessageResponse(message="Deconnexion reussie")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Profil utilisateur courant",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> User:
    """Retourne le profil de l'utilisateur authentifie.

    Args:
        current_user: Utilisateur extrait du token JWT.

    Returns:
        Profil de l'utilisateur courant.
    """
    return current_user
