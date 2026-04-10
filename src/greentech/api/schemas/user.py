"""Schemas Pydantic pour l'authentification et les utilisateurs.

Definit les modeles de requete et de reponse pour les endpoints /auth.
Inclut la validation stricte des entrees (email, mot de passe).

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """Schema pour la creation d'un compte utilisateur."""

    email: EmailStr = Field(..., description="Adresse email valide")
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Mot de passe (8 caracteres minimum)",
    )


class UserLogin(BaseModel):
    """Schema pour la connexion utilisateur."""

    email: EmailStr = Field(..., description="Adresse email")
    password: str = Field(..., description="Mot de passe")


class UserResponse(BaseModel):
    """Schema de reponse representant un utilisateur."""

    model_config = ConfigDict(from_attributes=True)

    id_utilisateur: uuid.UUID
    email: str
    est_actif: bool
    est_verifie: bool
    date_creation: datetime


class TokenResponse(BaseModel):
    """Schema de reponse pour un jeton JWT."""

    access_token: str = Field(..., description="Jeton JWT d'acces")
    token_type: str = Field(default="bearer", description="Type de jeton")


class MessageResponse(BaseModel):
    """Schema de reponse generique pour les messages."""

    message: str
