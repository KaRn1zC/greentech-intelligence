"""Tests d'integration pour les endpoints d'authentification.

Couvre les scenarios : inscription, connexion, acces au profil,
deconnexion, et les cas d'erreur (doublons, mauvais identifiants).

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from greentech.data.storage.models import User


@pytest.mark.integration
class TestRegister:
    """Tests pour POST /auth/register."""

    async def test_register_success(self, client: AsyncClient) -> None:
        """L'inscription avec des donnees valides retourne 201."""
        response = await client.post(
            "/auth/register",
            json={"email": "new@greentech.dev", "password": "SecurePass123"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "new@greentech.dev"
        assert data["est_actif"] is True
        assert "id_utilisateur" in data

    async def test_register_duplicate_email(
        self, client: AsyncClient, test_user: User
    ) -> None:
        """L'inscription avec un email existant retourne 409."""
        response = await client.post(
            "/auth/register",
            json={"email": test_user.email, "password": "AnotherPass123"},
        )
        assert response.status_code == 409

    async def test_register_weak_password(self, client: AsyncClient) -> None:
        """L'inscription avec un mot de passe trop court retourne 422."""
        response = await client.post(
            "/auth/register",
            json={"email": "weak@greentech.dev", "password": "short"},
        )
        assert response.status_code == 422

    async def test_register_invalid_email(self, client: AsyncClient) -> None:
        """L'inscription avec un email invalide retourne 422."""
        response = await client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "SecurePass123"},
        )
        assert response.status_code == 422


@pytest.mark.integration
class TestLogin:
    """Tests pour POST /auth/login."""

    async def test_login_success(
        self, client: AsyncClient, test_user: User
    ) -> None:
        """La connexion avec les bons identifiants retourne un token."""
        response = await client.post(
            "/auth/login",
            json={"email": test_user.email, "password": "TestPassword123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_wrong_password(
        self, client: AsyncClient, test_user: User
    ) -> None:
        """La connexion avec un mauvais mot de passe retourne 401."""
        response = await client.post(
            "/auth/login",
            json={"email": test_user.email, "password": "WrongPassword"},
        )
        assert response.status_code == 401

    async def test_login_unknown_email(self, client: AsyncClient) -> None:
        """La connexion avec un email inconnu retourne 401."""
        response = await client.post(
            "/auth/login",
            json={"email": "unknown@greentech.dev", "password": "Whatever123"},
        )
        assert response.status_code == 401


@pytest.mark.integration
class TestMe:
    """Tests pour GET /auth/me."""

    async def test_me_authenticated(
        self, client: AsyncClient, auth_headers: dict, test_user: User
    ) -> None:
        """Le profil est accessible avec un token valide."""
        response = await client.get("/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email

    async def test_me_no_token(self, client: AsyncClient) -> None:
        """Le profil est inaccessible sans token."""
        response = await client.get("/auth/me")
        assert response.status_code == 401

    async def test_me_invalid_token(self, client: AsyncClient) -> None:
        """Le profil est inaccessible avec un token invalide."""
        response = await client.get(
            "/auth/me", headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 401


@pytest.mark.integration
class TestLogout:
    """Tests pour POST /auth/logout."""

    async def test_logout_authenticated(
        self, client: AsyncClient, auth_headers: dict
    ) -> None:
        """La deconnexion avec un token valide retourne 200."""
        response = await client.post("/auth/logout", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["message"] == "Deconnexion reussie"

    async def test_logout_no_token(self, client: AsyncClient) -> None:
        """La deconnexion sans token retourne 401."""
        response = await client.post("/auth/logout")
        assert response.status_code == 401
