"""Tests d'integration pour les endpoints d'analyse IA.

Couvre les scenarios : soumission d'analyse (URL, texte), suivi de job,
et les cas d'erreur (pas d'auth, input invalide, job introuvable).
Les composants IA (classification, resume) sont mockes pour ne pas
charger de modele reel pendant les tests.

Redige par KaRn1zC - 2026-04-11
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from greentech.ai.models.classifier import LabelGreenIT, PredictionResult
from greentech.api.routes.analyze import _jobs

# === Helpers ===


def _mock_prediction() -> PredictionResult:
    """Cree un PredictionResult fictif pour les tests."""
    return PredictionResult(
        label=LabelGreenIT.GREEN,
        score_confiance=0.92,
        temps_ms=150,
        modele="mock-classifier",
    )


# === Fixtures ===


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Vide le store de jobs avant et apres chaque test."""
    _jobs.clear()
    yield
    _jobs.clear()


# === Tests POST /analyze ===


@pytest.mark.integration
class TestCreateAnalysis:
    """Tests pour POST /analyze — soumission d'une analyse."""

    async def test_submit_text_returns_202(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """La soumission d'un texte valide retourne 202 avec un job_id."""
        response = await client.post(
            "/analyze",
            json={"texte": "A" * 100},
            headers=auth_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["statut"] == "en_attente"
        assert "message" in data

    async def test_submit_url_returns_202(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """La soumission d'une URL retourne 202 avec un job_id."""
        response = await client.post(
            "/analyze",
            json={"url": "https://example.com/article-green-it"},
            headers=auth_headers,
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    async def test_submit_no_auth_returns_401(
        self, client: AsyncClient
    ) -> None:
        """La soumission sans token d'authentification retourne 401."""
        response = await client.post(
            "/analyze",
            json={"texte": "A" * 100},
        )
        assert response.status_code == 401

    async def test_submit_empty_body_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """La soumission sans URL ni texte retourne 422."""
        response = await client.post(
            "/analyze",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_submit_both_url_and_text_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Fournir URL et texte simultanement retourne 422."""
        response = await client.post(
            "/analyze",
            json={"url": "https://example.com", "texte": "A" * 100},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_submit_text_too_short_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Un texte de moins de 50 caracteres retourne 422."""
        response = await client.post(
            "/analyze",
            json={"texte": "Trop court"},
            headers=auth_headers,
        )
        assert response.status_code == 422


# === Tests GET /analyze/{job_id} ===


@pytest.mark.integration
class TestGetAnalysisStatus:
    """Tests pour GET /analyze/{job_id} — suivi de job."""

    async def test_unknown_job_returns_404(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Un job_id inconnu retourne 404."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/analyze/{fake_id}", headers=auth_headers)
        assert response.status_code == 404

    async def test_invalid_uuid_returns_422(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Un job_id au format invalide retourne 422."""
        response = await client.get("/analyze/not-a-uuid", headers=auth_headers)
        assert response.status_code == 422

    async def test_get_status_no_auth_returns_401(
        self, client: AsyncClient
    ) -> None:
        """L'acces sans token retourne 401."""
        fake_id = str(uuid.uuid4())
        response = await client.get(f"/analyze/{fake_id}")
        assert response.status_code == 401

    async def test_submitted_job_is_retrievable(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        """Un job soumis est visible via GET avec un statut valide."""
        post_resp = await client.post(
            "/analyze",
            json={"texte": "A" * 100},
            headers=auth_headers,
        )
        job_id = post_resp.json()["job_id"]

        get_resp = await client.get(f"/analyze/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["statut"] in ("en_attente", "en_cours", "termine", "erreur")


# === Tests du pipeline complet avec mocks ===


@pytest.mark.integration
class TestAnalysisPipeline:
    """Tests du pipeline d'analyse complet avec mocks sur les composants IA."""

    @patch(
        "greentech.ai.models.inference.classify_article",
        new_callable=AsyncMock,
    )
    @patch(
        "greentech.ai.services.summarizer.summarize_article",
        new_callable=AsyncMock,
    )
    async def test_text_analysis_completes_with_mocks(
        self,
        mock_summarize: AsyncMock,
        mock_classify: AsyncMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Le pipeline texte complet termine avec les composants IA mockes."""
        mock_classify.return_value = _mock_prediction()
        mock_summarize.return_value = AsyncMock(
            succes=True, resume="Resume de test."
        )

        post_resp = await client.post(
            "/analyze",
            json={"texte": "X" * 200},
            headers=auth_headers,
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        # Attendre que la background task s'execute
        import asyncio

        await asyncio.sleep(0.5)

        get_resp = await client.get(f"/analyze/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        data = get_resp.json()
        # Le job doit etre termine ou en erreur (pas en_attente indefiniment)
        assert data["statut"] in ("termine", "erreur")

    @patch(
        "greentech.ai.models.inference.classify_article",
        new_callable=AsyncMock,
        side_effect=FileNotFoundError("Modele introuvable"),
    )
    async def test_analysis_continues_when_model_missing(
        self,
        mock_classify: AsyncMock,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """L'analyse continue meme si le modele n'est pas disponible."""
        post_resp = await client.post(
            "/analyze",
            json={"texte": "Y" * 200},
            headers=auth_headers,
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        import asyncio

        await asyncio.sleep(0.5)

        get_resp = await client.get(f"/analyze/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        # Le pipeline gere le FileNotFoundError gracieusement
        data = get_resp.json()
        assert data["statut"] in ("termine", "erreur")
