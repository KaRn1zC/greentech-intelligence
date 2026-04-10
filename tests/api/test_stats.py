"""Tests d'integration pour les endpoints de statistiques.

Couvre les scenarios : statistiques globales, quotidiennes,
par source, health check et metriques Prometheus.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from greentech.data.storage.models import Article, Source
from tests.conftest import test_session_factory


@pytest.fixture
async def stats_data() -> None:
    """Cree des donnees de test pour les statistiques."""
    source = Source(nom="StatsSource", type="api")
    async with test_session_factory() as session:
        session.add(source)
        await session.commit()
        await session.refresh(source)

        articles = [
            Article(
                titre=f"Article {i}",
                url=f"https://test.com/stats-{i}",
                id_source=source.id_source,
                est_green_it=(i % 3 == 0),
                score_confiance=0.8 + (i % 3) * 0.05,
            )
            for i in range(10)
        ]
        session.add_all(articles)
        await session.commit()


@pytest.mark.integration
class TestGlobalStats:
    """Tests pour GET /stats."""

    async def test_stats_empty(self, client: AsyncClient) -> None:
        """Les stats vides retournent des compteurs a zero."""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_articles"] == 0

    async def test_stats_with_data(
        self, client: AsyncClient, stats_data: None
    ) -> None:
        """Les stats refletent les donnees en base."""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_articles"] == 10
        assert data["articles_green_it"] >= 1
        assert data["articles_non_green_it"] >= 1


@pytest.mark.integration
class TestDailyStats:
    """Tests pour GET /stats/daily."""

    async def test_daily_stats(
        self, client: AsyncClient, stats_data: None
    ) -> None:
        """Les stats quotidiennes retournent les donnees du jour."""
        response = await client.get("/stats/daily?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "periode_debut" in data
        assert "periode_fin" in data


@pytest.mark.integration
class TestSourceStats:
    """Tests pour GET /stats/sources."""

    async def test_source_stats(
        self, client: AsyncClient, stats_data: None
    ) -> None:
        """Les stats par source retournent au moins une source."""
        response = await client.get("/stats/sources")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) >= 1
        assert data["sources"][0]["total_articles"] == 10


@pytest.mark.integration
class TestHealth:
    """Tests pour GET /health."""

    async def test_health_check(self, client: AsyncClient) -> None:
        """Le health check retourne un statut et une version."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "0.1.0"


@pytest.mark.integration
class TestMetrics:
    """Tests pour GET /metrics."""

    async def test_prometheus_metrics(self, client: AsyncClient) -> None:
        """Les metriques Prometheus retournent du texte au format attendu."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "greentech_" in response.text
