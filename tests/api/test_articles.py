"""Tests d'integration pour les endpoints d'articles.

Couvre les scenarios : liste paginee, filtres, detail, recherche,
et les cas d'erreur (article introuvable).

"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.data.storage.models import Article, Source
from tests.conftest import test_session_factory


@pytest.fixture
async def sample_source() -> Source:
    """Cree une source de test en base."""
    source = Source(nom="TestSource", type="api", url_base="https://test.com")
    async with test_session_factory() as session:
        session.add(source)
        await session.commit()
        await session.refresh(source)
    return source


@pytest.fixture
async def sample_articles(sample_source: Source) -> list[Article]:
    """Cree des articles de test en base."""
    articles = [
        Article(
            titre="Green AI: Sustainable Computing",
            url="https://test.com/article-1",
            contenu="Content about sustainable AI and green computing practices.",
            id_source=sample_source.id_source,
            est_green_it=True,
            score_confiance=0.95,
        ),
        Article(
            titre="Quantum Computing Advances",
            url="https://test.com/article-2",
            contenu="Content about quantum computing breakthroughs.",
            id_source=sample_source.id_source,
            est_green_it=False,
            score_confiance=0.88,
        ),
        Article(
            titre="Energy Efficient Data Centers",
            url="https://test.com/article-3",
            contenu="Content about energy efficiency in modern data centers.",
            id_source=sample_source.id_source,
        ),
    ]
    async with test_session_factory() as session:
        session.add_all(articles)
        await session.commit()
        for a in articles:
            await session.refresh(a)
    return articles


@pytest.mark.integration
class TestListArticles:
    """Tests pour GET /articles."""

    async def test_list_empty(self, client: AsyncClient) -> None:
        """La liste vide retourne un tableau vide avec total=0."""
        response = await client.get("/articles")
        assert response.status_code == 200
        data = response.json()
        assert data["articles"] == []
        assert data["total"] == 0

    async def test_list_with_articles(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """La liste retourne les articles avec pagination."""
        response = await client.get("/articles")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["articles"]) == 3
        assert data["page"] == 1
        assert data["pages"] == 1

    async def test_list_pagination(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """La pagination fonctionne correctement."""
        response = await client.get("/articles?page=1&limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["articles"]) == 2
        assert data["total"] == 3
        assert data["pages"] == 2

    async def test_filter_green_it(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """Le filtre Green IT retourne uniquement les articles Green IT."""
        response = await client.get("/articles?is_green_it=true")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["articles"][0]["est_green_it"] is True


@pytest.mark.integration
class TestGetArticle:
    """Tests pour GET /articles/{id}."""

    async def test_get_existing(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """Le detail d'un article existant retourne 200."""
        article_id = sample_articles[0].id_article
        response = await client.get(f"/articles/{article_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["titre"] == "Green AI: Sustainable Computing"
        assert data["contenu"] is not None

    async def test_get_not_found(self, client: AsyncClient) -> None:
        """Le detail d'un article inexistant retourne 404."""
        response = await client.get("/articles/99999")
        assert response.status_code == 404


@pytest.mark.integration
class TestSearchArticles:
    """Tests pour GET /articles/search."""

    async def test_search_by_title(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """La recherche par titre retourne les articles correspondants."""
        response = await client.get("/articles/search?q=Green")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    async def test_search_no_results(
        self, client: AsyncClient, sample_articles: list[Article]
    ) -> None:
        """La recherche sans correspondance retourne une liste vide."""
        response = await client.get("/articles/search?q=ZZZZZZZZZ")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0

    async def test_search_too_short(self, client: AsyncClient) -> None:
        """Une recherche trop courte retourne 422."""
        response = await client.get("/articles/search?q=A")
        assert response.status_code == 422
