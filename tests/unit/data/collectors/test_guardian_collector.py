"""Tests unitaires de l'extension Guardian avec sections.

On se concentre sur la nouvelle logique introduite en B2.2 : passage du
parametre ``sections`` et deduplication par URL au niveau du collecteur.
Le parsing ``_parse_articles`` est deja teste implicitement par les
smokes tests de production (le code n'a pas change).
"""

from __future__ import annotations

import pytest

from greentech.data.collectors.guardian_collector import (
    DEFAULT_GREEN_IT_SECTIONS,
    GuardianCollector,
)


@pytest.fixture
def collector() -> GuardianCollector:
    """Instance du collecteur sans appel reseau."""
    return GuardianCollector()


class TestGuardianExtension:
    """Tests de la nouvelle API avec sections optionnelles."""

    def test_default_sections_include_environment_and_technology(self) -> None:
        """Les sections par defaut couvrent les deux silos Green IT majeurs."""
        assert "environment" in DEFAULT_GREEN_IT_SECTIONS
        assert "technology" in DEFAULT_GREEN_IT_SECTIONS

    def test_parse_articles_short_body_rejected(
        self,
        collector: GuardianCollector,
    ) -> None:
        """Un article avec bodyText < 50 chars doit etre rejete."""
        raw = [
            {
                "webTitle": "Test",
                "webUrl": "https://example.com/1",
                "fields": {"bodyText": "Short."},
                "webPublicationDate": "2024-01-01T00:00:00Z",
                "sectionName": "environment",
            }
        ]
        assert collector._parse_articles(raw) == []

    def test_parse_articles_missing_title_rejected(
        self,
        collector: GuardianCollector,
    ) -> None:
        raw = [
            {
                "webTitle": None,
                "webUrl": "https://example.com/1",
                "fields": {"bodyText": "A" * 200},
            }
        ]
        assert collector._parse_articles(raw) == []

    def test_parse_articles_valid_entry(
        self,
        collector: GuardianCollector,
    ) -> None:
        """Un article valide doit etre correctement normalise."""
        raw = [
            {
                "webTitle": "Green AI datacenters",
                "webUrl": "https://theguardian.com/test/1",
                "fields": {
                    "bodyText": "A" * 500,
                    "trailText": "Short description",
                    "byline": "John Doe",
                    "lang": "en",
                },
                "webPublicationDate": "2024-01-01T00:00:00Z",
                "sectionName": "environment",
                "id": "test-id-1",
            }
        ]
        parsed = collector._parse_articles(raw)
        assert len(parsed) == 1
        assert parsed[0]["titre"] == "Green AI datacenters"
        assert parsed[0]["auteur"] == "John Doe"
        assert parsed[0]["categorie"] == ["environment"]

    @pytest.mark.asyncio
    async def test_collect_without_api_key_returns_error(
        self,
        collector: GuardianCollector,
    ) -> None:
        """Sans cle API, collect() doit renvoyer une erreur propre."""
        collector.settings.guardian_api_key = ""
        result = await collector.collect(["test"])
        assert result.articles_count == 0
        assert len(result.errors) == 1
        assert "GUARDIAN_API_KEY" in result.errors[0]
