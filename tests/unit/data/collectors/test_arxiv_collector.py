"""Tests unitaires du collecteur arXiv API.

Ces tests couvrent la logique de parsing pure (``_parse_entry``) sans
appeler l'API arXiv. Les cas d'integration avec de vraies requetes HTTP
sont verifies en smoke test manuel (voir ``arxiv_collector.py``
``__main__``).
"""

from __future__ import annotations

import pytest

from greentech.data.collectors.arxiv_collector import ArxivCollector


@pytest.fixture
def collector() -> ArxivCollector:
    """Instance du collecteur sans appel reseau."""
    return ArxivCollector()


@pytest.fixture
def valid_entry() -> dict:
    """Entree feedparser typique d'un papier cs.DC Green IT.

    Les champs suivent la structure reelle observee sur l'API arXiv :
    ``id`` avec version, ``summary`` en abstract complet, ``tags`` avec
    ``term`` pour chaque categorie, ``authors`` comme liste de dicts.
    """
    return {
        "id": "http://arxiv.org/abs/2401.12345v2",
        "title": "Carbon-Aware Computing for Datacenters",
        "summary": (
            "The amount of CO2 emitted per kilowatt-hour varies by time of day. "
            "This paper introduces a carbon-intelligent compute management system "
            "that minimizes electricity-based carbon footprint by delaying "
            "temporally flexible workloads. We demonstrate that our approach "
            "reduces emissions by 20% compared to baseline scheduling."
        ),
        "published": "2024-01-22T10:15:30Z",
        "tags": [
            {"term": "cs.DC"},
            {"term": "eess.SY"},
        ],
        "arxiv_primary_category": {"term": "cs.DC"},
        "authors": [
            {"name": "Alice Doe"},
            {"name": "Bob Martin"},
        ],
    }


class TestParseEntry:
    """Tests du parsing d'une entree Atom XML arXiv."""

    def test_parse_valid_cs_entry(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Une entree cs.DC valide doit etre correctement normalisee."""
        parsed = collector._parse_entry(valid_entry, keyword="carbon-aware")

        assert parsed is not None
        assert parsed["titre"] == "Carbon-Aware Computing for Datacenters"
        assert parsed["url"] == "http://arxiv.org/abs/2401.12345"  # version strippee
        assert parsed["arxiv_id"] == "2401.12345"
        assert parsed["contenu"].startswith("The amount of CO2")
        assert parsed["auteur"] == "Alice Doe, Bob Martin"
        assert parsed["langue"] == "en"
        assert parsed["source_nom"] == "arxiv.org"
        assert "cs.DC" in parsed["categorie"]
        assert parsed["categorie_principale"] == "cs.DC"
        assert parsed["keyword_search"] == "carbon-aware"

    def test_version_stripped_from_url(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """L'URL canonique doit retirer le suffixe de version (v1, v2, ...)."""
        parsed = collector._parse_entry(valid_entry, keyword="x")
        assert parsed is not None
        assert "v" not in parsed["url"].rsplit("/", 1)[-1]

    def test_rejects_irrelevant_category(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Une entree hors scope (ex: math.NA) doit etre rejetee."""
        entry = {**valid_entry}
        entry["tags"] = [{"term": "math.NA"}]
        entry["arxiv_primary_category"] = {"term": "math.NA"}
        assert collector._parse_entry(entry, keyword="x") is None

    def test_rejects_short_abstract(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Un abstract de moins de 100 chars doit etre rejete."""
        entry = {**valid_entry}
        entry["summary"] = "Short."
        assert collector._parse_entry(entry, keyword="x") is None

    def test_rejects_missing_id(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Une entree sans id doit etre rejetee."""
        entry = {**valid_entry, "id": ""}
        assert collector._parse_entry(entry, keyword="x") is None

    def test_rejects_missing_title(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Une entree sans titre doit etre rejetee."""
        entry = {**valid_entry, "title": ""}
        assert collector._parse_entry(entry, keyword="x") is None

    def test_title_truncated_to_500_chars(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Le titre doit etre tronque a 500 chars (contrainte BDD)."""
        entry = {**valid_entry, "title": "A" * 600}
        parsed = collector._parse_entry(entry, keyword="x")
        assert parsed is not None
        assert len(parsed["titre"]) == 500

    def test_authors_capped_at_five(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Les auteurs doivent etre plafonnes a 5 noms."""
        entry = {**valid_entry}
        entry["authors"] = [{"name": f"Author {i}"} for i in range(10)]
        parsed = collector._parse_entry(entry, keyword="x")
        assert parsed is not None
        assert parsed["auteur"].count(",") == 4  # 5 noms = 4 virgules

    def test_accepts_eess_category(
        self,
        collector: ArxivCollector,
        valid_entry: dict,
    ) -> None:
        """Une entree eess.SP (Signal Processing) doit etre acceptee."""
        entry = {**valid_entry}
        entry["tags"] = [{"term": "eess.SP"}]
        entry["arxiv_primary_category"] = {"term": "eess.SP"}
        parsed = collector._parse_entry(entry, keyword="x")
        assert parsed is not None
        assert parsed["categorie_principale"] == "eess.SP"
