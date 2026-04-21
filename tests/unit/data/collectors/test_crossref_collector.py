"""Tests unitaires du collecteur Crossref.

Couvre le parsing des items JSON (``_parse_item``), le strip des balises
JATS autour des abstracts, et l'extraction de la date de publication
depuis les multiples champs possibles (``published-print``, etc.).
"""

from __future__ import annotations

import pytest

from greentech.data.collectors.crossref_collector import (
    CrossrefCollector,
    _extract_date,
    _strip_jats_tags,
)


@pytest.fixture
def collector() -> CrossrefCollector:
    """Instance du collecteur sans appel reseau."""
    return CrossrefCollector()


@pytest.fixture
def valid_item() -> dict:
    """Item Crossref typique d'un journal-article Green IT."""
    return {
        "DOI": "10.1000/test.12345",
        "type": "journal-article",
        "title": ["Green Cloud Computing: A Survey"],
        "abstract": (
            "<jats:p>This paper discusses the carbon footprint caused by computer "
            "resources, cloud computing and the environmental damage it causes. "
            "The growing need of data centers and the resulting environmental "
            "problems lie at the core of this survey, which reviews twenty "
            "energy-efficient cloud management strategies.</jats:p>"
        ),
        "published-print": {"date-parts": [[2024, 3, 15]]},
        "language": "EN",
        "author": [
            {"given": "Alice", "family": "Doe"},
            {"given": "Bob", "family": "Martin"},
        ],
        "subject": ["Computer Science", "Sustainability"],
    }


class TestStripJatsTags:
    """Tests du nettoyage JATS."""

    def test_strip_simple_jats_p(self) -> None:
        raw = "<jats:p>Hello world.</jats:p>"
        assert _strip_jats_tags(raw) == "Hello world."

    def test_strip_multiple_tags(self) -> None:
        raw = "<jats:sec><jats:title>Abstract</jats:title><jats:p>Content here.</jats:p></jats:sec>"
        result = _strip_jats_tags(raw)
        assert result == "Abstract Content here."

    def test_normalize_whitespace(self) -> None:
        raw = "<p>Line 1\n\n\n<br>Line 2</p>    extra   spaces"
        assert _strip_jats_tags(raw) == "Line 1 Line 2 extra spaces"

    def test_empty_input_returns_empty(self) -> None:
        assert _strip_jats_tags("") == ""
        assert _strip_jats_tags(None) == ""  # type: ignore[arg-type]


class TestExtractDate:
    """Tests de l'extraction de date Crossref (formats variables)."""

    def test_full_date_yyyy_mm_dd(self) -> None:
        item = {"published-print": {"date-parts": [[2024, 3, 15]]}}
        assert _extract_date(item) == "2024-03-15"

    def test_partial_date_year_month(self) -> None:
        """Un mois sans jour doit completer avec le 1er."""
        item = {"published-print": {"date-parts": [[2024, 5]]}}
        assert _extract_date(item) == "2024-05-01"

    def test_year_only(self) -> None:
        """Une annee seule doit completer avec 01-01."""
        item = {"published-print": {"date-parts": [[2024]]}}
        assert _extract_date(item) == "2024-01-01"

    def test_fallback_to_published_online(self) -> None:
        """Si published-print absent, fallback sur published-online."""
        item = {
            "published-online": {"date-parts": [[2023, 12, 1]]},
        }
        assert _extract_date(item) == "2023-12-01"

    def test_returns_none_if_no_date(self) -> None:
        assert _extract_date({}) is None


class TestParseItem:
    """Tests du parsing d'un item Crossref complet."""

    def test_parse_valid_journal_article(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        parsed = collector._parse_item(valid_item, keyword="green computing")

        assert parsed is not None
        assert parsed["titre"] == "Green Cloud Computing: A Survey"
        assert parsed["url"] == "https://doi.org/10.1000/test.12345"
        assert parsed["doi"] == "10.1000/test.12345"
        assert "jats:" not in parsed["contenu"]  # balises strippees
        assert parsed["contenu"].startswith("This paper discusses")
        assert parsed["auteur"] == "Alice Doe, Bob Martin"
        assert parsed["langue"] == "en"
        assert parsed["date_publication"] == "2024-03-15"
        assert parsed["source_nom"] == "crossref.org"
        assert parsed["type_crossref"] == "journal-article"

    def test_rejects_non_journal_article(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        """Les livres, chapitres, reports doivent etre rejetes."""
        item = {**valid_item, "type": "book-chapter"}
        assert collector._parse_item(item, keyword="x") is None

    def test_rejects_missing_doi(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        item = {**valid_item}
        del item["DOI"]
        assert collector._parse_item(item, keyword="x") is None

    def test_rejects_missing_title(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        item = {**valid_item, "title": []}
        assert collector._parse_item(item, keyword="x") is None

    def test_rejects_short_abstract(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        """Un abstract < 100 chars doit etre rejete."""
        item = {**valid_item, "abstract": "<jats:p>Too short.</jats:p>"}
        assert collector._parse_item(item, keyword="x") is None

    def test_accepts_proceedings_article(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        item = {**valid_item, "type": "proceedings-article"}
        parsed = collector._parse_item(item, keyword="x")
        assert parsed is not None
        assert parsed["type_crossref"] == "proceedings-article"

    def test_default_language_en(
        self,
        collector: CrossrefCollector,
        valid_item: dict,
    ) -> None:
        """Sans champ language, la langue par defaut est 'en'."""
        item = {**valid_item}
        del item["language"]
        parsed = collector._parse_item(item, keyword="x")
        assert parsed is not None
        assert parsed["langue"] == "en"
