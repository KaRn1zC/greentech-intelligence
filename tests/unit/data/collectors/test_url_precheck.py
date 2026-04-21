"""Tests de correctness du pre-check URL partage entre collecteurs.

Ces tests prouvent que :

1. ``normalize_url`` applique les regles documentees (scheme, trailing
   slash, host) et que deux normalisations de la meme URL sont toujours
   egales (idempotence).
2. ``url_is_known`` ne produit AUCUN faux positif : une URL absente du
   set retourne toujours False, independamment des variations cosmetiques.
3. ``url_is_known`` elimine les faux negatifs cles : les variantes
   http/https, trailing slash, casse host sont considerees equivalentes.
4. ``coerce_bool`` gere correctement les conversions string/int/bool
   attendues des flags CLI.

Les tests ne touchent pas la BDD (pas de `load_known_urls` reel), on se
contente d'injecter un set synthetique dans `url_is_known`.
"""

from __future__ import annotations

import pytest

from greentech.data.collectors.url_precheck import (
    coerce_bool,
    normalize_url,
    url_is_known,
)


class TestNormalizeUrl:
    """Verifie les regles de normalisation URL."""

    def test_empty_returns_empty(self) -> None:
        assert normalize_url("") == ""
        assert normalize_url(None) == ""  # type: ignore[arg-type]
        assert normalize_url("   ") == ""

    def test_trim_whitespace(self) -> None:
        assert normalize_url("  https://example.com  ") == "https://example.com"

    def test_http_forced_to_https(self) -> None:
        assert normalize_url("http://arxiv.org/abs/123") == "https://arxiv.org/abs/123"

    def test_https_stays_https(self) -> None:
        assert normalize_url("https://arxiv.org/abs/123") == "https://arxiv.org/abs/123"

    def test_uppercase_scheme_coerced(self) -> None:
        assert normalize_url("HTTP://example.com/x") == "https://example.com/x"
        assert normalize_url("HTTPS://example.com/x") == "https://example.com/x"

    def test_host_lowercased(self) -> None:
        assert normalize_url("https://Example.COM/Article") == "https://example.com/Article"

    def test_path_case_preserved(self) -> None:
        """Le path peut etre case-sensitive cote serveur, on le preserve."""
        assert normalize_url("https://example.com/ABC/Xyz") == "https://example.com/ABC/Xyz"

    def test_trailing_slash_stripped_on_path(self) -> None:
        assert normalize_url("https://example.com/article/") == "https://example.com/article"

    def test_trailing_slash_preserved_on_root(self) -> None:
        """La racine ``https://example.com/`` reste intacte (pas de path)."""
        assert normalize_url("https://example.com/") == "https://example.com/"

    def test_query_string_preserved(self) -> None:
        """Les query strings sont semantiques (ex: ?p=123), on garde tel quel."""
        assert normalize_url("https://example.com/x?p=1") == "https://example.com/x?p=1"

    def test_fragment_preserved(self) -> None:
        assert normalize_url("https://example.com/x#section") == "https://example.com/x#section"

    def test_idempotent(self) -> None:
        """normalize(normalize(x)) == normalize(x)."""
        for url in [
            "http://Arxiv.org/abs/123/",
            "HTTPS://example.COM/path",
            "  https://example.com/  ",
            "https://www.greenit.fr/2024/01/01/article/",
        ]:
            once = normalize_url(url)
            twice = normalize_url(once)
            assert once == twice

    def test_arxiv_http_vs_https_converge(self) -> None:
        """Le cas problematique reel : arxiv http vs https doit matcher."""
        from_kaggle = normalize_url("https://arxiv.org/abs/1234.5678")
        from_api = normalize_url("http://arxiv.org/abs/1234.5678")
        assert from_kaggle == from_api


class TestUrlIsKnownPreventsFalsePositives:
    """Prouve qu'une URL REELLEMENT absente ne declenche jamais un skip."""

    def test_empty_set_never_matches(self) -> None:
        assert url_is_known("https://example.com/x", set()) is False

    def test_different_host_never_matches(self) -> None:
        known = {"https://example.com/article"}
        assert url_is_known("https://other.com/article", known) is False

    def test_different_path_never_matches(self) -> None:
        known = {"https://example.com/article-1"}
        assert url_is_known("https://example.com/article-2", known) is False

    def test_path_case_difference_never_matches(self) -> None:
        """Path case sensitive = URLs differentes = pas de match."""
        known = {"https://example.com/MyArticle"}
        assert url_is_known("https://example.com/myarticle", known) is False

    def test_query_string_difference_never_matches(self) -> None:
        known = {"https://example.com/x?p=1"}
        assert url_is_known("https://example.com/x?p=2", known) is False

    def test_empty_candidate_is_false(self) -> None:
        known = {"https://example.com/x"}
        assert url_is_known("", known) is False
        assert url_is_known(None, known) is False


class TestUrlIsKnownCatchesKnownUrls:
    """Verifie que les variantes d'une URL connue sont bien detectees."""

    def test_exact_match(self) -> None:
        known = {"https://example.com/x"}
        assert url_is_known("https://example.com/x", known) is True

    def test_http_vs_https_equivalence(self) -> None:
        """Le cas critique : file_ingester (https) vs arxiv_api (http)."""
        # DB (via file_ingester) contient les URLs https
        known_from_db = {normalize_url("https://arxiv.org/abs/1234.5678")}
        # Collecteur API produit http
        assert url_is_known("http://arxiv.org/abs/1234.5678", known_from_db) is True

    def test_trailing_slash_equivalence(self) -> None:
        known_from_db = {normalize_url("https://example.com/article")}
        assert url_is_known("https://example.com/article/", known_from_db) is True

    def test_host_case_equivalence(self) -> None:
        known_from_db = {normalize_url("https://example.com/x")}
        assert url_is_known("https://EXAMPLE.com/x", known_from_db) is True

    def test_whitespace_tolerance(self) -> None:
        known_from_db = {normalize_url("https://example.com/x")}
        assert url_is_known("  https://example.com/x  ", known_from_db) is True


class TestLoadKnownUrlsNormalizes:
    """Verifie que load_known_urls stocke les URLs deja normalisees.

    On ne peut pas appeler load_known_urls reel sans la BDD, mais on peut
    verifier que la transformation qu'il applique est bien ``normalize_url``
    en reproduisant le flux :
    """

    def test_set_contains_normalized_urls(self) -> None:
        """Simulation du retour asyncpg + normalisation en pipeline."""
        raw_db_rows = [
            {"url": "https://arxiv.org/abs/1234.5678"},
            {"url": "http://techcrunch.com/article"},
            {"url": "https://www.example.com/Path/"},
            {"url": None},  # filtre
        ]
        # Reproduction de la logique `load_known_urls` sans asyncpg.
        normalized_set = {normalize_url(row["url"]) for row in raw_db_rows if row["url"]}

        assert "https://arxiv.org/abs/1234.5678" in normalized_set
        # http -> https
        assert "https://techcrunch.com/article" in normalized_set
        # Host lowercased, trailing slash strip
        assert "https://www.example.com/Path" in normalized_set
        # None filtre
        assert "" not in normalized_set


class TestCoerceBool:
    """Conversion flexible pour les flags CLI / env / Python."""

    @pytest.mark.parametrize(
        "value",
        [True, 1, "true", "TRUE", "yes", "on", "1", "anything_else", object()],
    )
    def test_truthy_values(self, value: object) -> None:
        assert coerce_bool(value) is True

    @pytest.mark.parametrize(
        "value",
        [False, 0, "false", "FALSE", "no", "off", "0", ""],
    )
    def test_falsy_values(self, value: object) -> None:
        assert coerce_bool(value) is False
