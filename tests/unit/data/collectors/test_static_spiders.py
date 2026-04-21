"""Tests unitaires des spiders Scrapy statiques (B2.3).

Couvrent la logique de parsing et de filtrage commune aux 4 spiders,
sans declencher d'appels reseau reels. Les tests d'integration (scrape
live sur chaque site) sont executes separement via smoke tests manuels.
"""

from __future__ import annotations

import re

import pytest
from scrapy.http import HtmlResponse, Request, TextResponse

from greentech.data.collectors.spiders.base import (
    MIN_CONTENT_LENGTH,
    StaticArticleSpider,
    _normalize_whitespace,
)
from greentech.data.collectors.spiders.climate_action_tech_spider import (
    ClimateActionTechSpider,
)
from greentech.data.collectors.spiders.greenit_fr_spider import GreenItFrSpider
from greentech.data.collectors.spiders.greensoftware_spider import GreenSoftwareSpider
from greentech.data.collectors.spiders.sustainable_web_spider import (
    SustainableWebSpider,
)


def _make_html_response(url: str, body: str) -> HtmlResponse:
    """Construit une HtmlResponse Scrapy synthetique pour les tests."""
    return HtmlResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=Request(url),
    )


def _make_xml_response(url: str, body: str) -> TextResponse:
    """Construit une TextResponse XML pour tester le parsing sitemap."""
    return TextResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=Request(url),
    )


class TestNormalizeWhitespace:
    """Helper de normalisation d'espaces."""

    def test_strip_multiple_spaces(self) -> None:
        assert _normalize_whitespace("hello    world") == "hello world"

    def test_strip_newlines_and_tabs(self) -> None:
        assert _normalize_whitespace("line1\n\n\tline2") == "line1 line2"

    def test_empty_returns_empty(self) -> None:
        assert _normalize_whitespace("") == ""
        assert _normalize_whitespace(None) == ""  # type: ignore[arg-type]


class TestStaticSpiderConfig:
    """Verifie la config par site est coherente avec B2.1."""

    def test_greenit_fr_is_french_sitemap_based(self) -> None:
        spider = GreenItFrSpider()
        assert spider.langue == "fr"
        assert spider.source_nom == "GreenIT.fr"
        assert len(spider.sitemap_urls) >= 2  # post-sitemap + post-sitemap3
        assert spider.article_url_pattern is not None
        # La regex doit matcher /YYYY/MM/DD/slug/
        assert spider.article_url_pattern.search("https://www.greenit.fr/2025/11/04/my-article/")
        # Elle ne doit pas matcher des URLs d'auteur ou de categorie
        assert not spider.article_url_pattern.search(
            "https://www.greenit.fr/author/frederic-bordage/"
        )

    def test_greensoftware_is_english_pagination_based(self) -> None:
        spider = GreenSoftwareSpider()
        assert spider.langue == "en"
        assert spider.source_nom == "Green Software Foundation"
        assert spider.pagination_start_url.endswith("/articles/")
        assert spider.pagination_max_pages == 17
        # Le pattern rejette les pages de pagination (digits uniquement)
        assert spider.article_url_pattern.search(
            "https://greensoftware.foundation/articles/valid-slug/"
        )
        assert not spider.article_url_pattern.search("https://greensoftware.foundation/articles/2")

    def test_sustainable_web_has_both_sitemaps(self) -> None:
        spider = SustainableWebSpider()
        assert spider.langue == "en"
        assert spider.source_nom == "Sustainable Web Design"
        assert len(spider.sitemap_urls) == 2
        # Pas de filtre URL : les deux sitemaps n'exposent que des articles
        assert spider.article_url_pattern is None

    def test_climate_action_tech_filters_blog_slugs_only(self) -> None:
        spider = ClimateActionTechSpider()
        assert spider.langue == "en"
        assert spider.source_nom == "Climate Action Tech"
        assert spider.article_url_pattern is not None
        assert spider.article_url_pattern.search("https://climateaction.tech/blog/my-post/")
        # Page /blog/ (index) ne doit pas matcher
        assert not spider.article_url_pattern.search("https://climateaction.tech/blog/")


class TestSitemapParsing:
    """Tests du parsing de sitemap XML et filtrage des URLs."""

    SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://example.com/blog/article-1/</loc></url>
    <url><loc>https://example.com/blog/article-2/</loc></url>
    <url><loc>https://example.com/author/john/</loc></url>
</urlset>"""

    SITEMAP_INDEX_XML = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap><loc>https://example.com/post-sitemap.xml</loc></sitemap>
    <sitemap><loc>https://example.com/page-sitemap.xml</loc></sitemap>
</sitemapindex>"""

    def test_urls_filtered_by_pattern(self) -> None:
        """Seules les URLs matchant article_url_pattern sont yieldees."""

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"
            sitemap_urls = ["https://example.com/sitemap.xml"]
            article_url_pattern = re.compile(r"https?://example\.com/blog/[^/]+/?$")

        spider = _Spider()
        response = _make_xml_response("https://example.com/sitemap.xml", self.SITEMAP_XML)
        requests = list(spider._parse_sitemap(response))
        urls = [r.url for r in requests]
        assert "https://example.com/blog/article-1/" in urls
        assert "https://example.com/blog/article-2/" in urls
        assert "https://example.com/author/john/" not in urls

    def test_sitemap_index_yields_sub_sitemaps(self) -> None:
        """Un sitemapindex doit yielder un Request par sub-sitemap."""

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"
            sitemap_urls = ["https://example.com/sitemap_index.xml"]

        spider = _Spider()
        response = _make_xml_response(
            "https://example.com/sitemap_index.xml", self.SITEMAP_INDEX_XML
        )
        requests = list(spider._parse_sitemap(response))
        urls = [r.url for r in requests]
        assert "https://example.com/post-sitemap.xml" in urls
        assert "https://example.com/page-sitemap.xml" in urls

    def test_skip_existing_filters_known_urls(self) -> None:
        """URLs presentes dans ``_db_existing_urls`` doivent etre sautees."""
        from greentech.data.collectors.url_precheck import normalize_url

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"
            sitemap_urls = ["https://example.com/sitemap.xml"]
            article_url_pattern = re.compile(r"https?://example\.com/blog/[^/]+/?$")

        spider = _Spider()
        # Le set interne stocke les URLs normalisees (comme retourne par
        # ``load_known_urls``). On simule ce comportement dans le test.
        spider._db_existing_urls = {normalize_url("https://example.com/blog/article-1/")}

        response = _make_xml_response("https://example.com/sitemap.xml", self.SITEMAP_XML)
        requests = list(spider._parse_sitemap(response))
        urls = [r.url for r in requests]

        # article-1 doit etre saute, article-2 yieldee
        assert "https://example.com/blog/article-1/" not in urls
        assert "https://example.com/blog/article-2/" in urls
        assert spider._skipped_existing == 1

    def test_skip_existing_default_true(self) -> None:
        """Le flag ``skip_existing`` est active par defaut."""
        spider = GreenItFrSpider()
        assert spider.skip_existing is True

    def test_skip_existing_kwarg_override(self) -> None:
        """Scrapy ``-a skip_existing=false`` doit desactiver le pre-check."""

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        # String "false" -> bool False (comme Scrapy CLI le passe)
        spider = _Spider(skip_existing="false")
        assert spider.skip_existing is False

        spider = _Spider(skip_existing="true")
        assert spider.skip_existing is True

        spider = _Spider(skip_existing=False)
        assert spider.skip_existing is False


class TestContentExtraction:
    """Tests de l'extraction titre/contenu/date depuis une page article."""

    ARTICLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Test - Mon site</title>
    <meta property="og:title" content="Article Fallback OG">
    <meta property="article:published_time" content="2024-06-15T10:00:00Z">
    <meta name="author" content="Alice Martin">
</head>
<body>
    <header><h1>Logo du site</h1></header>
    <article class="entry-content">
        <h1 class="entry-title">Mon article sur le Green IT</h1>
        <p>Premier paragraphe qui explique le contexte general. Ce paragraphe
        contient <a href="/lien">un lien</a> et <em>du texte emphase</em> que
        l'extraction doit capturer completement.</p>
        <h2>Premier sous-titre</h2>
        <p>Deuxieme paragraphe avec du contenu additionnel pour atteindre la
        longueur minimale de 300 caracteres. Ajout de contenu additionnel
        supplementaire afin de s'assurer que le seuil minimum est atteint avec
        une marge suffisante pour tous les tests qui utilisent ce fixture.</p>
        <ul>
            <li>Premier item de liste important</li>
            <li>Second item de liste complementaire</li>
        </ul>
    </article>
    <footer>Footer du site</footer>
</body>
</html>"""

    def test_title_extracted_from_h1(self) -> None:
        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.ARTICLE_HTML)
        titre = spider._extract_title(response)
        assert titre == "Mon article sur le Green IT"

    def test_title_fallback_to_og_title(self) -> None:
        """Si aucun selecteur h1 ne match, fallback sur og:title."""

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"
            title_selectors: list[str] = [".nonexistent::text"]

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.ARTICLE_HTML)
        assert spider._extract_title(response) == "Article Fallback OG"

    def test_content_captures_p_h_and_li_tags(self) -> None:
        """L'extraction doit inclure les paragraphes, titres, items de liste."""

        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.ARTICLE_HTML)
        contenu = spider._extract_content(response)
        # Doit contenir les <p>
        assert "Premier paragraphe" in contenu
        assert "Deuxieme paragraphe" in contenu
        # Doit contenir le <h2>
        assert "Premier sous-titre" in contenu
        # Doit contenir les <li>
        assert "Premier item de liste important" in contenu
        assert "Second item de liste complementaire" in contenu
        # Doit capturer le texte dans les <a> et <em>
        assert "un lien" in contenu
        assert "du texte emphase" in contenu
        # Doit depasser MIN_CONTENT_LENGTH
        assert len(contenu) >= MIN_CONTENT_LENGTH

    def test_extract_first_finds_date_in_meta(self) -> None:
        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.ARTICLE_HTML)
        date = spider._extract_first(response, spider.date_selectors)
        assert date == "2024-06-15T10:00:00Z"

    def test_extract_first_finds_author(self) -> None:
        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.ARTICLE_HTML)
        auteur = spider._extract_first(response, spider.author_selectors)
        assert auteur == "Alice Martin"


class TestParseArticle:
    """Test du pipeline complet parse_article."""

    SHORT_ARTICLE_HTML = """<!DOCTYPE html>
<html><body><article><h1>Titre</h1><p>Court.</p></article></body></html>"""

    def test_rejects_short_content(self) -> None:
        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"

        spider = _Spider()
        response = _make_html_response("https://example.com/article/", self.SHORT_ARTICLE_HTML)
        # parse_article est un generator-like mais ici ne retourne rien
        result = spider.parse_article(response)
        # Force consommation si c'est un iter
        if result is not None:
            list(result) if hasattr(result, "__iter__") else None
        assert spider.error_stats["empty_content"] >= 1
        assert len(spider.collected_articles) == 0

    def test_accepts_valid_article(self) -> None:
        class _Spider(StaticArticleSpider):
            name = "test"
            source_nom = "Test"
            langue = "en"

        spider = _Spider()
        response = _make_html_response(
            "https://example.com/article/", TestContentExtraction.ARTICLE_HTML
        )
        spider.parse_article(response)
        assert len(spider.collected_articles) == 1
        art = spider.collected_articles[0]
        assert art["titre"] == "Mon article sur le Green IT"
        assert art["source_nom"] == "Test"
        assert art["langue"] == "en"
        assert art["url"] == "https://example.com/article/"
        assert art["auteur"] == "Alice Martin"


@pytest.mark.parametrize(
    "spider_cls",
    [GreenItFrSpider, GreenSoftwareSpider, SustainableWebSpider, ClimateActionTechSpider],
)
def test_all_spiders_inherit_from_base(spider_cls) -> None:
    """Meta-test : tous les spiders doivent heriter de StaticArticleSpider."""
    assert issubclass(spider_cls, StaticArticleSpider)
    spider = spider_cls()
    assert spider.source_nom  # Non vide
    assert spider.langue in {"fr", "en"}
