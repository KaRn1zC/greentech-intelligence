"""Spider Green Software Foundation : articles de la fondation GSF.

170 articles issus de la Green Software Foundation, 100% dedies au
green software engineering (SCI standard, carbon-aware computing,
mesure d'emissions, etc.). Source anglophone de reference pour le
classifieur.

Caracteristiques techniques
---------------------------

- Site statique (SSG, HTML pre-rendu)
- Pas de sitemap.xml accessible (404), on utilise la pagination HTML
- 17 pages : ``/articles/`` (page 1), ``/articles/2``, ..., ``/articles/17``
- Chaque page liste 10 articles avec liens ``/articles/<slug>/``
- robots.txt permissif (content-signal Cloudflare sans regle bloquante)
- Titre : ``<h1>`` standard
- Contenu : corps article dans ``<article>`` ou ``<main>``
- Date : meta ``article:published_time`` ou bloc dedicace
"""

from __future__ import annotations

import re

from greentech.data.collectors.spiders.base import StaticArticleSpider


class GreenSoftwareSpider(StaticArticleSpider):
    """Spider pagination-based pour greensoftware.foundation, langue EN."""

    name = "greensoftware"
    allowed_domains = ["greensoftware.foundation"]
    source_nom = "Green Software Foundation"
    langue = "en"

    # Pagination HTML : page 1 = /articles/, page N = /articles/N
    pagination_start_url = "https://greensoftware.foundation/articles/"
    pagination_max_pages = 17  # Verifie en B2.1 (pagination : "Page 1 of 17")
    pagination_next_format = "/{page}"  # Sans trailing slash, pas comme WordPress

    # Selecteur large : tous les liens sous /articles/. On filtre ensuite
    # les liens de pagination (pures digits) via ``article_url_pattern``.
    pagination_article_selector = 'a[href^="/articles/"]::attr(href)'

    # Filtre : URL doit commencer par /articles/ ET contenir au moins une
    # lettre apres (exclut les pages de pagination /articles/N ou N = digits).
    article_url_pattern = re.compile(
        r"https?://greensoftware\.foundation/articles/[a-z][a-z0-9-]+/?$"
    )

    max_articles = 0  # Tout prendre
    download_delay = 2.0

    # Selectors : les defauts WordPress suffisent pour GSF (theme moderne
    # avec <h1> pour titre et <article><p> pour contenu).
