"""Spider Climate Action Tech : communaute tech engagee pour le climat.

71 posts issus de climateaction.tech, regroupant les professionnels de
la tech engages sur les questions climatiques. Contenu 100% axe
tech + climat : strategies de communication climatique, outils
numeriques, pratiques de l'industrie.

Caracteristiques techniques
---------------------------

- WordPress + Yoast SEO (sitemap_index.xml)
- robots.txt : ``Disallow: /wp-admin/`` (tout le reste est allowe)
- Sitemap : ``post-sitemap.xml`` (71 URLs)
- URL pattern articles : ``/blog/<slug>/``
- Theme WordPress classique, HTML statique
"""

from __future__ import annotations

import re

from greentech.data.collectors.spiders.base import StaticArticleSpider


class ClimateActionTechSpider(StaticArticleSpider):
    """Spider sitemap-based pour climateaction.tech, langue EN."""

    name = "climate_action_tech"
    allowed_domains = ["climateaction.tech"]
    source_nom = "Climate Action Tech"
    langue = "en"

    sitemap_urls = [
        "https://climateaction.tech/post-sitemap.xml",
    ]

    # Filtre : seuls les articles /blog/<slug>/ sont pris. Les pages
    # d'accueil /blog/ sont exclues (ne matche pas [a-z][a-z0-9-]+).
    article_url_pattern = re.compile(r"https?://climateaction\.tech/blog/[a-z][a-z0-9-]+/?$")

    max_articles = 0
    download_delay = 2.0
