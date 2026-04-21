"""Spider Sustainable Web Design : posts + guidelines.

131 items issus de sustainablewebdesign.org, reference WSDG (Web
Sustainability Design Guidelines). La source mele deux formats :

- 50 ``posts`` (articles de blog classiques)
- 81 ``guidelines`` (fiches-guidelines numerotees, ex: ``2-3-integrate-
  sustainability-into-every-stage-of-the-ideation-process``)

Les deux sont exploitables comme articles Green IT : les guidelines
sont rediges en prose avec sections "Criteria", "Resources",
"Benefits" qui forment un contenu exploitable par le classifieur.

Caracteristiques techniques
---------------------------

- WordPress + Yoast SEO (sitemap_index.xml standard)
- robots.txt : ``User-agent: * / Disallow:`` = all allowed
- Sitemaps : ``post-sitemap.xml`` (50) et ``guidelines-sitemap.xml`` (81)
- HTML statique, theme WordPress classique
"""

from __future__ import annotations

from greentech.data.collectors.spiders.base import StaticArticleSpider


class SustainableWebSpider(StaticArticleSpider):
    """Spider sitemap-based pour sustainablewebdesign.org, langue EN."""

    name = "sustainable_web"
    allowed_domains = ["sustainablewebdesign.org"]
    source_nom = "Sustainable Web Design"
    langue = "en"

    # Deux sitemaps complementaires : posts classiques + guidelines.
    # On inclut les deux pour maximiser le volume exploitable.
    sitemap_urls = [
        "https://sustainablewebdesign.org/post-sitemap.xml",
        "https://sustainablewebdesign.org/guidelines-sitemap.xml",
    ]

    # Pas de filtre URL : les deux sitemaps ne contiennent QUE des URLs
    # d'articles (pas de pages auteurs/categories). Les defauts de la
    # base class filtrent deja les index sitemaps.
    article_url_pattern = None

    max_articles = 0
    download_delay = 2.0

    # Selectors : les defauts WordPress couvrent a la fois les posts et
    # les guidelines (meme theme, meme structure h1 + contenu article).
