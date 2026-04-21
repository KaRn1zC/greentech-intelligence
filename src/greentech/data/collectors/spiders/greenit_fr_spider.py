"""Spider GreenIT.fr : blog francophone reference sur le Green IT.

Source francophone majeure pour le dataset : 1 001 posts dedies a
l'eco-conception web, l'impact environnemental du numerique, les etudes
Green IT. Contenu 100% pertinent pour le classifieur (tous les articles
sont des candidats Green IT positifs).

Caracteristiques techniques
---------------------------

- WordPress avec theme custom, HTML statique (pas de JS critique)
- Sitemap principal : ``post-sitemap.xml`` (500 URLs) et ``post-sitemap3.xml``
  (501 URLs). Le ``post-sitemap2.xml`` historique est fige a 2013.
- URL pattern articles : ``/YYYY/MM/DD/slug/``
- Pas de robots.txt (convention = allow all, verifie en B2.1)
- Titre : ``<h1>`` standard WordPress
- Contenu : ``.entry-content p``
- Date : extraite de l'URL (``/2025/11/04/``) ou du bloc ``<time>``
- Auteur : generalement ``Frederic Bordage`` (unique contributeur)
"""

from __future__ import annotations

import re

from greentech.data.collectors.spiders.base import StaticArticleSpider


class GreenItFrSpider(StaticArticleSpider):
    """Spider WordPress pour greenit.fr, sitemap-based, langue FR."""

    name = "greenit_fr"
    allowed_domains = ["greenit.fr", "www.greenit.fr"]
    source_nom = "GreenIT.fr"
    langue = "fr"

    # post-sitemap2.xml est figee (2013) mais on l'inclut au cas ou les
    # articles retro seraient utiles. Les doublons sont dedupliques par
    # ``_seen_urls`` dans la base class.
    sitemap_urls = [
        "https://www.greenit.fr/post-sitemap.xml",
        "https://www.greenit.fr/post-sitemap2.xml",
        "https://www.greenit.fr/post-sitemap3.xml",
    ]

    # Filtre regex : seules les URLs /YYYY/MM/DD/slug/ sont des articles.
    # Les autres URLs du sitemap (pages auteurs, categories, index) sont
    # exclues. La regex tolere la presence ou non du trailing slash.
    article_url_pattern = re.compile(r"https?://(?:www\.)?greenit\.fr/\d{4}/\d{2}/\d{2}/[^/]+/?$")

    # Pas de plafond : on veut les 1 001 articles complets pour maximiser
    # le volume de positifs Green IT dans le dataset.
    max_articles = 0

    # Courtoisie : 2s entre chaque requete = ~35 min pour 1 001 articles,
    # conforme a la bonne pratique de scraping ethique.
    download_delay = 2.0

    # Overrides selectors pour ce theme specifique si necessaire. Les
    # defauts WordPress de la base class suffisent pour greenit.fr.
