"""Base class pour les spiders de sites Green IT statiques.

Centralise la logique commune aux 4 spiders B2.3 :

- Decouverte d'URLs via sitemap XML OU pagination HTML
- Extraction titre/contenu/date/auteur via chaines de selecteurs CSS
  avec fallback en cascade (robuste aux refontes mineures)
- Fallback trafilatura sur le HTML complet si tous les selecteurs
  echouent (filet de securite identique a TechCrunch)
- Filtrage des articles trop courts (seuil ``MIN_CONTENT_LENGTH``
  coherent avec les autres collecteurs)
- Collecte dans ``self.collected_articles`` pour recuperation par
  l'orchestrator apres la fin du crawl

Design Scrapy HTTP (sans Playwright)
------------------------------------

Les 4 sites cibles (greenit.fr, greensoftware.foundation,
sustainablewebdesign.org, climateaction.tech) sont tous en HTML
statique verifie en B2.1 : aucun rendu JS n'est necessaire. On utilise
donc Scrapy + Twisted + HTTP async directement, bien plus rapide et
moins energivore que Playwright.

Un hook ``enable_playwright`` est laisse dans les sous-classes pour
activer Playwright sur une base per-spider si un site ajoute du JS
critique plus tard (ex: lazy-loading du contenu).
"""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree as ET

from loguru import logger
from scrapy import Request, Spider
from scrapy.http import Response

from greentech.data.collectors.url_precheck import coerce_bool, load_known_urls, url_is_known

# Longueur minimale du contenu pour qu'un article soit conserve. Seuil
# coherent avec les autres collecteurs du projet (Dev.to, TechCrunch,
# arxiv_collector) pour preserver l'homogeneite du dataset.
MIN_CONTENT_LENGTH = 300

# Longueur minimale du titre. Un titre < 5 chars signale presque
# toujours un cas limite (page liste, page 404 stylee en article, etc.).
MIN_TITLE_LENGTH = 5

# Taille maximale du titre avant troncature. La contrainte BDD est a
# 500 chars ; on se laisse 20 chars de marge pour les signes typographiques.
MAX_TITLE_LENGTH = 480


class StaticArticleSpider(Spider):
    """Spider generique pour sites Green IT en HTML statique.

    Les sous-classes doivent au minimum definir :

    - ``name`` : identifiant Scrapy du spider (ex: ``"greenit_fr"``)
    - ``allowed_domains`` : liste des domaines autorises
    - ``source_nom`` : nom lisible de la source pour la BDD / les logs
    - ``langue`` : code ISO 2 lettres (``"fr"``, ``"en"``, ...)
    - Une methode de discovery :
      * ``sitemap_urls`` + ``article_url_pattern`` (regex articles)
      * OU ``pagination_start_url`` + ``pagination_max_pages`` +
        ``pagination_article_selector``

    Les selecteurs d'extraction (``title_selectors``,
    ``content_selectors``, etc.) ont des valeurs par defaut compatibles
    avec la plupart des themes WordPress (cas majoritaire des 4 sites).
    """

    # --- Config per-site (override dans les sous-classes) ---

    source_nom: str = ""
    langue: str = "en"

    # Mode discovery n1 : sitemap XML
    sitemap_urls: list[str] = []
    article_url_pattern: re.Pattern[str] | None = None  # match URLs d'articles

    # Mode discovery n2 : pagination HTML
    pagination_start_url: str = ""
    pagination_max_pages: int = 0
    pagination_article_selector: str = ""  # ex: "a.article::attr(href)"
    pagination_next_format: str = "/page/{page}/"  # format pour construire les pages suivantes

    # Nombre max d'articles a scraper par spider. 0 = illimite (on prend
    # tout ce que le sitemap / la pagination ramenent).
    max_articles: int = 0

    # Scraping ethique : delai entre requetes (surcharge par site si besoin).
    download_delay: float = 2.0

    # Pre-check BDD : avant de scheduler le scraping d'un article, on
    # verifie si son URL est deja en table ``articles``. Si oui, on skip
    # le fetch (gain de temps massif sur les re-runs du pipeline). Le
    # garde-fou ``ON CONFLICT (url) DO NOTHING`` cote SQL ingester reste
    # la derniere ligne de defense, mais ce pre-check evite le cout
    # reseau + parsing HTML pour les articles deja connus.
    #
    # Par defaut True : lors d'une premiere collecte, la BDD est vide
    # donc le pre-check ne filtre rien (cout negligeable). Lors d'un
    # re-run, il saute la grande majorite des articles deja scrapes.
    #
    # Mettre a False (via CLI ou override subclasse) pour forcer un
    # re-scrape complet, utile pour :
    # - refetcher un article dont le contenu a ete mis a jour cote source
    # - diagnostiquer un probleme d'extraction (selecteurs, encodage, ...)
    # - generer des mesures reproductibles (carbon footprint d'une
    #   collecte complete pour comparaison)
    skip_existing: bool = True

    # --- Selectors d'extraction (override si besoin) ---

    # Sequence de selecteurs testee dans l'ordre pour extraire le TITRE.
    # Le premier selecteur non-vide l'emporte. Compatible themes WordPress
    # standards + OpenGraph en fallback final.
    title_selectors: list[str] = [
        "h1.entry-title::text",
        "h1.post-title::text",
        "article h1::text",
        "main h1::text",
        "h1::text",
    ]

    # Sequence de selecteurs pour extraire le CONTENU. Chaque selecteur
    # designe un conteneur d'article (article, main, .entry-content, ...) ;
    # on extrait ensuite tous les ``<p>`` descendants via XPath pour
    # recuperer le texte INCLUANT les enfants (``<a>``, ``<em>``, ``<strong>``,
    # etc.). L'approche ``::text`` CSS ne capture que les text nodes directs
    # et perd tout le texte dans les liens, ce qui tronque massivement les
    # articles modernes.
    content_selectors: list[str] = [
        ".entry-content",
        ".post-content",
        "article .content",
        "article",
        "main article",
        "main",
    ]

    # Selecteurs DATE. Les themes WordPress exposent la date via <time
    # datetime="...">, meta article:published_time, ou un span dedie.
    date_selectors: list[str] = [
        "time::attr(datetime)",
        'meta[property="article:published_time"]::attr(content)',
        'meta[name="date"]::attr(content)',
        ".entry-date::attr(datetime)",
    ]

    # Selecteurs AUTEUR. Optionnel : un site peut ne pas afficher
    # d'auteur, auquel cas on laisse None.
    author_selectors: list[str] = [
        'meta[name="author"]::attr(content)',
        'a[rel="author"]::text',
        ".author-name::text",
        ".entry-author::text",
        ".byline a::text",
    ]

    # --- Scrapy custom settings par defaut ---

    # Les sous-classes heritent de ces settings et peuvent les surcharger
    # via leur propre ``custom_settings``. Config minimaliste : pas de
    # Playwright (handler HTTP standard), respect strict du robots.txt,
    # concurrency reduite pour rester courtois.
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS": 2,
        "DOWNLOAD_DELAY": 2.0,
        "LOG_LEVEL": "WARNING",
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 2,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.collected_articles: list[dict[str, Any]] = []
        # Compteurs d'erreurs pour telemetrie (cf. TechCrunch scraper).
        self.error_stats: dict[str, int] = {
            "missing_title": 0,
            "empty_content": 0,
            "http_error": 0,
            "parsing_error": 0,
        }
        # URLs deja vues pour dedup global (un article peut apparaitre
        # dans plusieurs sitemaps ou etre liste dans plusieurs pages).
        self._seen_urls: set[str] = set()
        # URLs deja presentes en BDD (chargees au demarrage si
        # ``skip_existing=True``). Le set est vide si le pre-check est
        # desactive, et les URLs deja connues sont filtrees avant le
        # scheduling de Request HTTP.
        self._db_existing_urls: set[str] = set()
        # Nombre d'URLs sautees par le pre-check BDD, pour telemetrie.
        self._skipped_existing: int = 0

        # Scrapy propose un override par argument CLI :
        # ``scrapy crawl greenit_fr -a skip_existing=false`` ou via spider
        # init. Les kwargs Scrapy sont stocks comme attributs de la
        # classe ; on les convertit en bool ici pour robustesse.
        skip_kw = kwargs.get("skip_existing")
        if skip_kw is not None:
            self.skip_existing = coerce_bool(skip_kw)

    # --- Discovery ---

    async def start(self):
        """Lance la decouverte d'URLs selon le mode configure.

        Deux modes mutuellement exclusifs :

        - Si ``sitemap_urls`` est configure, on fetch chaque sitemap XML
          et on en extrait les URLs d'articles via ``article_url_pattern``.
        - Sinon si ``pagination_start_url`` est configure, on pagine sur
          les pages de listing et on extrait les liens d'articles via
          ``pagination_article_selector``.

        Si ``skip_existing=True`` (defaut), on pre-charge toutes les URLs
        deja presentes en table ``articles`` pour eviter de re-scraper
        ce qu'on a deja. Cette etape ajoute ~1s au demarrage (une query
        SQL sur l'index unique ``url``) mais economise potentiellement
        des dizaines de minutes sur les re-runs.

        Yields:
            Request Scrapy pointant soit sur un sitemap XML, soit sur une
            page de listing HTML, avec le callback approprie.
        """
        if self.skip_existing:
            self._db_existing_urls = await load_known_urls(self.source_nom)
        if self.sitemap_urls:
            for sitemap_url in self.sitemap_urls:
                yield Request(
                    sitemap_url,
                    callback=self._parse_sitemap,
                    errback=self._on_error,
                )
        elif self.pagination_start_url:
            # On genere toutes les URLs de pagination d'un coup : les
            # callbacks extraient les liens article et suivent.
            for page_num in range(1, self.pagination_max_pages + 1):
                url = (
                    self.pagination_start_url
                    if page_num == 1
                    else self.pagination_start_url.rstrip("/")
                    + self.pagination_next_format.format(page=page_num)
                )
                yield Request(
                    url,
                    callback=self._parse_listing,
                    errback=self._on_error,
                )
        else:
            logger.error(
                f"Spider '{self.name}' : aucune config discovery. "
                "Definir sitemap_urls OU pagination_start_url."
            )

    def _parse_sitemap(self, response: Response):
        """Extrait les URLs d'articles d'un sitemap XML.

        Supporte deux cas :
        - Sitemap standard ``<urlset>`` avec des ``<url><loc>...``
        - Sitemap index ``<sitemapindex>`` avec des sub-sitemaps a
          recursivement parser.

        Les URLs retournees sont filtrees via ``article_url_pattern``
        (regex) pour ne garder que les articles, pas les pages
        d'accueil/categorie/auteur.

        Yields:
            Request vers chaque URL d'article (callback = parse_article)
            ou vers chaque sub-sitemap (callback = _parse_sitemap).
        """
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as exc:
            logger.error(f"Sitemap XML invalide : {response.url} ({exc})")
            self.error_stats["parsing_error"] += 1
            return

        # Namespace sitemaps.org (standard) - les XML sitemaps le declarent.
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        nsless_tag = root.tag.split("}", 1)[-1] if "}" in root.tag else root.tag

        if nsless_tag == "sitemapindex":
            # Index sitemap : parser chaque sub-sitemap
            for sub_sitemap in root.findall("sm:sitemap/sm:loc", ns):
                url = (sub_sitemap.text or "").strip()
                if url:
                    yield Request(
                        url,
                        callback=self._parse_sitemap,
                        errback=self._on_error,
                    )
            return

        # Sitemap standard : extraire les URLs
        url_count = 0
        yielded_count = 0
        skipped_existing_local = 0
        for url_elem in root.findall("sm:url/sm:loc", ns):
            url = (url_elem.text or "").strip()
            if not url:
                continue
            url_count += 1

            if self.article_url_pattern and not self.article_url_pattern.search(url):
                continue

            # Pre-check BDD : evite le cout reseau + parsing HTML pour
            # les articles deja connus (re-run du pipeline). Comparaison
            # normalisee (http/https equivalents, trailing slash ignore).
            if url_is_known(url, self._db_existing_urls):
                skipped_existing_local += 1
                self._skipped_existing += 1
                continue

            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            if self.max_articles and yielded_count >= self.max_articles:
                break

            yielded_count += 1
            yield Request(
                url,
                callback=self.parse_article,
                errback=self._on_error,
            )

        logger.info(
            f"[{self.source_nom}] Sitemap {response.url} : "
            f"{url_count} URLs totales, "
            f"{skipped_existing_local} deja en BDD (skip), "
            f"{yielded_count} articles a scraper"
        )

    def _parse_listing(self, response: Response):
        """Extrait les liens d'articles d'une page de listing paginee.

        Utilise ``pagination_article_selector`` pour identifier les liens
        vers les articles individuels. Les URLs sont resolues en absolu
        via ``response.urljoin`` pour tolerer les liens relatifs.

        Yields:
            Request vers chaque URL d'article unique (callback = parse_article).
        """
        links = response.css(self.pagination_article_selector).getall()
        yielded_count = 0
        skipped_existing_local = 0

        for href in links:
            if not href:
                continue
            url = response.urljoin(href.strip())

            # Filtre par regex si configure : exclut les liens vers des
            # pages non-article (pagination, auteur, categorie, ...).
            if self.article_url_pattern and not self.article_url_pattern.search(url):
                continue

            # Pre-check BDD : evite fetch + parsing pour les articles
            # deja connus (re-run). Comparaison normalisee.
            if url_is_known(url, self._db_existing_urls):
                skipped_existing_local += 1
                self._skipped_existing += 1
                continue

            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            if self.max_articles and len(self._seen_urls) > self.max_articles:
                break

            yielded_count += 1
            yield Request(
                url,
                callback=self.parse_article,
                errback=self._on_error,
            )

        logger.info(
            f"[{self.source_nom}] Listing {response.url} : "
            f"{skipped_existing_local} deja en BDD (skip), "
            f"{yielded_count} articles a scraper"
        )

    # --- Parsing d'un article individuel ---

    def parse_article(self, response: Response):
        """Parse un article HTML et normalise les champs extraits.

        Applique la chaine de selecteurs pour titre/contenu/date/auteur,
        fallback trafilatura si le contenu est trop court, et rejette
        les articles qui ne respectent pas les seuils minimaux. Les
        articles valides sont ajoutes a ``self.collected_articles``
        pour recuperation par l'orchestrator.

        Args:
            response: Reponse Scrapy contenant le HTML de l'article.
        """
        titre = self._extract_title(response)
        if not titre or len(titre) < MIN_TITLE_LENGTH:
            self.error_stats["missing_title"] += 1
            logger.debug(f"[{self.source_nom}] Article ignore (titre manquant) : {response.url}")
            return

        contenu = self._extract_content(response)
        if len(contenu) < MIN_CONTENT_LENGTH:
            self.error_stats["empty_content"] += 1
            logger.debug(
                f"[{self.source_nom}] Article ignore "
                f"({len(contenu)} chars < {MIN_CONTENT_LENGTH}) : {titre[:60]}"
            )
            return

        date_publication = self._extract_first(response, self.date_selectors)
        auteur = self._extract_first(response, self.author_selectors)

        # Description : OpenGraph ou meta description (si present).
        description = (
            response.css('meta[property="og:description"]::attr(content)').get()
            or response.css('meta[name="description"]::attr(content)').get()
            or ""
        )

        article = {
            "titre": titre[:MAX_TITLE_LENGTH],
            "url": response.url,
            "description": description[:500] if description else "",
            "contenu": contenu,
            "date_publication": date_publication,
            "auteur": auteur,
            "source_nom": self.source_nom,
            "langue": self.langue,
        }
        self.collected_articles.append(article)
        logger.info(f"[{self.source_nom}] Scrape OK ({len(contenu)} chars) : {titre[:80]}")

    # --- Extraction utils ---

    def _extract_title(self, response: Response) -> str:
        """Extrait le titre via la chaine ``title_selectors`` + fallback og:title.

        Returns:
            Titre nettoye (espaces normalises), ou chaine vide si rien
            trouve.
        """
        for selector in self.title_selectors:
            candidate = response.css(selector).get()
            if candidate:
                cleaned = _normalize_whitespace(candidate)
                if cleaned:
                    return cleaned

        og_title = response.css('meta[property="og:title"]::attr(content)').get()
        return _normalize_whitespace(og_title) if og_title else ""

    def _extract_content(self, response: Response) -> str:
        """Extrait le corps d'article via selecteurs + fallback trafilatura.

        Pour chaque selecteur de ``content_selectors`` (qui designe un
        conteneur d'article), on extrait **tous les paragraphes
        descendants** avec leur texte complet (via XPath ``.//p//text()``
        qui capture aussi le texte dans les enfants comme ``<a>``,
        ``<em>``, ``<strong>``, contrairement au selecteur CSS ``::text``
        qui ne capture que les text nodes directs).

        Le premier selecteur qui atteint ``MIN_CONTENT_LENGTH`` l'emporte.
        Si tous echouent, fallback trafilatura sur le HTML complet.

        Returns:
            Texte du corps d'article, ou chaine vide si rien exploitable.
        """
        for selector in self.content_selectors:
            container = response.css(selector)
            if not container:
                continue
            # Extraction des blocs structures : paragraphes, titres, items
            # de liste, citations. Un article WordPress / site Green IT
            # mele souvent ces formes (les guidelines SWD sont quasiment
            # uniquement des <h2>+<ul>, sans <p>). XPath ``.//text()`` sur
            # chaque bloc capture aussi le texte dans les enfants (<a>,
            # <em>, <strong>), ce que ``::text`` CSS ne fait pas.
            block_texts: list[str] = []
            for block in container.css("p, h2, h3, h4, h5, li, blockquote"):
                text_parts = block.xpath(".//text()").getall()
                joined = _normalize_whitespace(" ".join(text_parts))
                if joined:
                    block_texts.append(joined)
            text = "\n\n".join(block_texts)
            if len(text) >= MIN_CONTENT_LENGTH:
                return text

        try:
            from trafilatura import extract

            extracted = extract(
                response.text,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if extracted and len(extracted.strip()) >= MIN_CONTENT_LENGTH:
                logger.debug(
                    f"[{self.source_nom}] Fallback trafilatura "
                    f"({len(extracted)} chars) : {response.url}"
                )
                return extracted.strip()
        except Exception as exc:
            logger.debug(f"[{self.source_nom}] Fallback trafilatura echec : {response.url} ({exc})")

        return ""

    @staticmethod
    def _extract_first(response: Response, selectors: list[str]) -> str | None:
        """Retourne la premiere valeur non-vide trouvee via la chaine de selecteurs.

        Args:
            response: Reponse Scrapy.
            selectors: Liste de selecteurs CSS a essayer dans l'ordre.

        Returns:
            Valeur nettoyee du premier selecteur non-vide, ou ``None`` si
            aucun selecteur ne ramene de valeur.
        """
        for selector in selectors:
            value = response.css(selector).get()
            if value:
                cleaned = _normalize_whitespace(value)
                if cleaned:
                    return cleaned
        return None

    def _on_error(self, failure) -> None:
        """Categorise les erreurs Scrapy pour telemetrie.

        Args:
            failure: Twisted Failure wrapper sur l'exception Scrapy.
        """
        self.error_stats["http_error"] += 1
        logger.warning(
            f"[{self.source_nom}] HTTP error sur {failure.request.url} : {str(failure.value)[:120]}"
        )


def _normalize_whitespace(text: str) -> str:
    """Normalise les espaces/retours a la ligne d'un fragment texte.

    Args:
        text: Texte brut (peut contenir \\n, tab, espaces multiples).

    Returns:
        Texte avec espaces normalises a un seul espace, trim.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()
