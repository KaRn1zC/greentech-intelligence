"""Module 2 : Scraping hybride RSS + Scrapy/Playwright sur TechCrunch Climate.

Architecture en deux temps pour allier robustesse, pertinence thematique et
conformite au referentiel C1 :

1. **Decouverte d'URLs** via le flux RSS officiel de la section Climate
   (``https://techcrunch.com/category/climate/feed/``). Le RSS est
   standardise, stable, et resiste aux refontes de la page d'index HTML.
   Les URLs sont ensuite filtrees en deux passes :

   - **Blacklist regex** : on elimine les articles promotionnels
     TechCrunch (tickets Disrupt, Startup Battlefield, savings campagnes)
     qui polluent le RSS mais n'ont aucun rapport avec le Green IT.
   - **Match mots-cles** : les mots-cles de la table ``search_config``
     (``type_source='scraping'``) filtrent encore les URLs dont le titre
     ou le summary RSS ne mentionnent pas le sujet vise. Si aucune URL ne
     matche, on fallback sur la liste post-blacklist pour ne pas rater
     l'integralite du flux.

2. **Scraping HTML** de chaque article individuel avec Scrapy + Playwright.
   Le navigateur headless charge la page d'article, attend le rendu React,
   puis Scrapy parse le DOM via une **chaine de selecteurs CSS avec
   fallback**. Si aucun selecteur ne ramene assez de contenu, on applique
   trafilatura sur le HTML brut en dernier recours pour extraire le corps
   de l'article meme si TechCrunch a refactor sa structure.

Cette architecture coche les criteres de certification C1 :
- "telechargement de l'HTML d'une ou plusieurs pages web visees par une
  action de scraping"
- "filtrage/parsing des donnees utiles dans les resultats obtenus depuis
  l'HTML collecte d'un site web (scraping)"

Robustesse Playwright
---------------------

Le run initial (avril 2026) a revele que Chromium crashait apres ~7 pages
(message ``BrowserType.launch: Connection closed while reading from the
driver``), entrainant la perte de toutes les URLs suivantes. Les mesures
suivantes ont ete ajoutees pour garantir un taux de succes stable :

- Timeouts plus genereux (60s navigation, 30s selector) car TechCrunch
  charge beaucoup de scripts tiers (analytics, ads) meme quand on bloque
  images/medias.
- ``PLAYWRIGHT_MAX_PAGES_PER_CONTEXT=5`` pour recycler le contexte
  periodiquement et eviter l'accumulation de state qui fait crasher
  Chromium.
- ``--disable-dev-shm-usage`` et ``--no-sandbox`` dans les launch options
  (evitent les crashes de memoire partagee sur Docker et sur Windows).
- Retries automatiques Scrapy (2 tentatives par URL).

"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

import feedparser
import httpx
from loguru import logger
from playwright.async_api import Request as PlaywrightRequest
from scrapy import Request, Spider
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from scrapy.utils.project import get_project_settings
from scrapy_playwright.page import PageMethod

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult, get_config_from_db
from greentech.data.collectors.url_precheck import load_known_urls, url_is_known
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# -----------------------------------------------------------------------------
# Parametres generaux
# -----------------------------------------------------------------------------

# URL du flux RSS pour la decouverte des articles (section Climate)
TECHCRUNCH_CLIMATE_RSS = "https://techcrunch.com/category/climate/feed/"

# Nombre maximum d'articles a scraper par session. Releve a 100 pour
# equilibrer le volume TechCrunch avec les autres sources du dataset
# (432 Guardian, 120 Dev.to). La pagination RSS (cf. MAX_RSS_PAGES) permet
# d'atteindre ce volume bien au-dela des 20 articles/page par defaut de
# WordPress.
MAX_ARTICLES = 100

# Nombre maximum de pages RSS a interroger via le parametre WordPress
# ``?paged=N``. Chaque page fournit jusqu'a 20 articles (comportement par
# defaut de WordPress). 10 pages maximum donnent donc 200 entrees RSS
# candidates, largement suffisant pour atteindre MAX_ARTICLES apres le
# filtrage blacklist + mots-cles.
MAX_RSS_PAGES = 10

# Pause entre deux requetes de pagination RSS. TechCrunch ne publie pas de
# rate limit pour les feeds RSS, mais on reste courtois avec 0.5s : la
# pagination complete (10 pages) prend donc ~5 secondes.
RSS_PAGINATION_DELAY = 0.5

# Delai entre requetes HTML (scraping ethique, en secondes)
DOWNLOAD_DELAY = 2.0

# Timeout HTTP pour l'appel RSS initial (pas le scraping HTML)
REQUEST_TIMEOUT = 20.0

# Longueur minimale du contenu extrait pour considerer l'article exploitable.
# En dessous on rejette, conforme au seuil Dev.to pour coherence inter-sources.
MIN_CONTENT_LENGTH = 300

# -----------------------------------------------------------------------------
# Parametres Playwright (robustesse)
# -----------------------------------------------------------------------------

# Timeout de navigation (page.goto). Le defaut Playwright de 30s etait trop
# serre pour les articles longs / live blogs de TechCrunch.
PLAYWRIGHT_NAVIGATION_TIMEOUT = 60_000  # 60 secondes

# Timeout d'attente du selecteur h1 apres la navigation.
PLAYWRIGHT_SELECTOR_TIMEOUT = 30_000  # 30 secondes

# Nombre maximum de pages par contexte Playwright avant recyclage. Un contexte
# qui accumule trop de state finit par crasher Chromium. Recycler tous les
# 5 articles permet de garder Chromium stable sur une session de 50 URLs.
PLAYWRIGHT_MAX_PAGES_PER_CONTEXT = 5

# Ressources dont le telechargement est bloque par Playwright : elles
# n'apportent rien au texte extrait mais generent des tasks asyncio qui
# peuvent rester en attente et declenchent des leaks de memoire Chromium.
_ABORTED_RESOURCE_TYPES = frozenset({"image", "media", "font", "stylesheet"})

# -----------------------------------------------------------------------------
# Filtres URL : blacklist promos TechCrunch
# -----------------------------------------------------------------------------

# Regex de detection des articles promotionnels TechCrunch qui arrivent dans
# le RSS Climate mais n'ont aucun rapport avec le Green IT. Un article qui
# matche UN SEUL de ces patterns est exclu des candidats au scraping.
PROMO_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"disrupt-20\d{2}", re.IGNORECASE),
    re.compile(r"startup-battlefield", re.IGNORECASE),
    re.compile(r"save-up-to-\$?\d+", re.IGNORECASE),
    re.compile(r"save-close-to-\$?\d+", re.IGNORECASE),
    re.compile(r"tc-disrupt-tickets?", re.IGNORECASE),
    re.compile(r"-(hours?|days?|weeks?)-left-", re.IGNORECASE),
    re.compile(r"final-\d+-(hour|day|week)s?-", re.IGNORECASE),
    re.compile(r"techcrunch-coverage-and-\$?\d+", re.IGNORECASE),
    re.compile(r"massive-ticket-savings", re.IGNORECASE),
    re.compile(r"early-bird-pricing", re.IGNORECASE),
]

# -----------------------------------------------------------------------------
# Selectors CSS : fallback en cascade
# -----------------------------------------------------------------------------

# Sequence de selecteurs pour extraire le TITRE de l'article. On essaie dans
# l'ordre ; le premier non-vide est retenu. Le fallback og:title est gere a
# part dans `parse_article`.
TITLE_SELECTORS: list[str] = [
    "h1.article-hero__title::text",
    "h1[class*='title']::text",
    "h1.wp-block-post-title::text",
    "h1::text",
]

# Sequence de selecteurs pour extraire le CONTENU de l'article. On applique
# chaque selecteur, on concatene les paragraphes, et on garde le premier
# resultat qui atteint `MIN_CONTENT_LENGTH`. Si aucun selecteur ne ramene
# assez de contenu, on fallback sur trafilatura (voir `_extract_content`).
CONTENT_SELECTORS: list[str] = [
    "div.entry-content p::text",
    "article[class*='article'] p::text",
    "main [class*='content'] p::text",
    "main p::text",
    "article p::text",
]


def _should_abort_request(request: PlaywrightRequest) -> bool:
    """Filtre Playwright : bloque les ressources non essentielles au scraping texte.

    Args:
        request: Requete Playwright sur le point d'etre emise.

    Returns:
        True pour abandonner la requete (images, medias, polices, CSS).
    """
    return request.resource_type in _ABORTED_RESOURCE_TYPES


def _is_promo_url(url: str) -> bool:
    """Retourne True si l'URL correspond a un article promotionnel TechCrunch.

    Les articles promotionnels (tickets Disrupt, Startup Battlefield, etc.)
    arrivent dans le flux RSS Climate mais n'ont aucune valeur pour le
    classifieur Green IT. Les filtrer en amont evite de gaspiller des
    requetes Playwright dessus.

    Args:
        url: URL candidate issue du RSS.

    Returns:
        True si l'URL matche au moins un pattern de ``PROMO_URL_PATTERNS``.
    """
    return any(pattern.search(url) for pattern in PROMO_URL_PATTERNS)


def _match_keywords(text: str, keywords: list[str]) -> bool:
    """Teste si un texte contient au moins un mot-cle (match insensible a la casse).

    Le match est permissif : on accepte soit la phrase entiere (ex. "Carbon
    Footprint Software" -> recherche la chaine complete), soit tous les mots
    significatifs (>= 4 caracteres) de la phrase present dans le texte.
    Cela evite de rater un article qui parle de "carbon footprint of software"
    juste parce que le titre RSS ne reproduit pas la phrase mot pour mot.

    Args:
        text: Titre + summary de l'entree RSS (deja en minuscules).
        keywords: Liste de mots-cles en minuscules.

    Returns:
        True si au moins un mot-cle match.
    """
    if not keywords:
        return True
    for needle in keywords:
        if needle in text:
            return True
        significant_words = [w for w in needle.split() if len(w) >= 4]
        if significant_words and all(w in text for w in significant_words):
            return True
    return False


def _extract_content(response: Response) -> str:
    """Extrait le corps d'article via une chaine de selecteurs CSS + fallback trafilatura.

    Teste chaque selecteur de ``CONTENT_SELECTORS`` dans l'ordre, concatene
    les paragraphes, et retourne le premier resultat qui depasse
    ``MIN_CONTENT_LENGTH``. Si tous les selecteurs echouent, on applique
    trafilatura sur le DOM rendu (``response.text``) comme filet de securite.
    Cette couche fallback absorbe les futures refontes de TechCrunch sans
    necessiter de modifier le code.

    Args:
        response: Reponse Scrapy contenant le DOM rendu par Playwright.

    Returns:
        Texte du corps d'article, ou chaine vide si rien n'a pu etre extrait.
    """
    for selector in CONTENT_SELECTORS:
        paragraphs = response.css(selector).getall()
        text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        if len(text) >= MIN_CONTENT_LENGTH:
            logger.debug(f"Contenu extrait via selecteur '{selector}' ({len(text)} chars)")
            return text

    # Dernier recours : trafilatura sur le HTML rendu par Playwright.
    # L'import est differe pour ne pas alourdir les appels qui n'en ont pas
    # besoin (la plupart des articles TechCrunch passent par les selecteurs).
    try:
        from trafilatura import extract

        extracted = extract(
            response.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted and len(extracted.strip()) >= MIN_CONTENT_LENGTH:
            logger.info(f"Fallback trafilatura applique ({len(extracted)} chars) : {response.url}")
            return extracted.strip()
    except Exception as exc:
        logger.debug(f"Fallback trafilatura echec sur {response.url} : {exc}")

    return ""


def _extract_title(response: Response) -> str:
    """Extrait le titre de l'article via une chaine de selecteurs CSS + og:title.

    Args:
        response: Reponse Scrapy contenant le DOM rendu.

    Returns:
        Titre nettoye, ou chaine vide si aucun selecteur ne donne de resultat.
    """
    for selector in TITLE_SELECTORS:
        candidate = response.css(selector).get()
        if candidate and candidate.strip():
            return candidate.strip()

    # Fallback final : metadonnee OpenGraph
    og_title = response.css('meta[property="og:title"]::attr(content)').get() or ""
    return og_title.strip()


class _IgnoreTaskDestroyedFilter(logging.Filter):
    """Supprime les messages asyncio benins inherents a Playwright sur Windows.

    Deux patterns sont masques car ils n'affectent ni les articles collectes
    ni l'integrite des donnees (taux de succes 100% observe en presence de
    ces logs). Le bruit vient de la combinaison Twisted + scrapy-playwright
    + Proactor event loop Windows :

    - ``"Task was destroyed but it is pending"`` : tasks asyncio internes
      a Playwright qui restent en attente quand Twisted arrete le reactor.
    - ``"_ProactorBaseWritePipeTransport._loop_writing"`` : Chromium ferme
      un pipe IPC (typiquement lors du recycling de contexte configure via
      ``PLAYWRIGHT_MAX_PAGES_PER_CONTEXT``) pendant qu'asyncio avait une
      ecriture en attente. L'erreur est loggee mais l'operation a bien eu
      lieu cote browser.
    """

    _NOISE_PATTERNS = (
        "Task was destroyed but it is pending",
        "_ProactorBaseWritePipeTransport._loop_writing",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern in message for pattern in self._NOISE_PATTERNS)


class TechCrunchArticleSpider(Spider):
    """Spider Scrapy + Playwright pour scraper les articles TechCrunch Climate.

    Recoit une liste d'URLs via le constructeur (fournie par le flux RSS
    apres filtrage), charge chaque page avec Playwright pour gerer le rendu
    React, puis extrait les champs metier via la chaine de selecteurs
    ``TITLE_SELECTORS`` et ``CONTENT_SELECTORS`` avec fallback trafilatura.
    """

    name = "techcrunch_article"
    allowed_domains = ["techcrunch.com"]

    custom_settings = {
        "DOWNLOAD_HANDLERS": {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
        "CONCURRENT_REQUESTS": 1,
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "WARNING",
        # --- Retries Scrapy ---
        # 2 tentatives supplementaires sur erreurs reseau et timeouts : le
        # premier essai peut echouer sur un timeout Playwright isole mais le
        # second passe souvent, notamment apres recyclage du contexte.
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 2,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
    }

    def __init__(self, urls: list[str] | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_urls_list: list[str] = urls or []
        self._scraping_user_agent: str = get_settings().scraping_user_agent
        self.collected_articles: list[dict[str, Any]] = []
        # Compteurs d'erreurs par categorie pour la telemetrie finale (niveau 4).
        # Permet de distinguer les vraies erreurs Playwright (crash, timeout)
        # des rejets metier (titre/contenu manquant) dans le bilan du run.
        self.error_stats: dict[str, int] = {
            "timeout": 0,
            "browser_crash": 0,
            "missing_title": 0,
            "empty_content": 0,
            "other": 0,
        }

    async def start(self):
        """Genere une requete Playwright par URL d'article a scraper.

        Remplace ``start_requests()`` depuis Scrapy 2.13 pour autoriser la
        generation asynchrone de requetes. Chaque requete inclut les
        instructions Playwright (page meta, wait_for_selector h1, errback).
        """
        for url in self.start_urls_list:
            yield Request(
                url,
                callback=self.parse_article,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_methods": [
                        PageMethod(
                            "wait_for_selector",
                            "h1",
                            timeout=PLAYWRIGHT_SELECTOR_TIMEOUT,
                        ),
                    ],
                },
                headers={"User-Agent": self._scraping_user_agent},
                errback=self._on_error,
            )

    async def parse_article(self, response: Response) -> None:
        """Parse le HTML d'une page d'article TechCrunch avec fallback extraction.

        Applique la chaine de selecteurs pour titre et contenu, fallback
        trafilatura si necessaire, et rejette les articles dont le contenu
        extrait est en dessous de ``MIN_CONTENT_LENGTH`` (seuil coherent
        avec le collecteur Dev.to).
        """
        page = response.meta.get("playwright_page")
        if page:
            await page.close()

        # Titre : chaine de selecteurs + fallback og:title (dans _extract_title)
        titre = _extract_title(response)

        # Date de publication (ISO 8601 depuis la balise <time>)
        date_str = response.css("time::attr(datetime)").get()

        # Auteur : liens dans le bloc auteur (plusieurs auteurs possibles).
        # Les pages TechCrunch repetent le lien auteur a plusieurs endroits
        # (header, footer, related posts) : on deduplique en preservant l'ordre
        # d'apparition pour eviter "Tim De Chant, Tim De Chant, Tim De Chant, ...".
        auteurs = response.css('a[href*="/author/"]::text').getall()
        auteurs_uniques = list(dict.fromkeys(a.strip() for a in auteurs if a.strip()))
        auteur = ", ".join(auteurs_uniques) or None

        # Contenu : chaine de selecteurs + fallback trafilatura
        contenu_texte = _extract_content(response)

        # HTML brut du bloc article (pour analyse NLP plus riche en aval)
        contenu_html = response.css("div.entry-content").get() or ""

        # Description OpenGraph (resume court, utile en fallback UI)
        og_description = response.css('meta[property="og:description"]::attr(content)').get()

        if not titre:
            self.error_stats["missing_title"] += 1
            logger.warning(f"Article ignore (titre manquant) : url={response.url}")
            return

        if len(contenu_texte) < MIN_CONTENT_LENGTH:
            self.error_stats["empty_content"] += 1
            logger.warning(
                f"Article ignore (contenu {len(contenu_texte)} < {MIN_CONTENT_LENGTH} chars) : "
                f"{titre[:60]}"
            )
            return

        self.collected_articles.append(
            {
                "titre": titre,
                "url": response.url,
                "auteur": auteur,
                "date_publication": date_str,
                "source_nom": "TechCrunch Climate",
                "contenu": contenu_texte,
                "contenu_html": contenu_html,
                "resume": og_description,
            }
        )
        logger.info(f"Scrape OK ({len(contenu_texte)} chars) : {titre[:80]}")

    async def _on_error(self, failure) -> None:
        """Categorise l'erreur Scrapy/Playwright et met a jour la telemetrie.

        Distingue les vraies erreurs Playwright (timeout de navigation,
        crash du browser Chromium, connexion coupee) des echecs reseau
        classiques (Scrapy gere lui-meme les retries 5xx). Cette
        categorisation nourrit le bilan final pour suivre la sante du
        scraper au fil des runs.
        """
        err_repr = str(failure.value)
        url = failure.request.url

        err_lower = err_repr.lower()
        if "timeout" in err_lower:
            self.error_stats["timeout"] += 1
            logger.error(f"TIMEOUT : {url} ({err_repr[:120]})")
        elif "connection closed" in err_lower or "target.createtarget" in err_lower:
            self.error_stats["browser_crash"] += 1
            logger.error(f"BROWSER CRASH : {url} ({err_repr[:120]})")
        else:
            self.error_stats["other"] += 1
            logger.error(f"Echec scraping : {url} ({err_repr[:120]})")


class ScrapingCollector(BaseCollector):
    """Collecteur hybride RSS (decouverte d'URLs) + Scrapy/Playwright (HTML).

    Le flux RSS fournit la liste des articles recents de la section Climate,
    puis Scrapy telecharge le HTML de chaque article via Playwright et en
    extrait le contenu structure. Cette architecture respecte le critere de
    certification C1 : "telechargement de l'HTML d'une ou plusieurs pages
    web visees par une action de scraping".
    """

    def __init__(self) -> None:
        super().__init__(source_name="techcrunch")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        *,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> CollectResult:
        """Orchestre la collecte : RSS -> filtrage -> scraping HTML Scrapy.

        Args:
            keywords: Mots-cles (de ``search_config`` ``type_source='scraping'``)
                pour filtrer les URLs RSS avant scraping. Si vide, aucun
                filtre mots-cles n'est applique (blacklist promos conservee).
            skip_existing: Si True, pre-charge les URLs deja en BDD et
                saute les articles connus avant le fetch Playwright
                (economie enorme : Playwright = 5-10s par page).
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et nombre d'articles.
        """
        result = CollectResult(source_name=self.source_name)

        try:
            urls = await self._discover_urls_via_rss(keywords)
            if not urls:
                logger.warning("Aucune URL decouverte via le flux RSS TechCrunch")
                return result

            # Pre-check BDD : filtre avant le fetch Playwright (5-10s/page).
            # Sur les re-runs, saute la quasi-totalite des URLs deja en BDD
            # et ne lance Playwright que sur les nouveaux articles du RSS.
            if skip_existing:
                known_urls = await load_known_urls(self.source_name)
                # Comparaison normalisee via url_is_known (http/https, trailing slash).
                skipped = [u for u in urls if url_is_known(u, known_urls)]
                urls = [u for u in urls if not url_is_known(u, known_urls)]
                if skipped:
                    logger.info(
                        f"TechCrunch : {len(skipped)} URLs deja en BDD (skip), "
                        f"{len(urls)} nouveaux a scraper via Playwright"
                    )
                if not urls:
                    logger.info("TechCrunch : toutes les URLs RSS sont deja en BDD")
                    return result

            logger.info(f"Lancement du scraping HTML sur {len(urls)} articles...")
            articles, error_stats = await self._scrape_html_pages(urls)

            self._log_scraping_summary(len(urls), articles, error_stats)

            if not articles:
                logger.warning("Aucun article extrait apres scraping HTML")
                return result

            raw_path = generate_raw_path("scraping", "techcrunch")
            payload = {
                "source": "techcrunch_climate_html_scraping",
                "scrape_date": datetime.now().isoformat(),
                "articles_count": len(articles),
                "articles": articles,
                "error_stats": error_stats,
            }
            path = await upload_json_to_minio(
                payload,
                bucket=self.settings.minio_bucket_raw,
                object_name=raw_path,
            )
            result.raw_paths.append(path)
            result.articles_count = len(articles)
            logger.info(f"Scraping termine : {len(articles)} articles -> {path}")

        except Exception as exc:
            error_msg = f"Erreur scraping TechCrunch : {exc}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

        return result

    async def _discover_urls_via_rss(self, keywords: list[str]) -> list[str]:
        """Recupere et filtre les URLs d'articles depuis le flux RSS officiel.

        Le flux RSS TechCrunch renvoie 20 articles par page. Pour atteindre
        un volume comparable aux autres sources du dataset (Guardian, Dev.to),
        on **pagine** le flux via le parametre WordPress ``?paged=N`` jusqu'a
        ``MAX_RSS_PAGES`` pages, soit jusqu'a 200 entrees candidates. La
        pagination s'arrete tot si une page est vide ou ne ramene aucune
        nouvelle URL (dedup global par URL pour absorber les chevauchements
        eventuels entre pages).

        Applique ensuite un filtrage en deux passes sur l'ensemble agrege :

        1. **Blacklist promos** (``PROMO_URL_PATTERNS``) : elimine les
           articles Disrupt / Startup Battlefield / campagnes ticket.
        2. **Match mots-cles** (``keywords``) : conserve les articles dont
           le titre ou summary RSS matche au moins un mot-cle. Si 0 URL
           matche, on fallback sur la liste post-blacklist pour eviter un
           run a 0 article.

        Args:
            keywords: Mots-cles de ``search_config`` pour filtrer les URLs.

        Returns:
            Liste d'URLs retenues (max ``MAX_ARTICLES``), prete pour le
            scraping HTML.
        """
        all_entries = await self._fetch_all_rss_pages()
        logger.info(f"Flux RSS pagine : {len(all_entries)} entrees uniques collectees")

        needles = [k.lower() for k in keywords] if keywords else []

        # Passe 1 : blacklist promos
        post_blacklist: list[tuple[str, str]] = []  # (url, haystack)
        filtered_promo = 0
        for entry in all_entries:
            url = entry.get("link", "").strip()
            if not url:
                continue
            if _is_promo_url(url):
                filtered_promo += 1
                logger.debug(f"URL promo filtree : {url}")
                continue
            haystack = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
            post_blacklist.append((url, haystack))

        # Passe 2 : filtre mots-cles
        filtered_keyword = 0
        matched: list[str] = []
        for url, haystack in post_blacklist:
            if _match_keywords(haystack, needles):
                matched.append(url)
                if len(matched) >= MAX_ARTICLES:
                    break
            else:
                filtered_keyword += 1

        # Fallback si le filtre mots-cles est trop strict : on retombe sur
        # la liste post-blacklist pour garantir qu'on a toujours des URLs
        # a scraper, meme si elles ne matchent pas les mots-cles exacts.
        if needles and not matched:
            logger.warning(
                "Aucune URL RSS ne matche les mots-cles configures. "
                "Fallback : conservation des URLs post-blacklist promos."
            )
            matched = [url for url, _ in post_blacklist[:MAX_ARTICLES]]

        logger.info(
            f"{len(matched)} URLs retenues pour le scraping HTML "
            f"(RSS scanne: {len(all_entries)}, "
            f"promos filtrees: {filtered_promo}, hors-sujet: {filtered_keyword})"
        )
        return matched

    async def _fetch_all_rss_pages(self) -> list[dict[str, Any]]:
        """Itere sur les pages RSS TechCrunch via le parametre ``?paged=N``.

        WordPress (moteur de TechCrunch) expose la pagination RSS via ce
        parametre : page 1 = URL brute, page 2 = ``?paged=2``, etc. Chaque
        page renvoie jusqu'a 20 articles. On s'arrete tot si :

        - une page retourne un statut HTTP d'erreur (4xx/5xx),
        - une page est vide (``<item>`` absents),
        - une page ne ramene que des URLs deja vues (pagination epuisee,
          WordPress peut boucler sur la derniere page).

        Un dedup global par URL est applique : si deux pages contiennent
        le meme article (edge case possible quand un nouvel article est
        publie pendant la pagination), on ne garde qu'une seule entree.

        Returns:
            Liste d'entrees RSS dedupliquees (au sens feedparser : dict-like
            avec ``title``, ``link``, ``summary``, ``published``, ...).
        """
        headers = {
            "User-Agent": self.settings.scraping_user_agent,
            "Accept": "application/rss+xml, application/xml",
        }
        all_entries: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            for page in range(1, MAX_RSS_PAGES + 1):
                page_url = (
                    TECHCRUNCH_CLIMATE_RSS
                    if page == 1
                    else f"{TECHCRUNCH_CLIMATE_RSS}?paged={page}"
                )
                try:
                    response = await client.get(page_url, headers=headers)
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        f"RSS page {page} : HTTP {exc.response.status_code}, arret pagination"
                    )
                    break
                except httpx.HTTPError as exc:
                    logger.warning(f"RSS page {page} : erreur reseau {exc}, arret pagination")
                    break

                feed = feedparser.parse(response.text)
                if not feed.entries:
                    logger.info(f"RSS page {page} : 0 entree, arret pagination")
                    break

                new_count = 0
                for entry in feed.entries:
                    url = entry.get("link", "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    all_entries.append(entry)
                    new_count += 1

                logger.debug(
                    f"RSS page {page} : {len(feed.entries)} entrees, "
                    f"{new_count} nouvelles apres dedup global"
                )

                # Si aucune nouvelle URL n'est ramenee, les pages suivantes
                # seraient identiques -> on evite les appels inutiles.
                if new_count == 0:
                    logger.info(f"RSS page {page} : aucune URL nouvelle, pagination epuisee")
                    break

                # Pause courte entre pages RSS (courtoisie serveur).
                if page < MAX_RSS_PAGES:
                    await asyncio.sleep(RSS_PAGINATION_DELAY)

        return all_entries

    async def _scrape_html_pages(
        self, urls: list[str]
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Lance le spider Scrapy + Playwright sur la liste d'URLs fournie.

        Scrapy est synchrone par nature ; on l'encapsule dans un executor
        pour ne pas bloquer l'event loop FastAPI.

        Returns:
            Tuple (liste d'articles collectes, stats d'erreurs par categorie).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._crawl_sync, urls)

    @staticmethod
    def _crawl_sync(
        urls: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Execute le crawler Scrapy de maniere synchrone.

        Scrapy + Playwright partagent un seul reactor Twisted : on utilise
        un ``CrawlerProcess`` classique et on recupere les articles collectes
        + la telemetrie d'erreurs via le spider apres la fin du process.

        Returns:
            Tuple (articles, error_stats) extraits du spider termine.
        """
        settings = get_project_settings()
        settings.update(
            {
                "DOWNLOAD_HANDLERS": {
                    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                },
                "TWISTED_REACTOR": ("twisted.internet.asyncioreactor.AsyncioSelectorReactor"),
                # --- Configuration Playwright : robustesse ---
                "PLAYWRIGHT_BROWSER_TYPE": "chromium",
                "PLAYWRIGHT_LAUNCH_OPTIONS": {
                    "headless": True,
                    # Flags de stabilite Chromium. `--disable-dev-shm-usage`
                    # et `--no-sandbox` sont essentiels sur Docker et sur
                    # Windows pour eviter les crashes "Connection closed
                    # while reading from the driver" observes en avril 2026.
                    "args": [
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-gpu",
                    ],
                },
                # Timeout de navigation releve (les articles long format /
                # live blogs TechCrunch depassent les 30s par defaut).
                "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": (PLAYWRIGHT_NAVIGATION_TIMEOUT),
                # Recyclage du contexte Playwright pour eviter l'accumulation
                # de state qui fait crasher Chromium apres ~7 pages.
                "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": PLAYWRIGHT_MAX_PAGES_PER_CONTEXT,
                # Bloque les ressources inutiles (cf. `_should_abort_request`)
                # pour economiser bande passante et RAM cote browser.
                "PLAYWRIGHT_ABORT_REQUEST": _should_abort_request,
                # --- Scrapy core ---
                "ROBOTSTXT_OBEY": True,
                "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
                "LOG_LEVEL": "WARNING",
                "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
                # Retries automatiques Scrapy (applique aussi sur les
                # TimeoutError et ConnectionError propages par Playwright).
                "RETRY_ENABLED": True,
                "RETRY_TIMES": 2,
                "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
            }
        )

        process = CrawlerProcess(settings, install_root_handler=False)
        crawler = process.create_crawler(TechCrunchArticleSpider)
        process.crawl(crawler, urls=urls)
        process.start(install_signal_handlers=False)

        if crawler.spider is None:
            return [], {}
        spider = crawler.spider
        articles = list(spider.collected_articles)
        stats = dict(spider.error_stats) if hasattr(spider, "error_stats") else {}
        return articles, stats

    @staticmethod
    def _log_scraping_summary(
        url_count: int,
        articles: list[dict[str, Any]],
        error_stats: dict[str, int],
    ) -> None:
        """Affiche un bilan telemetrique complet du run de scraping.

        Args:
            url_count: Nombre d'URLs soumises au spider.
            articles: Articles effectivement collectes (contenu valide).
            error_stats: Stats d'erreurs par categorie (timeout, crash, ...).
        """
        logger.info("")
        logger.info("-" * 72)
        logger.info("  BILAN SCRAPING TECHCRUNCH")
        logger.info("-" * 72)
        logger.info(f"  URLs soumises       : {url_count}")
        logger.info(f"  Articles collectes  : {len(articles)}")
        if url_count > 0:
            taux = (len(articles) / url_count) * 100
            logger.info(f"  Taux de reussite    : {taux:.1f}%")
        total_err = sum(error_stats.values())
        logger.info(f"  Echecs total        : {total_err}")
        if total_err > 0:
            for categorie, nb in error_stats.items():
                if nb > 0:
                    logger.info(f"    - {categorie:<16}: {nb}")
        if articles:
            lens = [len(a.get("contenu", "")) for a in articles]
            logger.info(
                f"  Longueur contenu    : moyenne={sum(lens) // len(lens)} chars, "
                f"min={min(lens)}, max={max(lens)}"
            )
        logger.info("-" * 72)


async def run_scraping_collection() -> CollectResult:
    """Point d'entree principal pour le scraping hybride RSS + HTML.

    Charge les mots-cles de ``search_config`` (``type_source='scraping'``)
    pour filtrer les URLs RSS selon les thematiques Green IT configurees.
    Si aucun mot-cle n'est configure, le scraping fonctionne sans filtrage
    thematique (mais la blacklist anti-promo TechCrunch reste active).

    Returns:
        Resultat de la collecte avec nombre d'articles et chemins MinIO.
    """
    logger.info("=== Demarrage scraping hybride RSS + HTML (TechCrunch Climate) ===")

    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="scraping")
        keywords = [cfg.mot_cle for cfg in configs]

    if keywords:
        logger.info(f"Filtrage RSS par {len(keywords)} mots-cles : {', '.join(keywords)}")
    else:
        logger.info(
            "Aucun mot-cle 'scraping' en DB : le scraping n'applique que "
            "la blacklist anti-promo TechCrunch."
        )

    collector = ScrapingCollector()
    return await collector.collect(keywords=keywords)


def _strip_html(text: str) -> str:
    """Nettoie basiquement un fragment HTML (suppression des balises).

    Utilitaire conserve pour compatibilite avec d'eventuels pipelines aval
    qui manipulent le champ ``contenu_html`` brut (non utilise en interne
    depuis la bascule trafilatura, mais expose pour flexibilite).

    Args:
        text: Fragment HTML.

    Returns:
        Texte depouille des balises et des espaces multiples.
    """
    clean = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", clean).strip()


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    logging.getLogger("asyncio").addFilter(_IgnoreTaskDestroyedFilter())
    asyncio.run(run_scraping_collection())
