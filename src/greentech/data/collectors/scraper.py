"""Module 2 : Scraping hybride RSS + Scrapy/Playwright sur TechCrunch Climate.

Architecture en deux temps pour allier robustesse et conformite au referentiel :

1. **Decouverte d'URLs** via le flux RSS officiel
   (`https://techcrunch.com/category/climate/feed/`). Le RSS est standardise
   et stable, il resiste aux refontes de la page d'index HTML.

2. **Scraping HTML** de chaque article individuel avec Scrapy + Playwright.
   Le navigateur headless charge la page d'article, attend le rendu React,
   puis Scrapy parse le DOM via des selecteurs CSS. Cette etape coche les
   criteres de certification C1 (telechargement HTML + parsing HTML).

Pourquoi le double niveau plutot que RSS seul ? Le flux RSS ne fournit qu'un
resume (summary) partiel, pas le contenu complet. Le scraping HTML recupere
l'integralite du texte de l'article, ce qui ameliore la qualite du modele
de classification et du resume genere par Qwen.

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
from greentech.data.collectors.base import BaseCollector, CollectResult
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# URL du flux RSS pour la decouverte des articles
TECHCRUNCH_CLIMATE_RSS = "https://techcrunch.com/category/climate/feed/"

# Nombre maximum d'articles a scraper par session (respect ethique du site)
MAX_ARTICLES = 20

# Delai entre requetes HTML (scraping ethique)
DOWNLOAD_DELAY = 2.0

# Timeout HTTP (RSS + pages articles)
REQUEST_TIMEOUT = 20.0

# Ressources dont le telechargement est bloque par Playwright : elles n'apportent
# rien au texte extrait mais generent des tasks asyncio qui peuvent rester en
# attente lors de l'arret du reactor.
_ABORTED_RESOURCE_TYPES = frozenset({"image", "media", "font", "stylesheet"})


def _should_abort_request(request: PlaywrightRequest) -> bool:
    """Filtre Playwright : bloque les ressources non essentielles au scraping texte."""
    return request.resource_type in _ABORTED_RESOURCE_TYPES


class _IgnoreTaskDestroyedFilter(logging.Filter):
    """Supprime le message asyncio "Task was destroyed but it is pending".

    Ce warning est inherent a la combinaison Twisted + scrapy-playwright sur
    Windows : des tasks asyncio internes a Playwright peuvent rester en
    attente au moment ou Twisted arrete le reactor. Ce bruit n'affecte ni
    les articles collectes ni l'integrite des donnees, on le masque pour
    clarifier la sortie du pipeline.
    """

    _NOISE = "Task was destroyed but it is pending"

    def filter(self, record: logging.LogRecord) -> bool:
        return self._NOISE not in record.getMessage()


class TechCrunchArticleSpider(Spider):
    """Spider Scrapy qui scrape le HTML de pages d'articles individuelles.

    Recoit la liste d'URLs via le constructeur (fournie par le flux RSS),
    charge chaque page avec Playwright pour gerer le rendu dynamique
    (React hydratation), puis extrait les champs metier via des selecteurs
    CSS sur le DOM rendu.
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
    }

    def __init__(self, urls: list[str] | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_urls_list: list[str] = urls or []
        self._scraping_user_agent: str = get_settings().scraping_user_agent
        self.collected_articles: list[dict[str, Any]] = []

    async def start(self):
        """Genere une requete Playwright par URL d'article a scraper.

        Remplace `start_requests()` depuis Scrapy 2.13 pour autoriser la
        generation asynchrone de requetes (coroutine native).
        """
        for url in self.start_urls_list:
            yield Request(
                url,
                callback=self.parse_article,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "h1", timeout=15000),
                    ],
                },
                headers={"User-Agent": self._scraping_user_agent},
                errback=self._on_error,
            )

    async def parse_article(self, response: Response) -> None:
        """Parse le HTML d'une page d'article TechCrunch.

        Extrait le titre, la date, l'auteur, le contenu et les metadonnees
        OpenGraph a partir du DOM rendu par Playwright.
        """
        page = response.meta.get("playwright_page")
        if page:
            await page.close()

        # Titre : h1 de la page, fallback sur og:title si absent
        titre = response.css("h1.article-hero__title::text").get()
        if not titre:
            titre = response.css('meta[property="og:title"]::attr(content)').get() or ""
        titre = titre.strip()

        # Date de publication (ISO 8601 depuis la balise <time>)
        date_str = response.css("time::attr(datetime)").get()

        # Auteur : liens dans le bloc auteur (plusieurs possibles)
        auteurs = response.css('a[href*="/author/"]::text').getall()
        auteur = ", ".join(a.strip() for a in auteurs if a.strip()) or None

        # Contenu : concatener les paragraphes de la zone entry-content
        paragraphes = response.css("div.entry-content p::text").getall()
        contenu_texte = "\n\n".join(p.strip() for p in paragraphes if p.strip())

        # HTML brut du bloc article (pour analyse NLP plus riche si besoin)
        contenu_html = response.css("div.entry-content").get() or ""

        # Description OpenGraph (resume court, utile en fallback)
        og_description = response.css('meta[property="og:description"]::attr(content)').get()

        if not titre or not contenu_texte:
            logger.warning(
                f"Article ignore (champs manquants) : titre={bool(titre)}, "
                f"contenu={bool(contenu_texte)}, url={response.url}"
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
        logger.info(f"Scrape OK : {titre[:80]}...")

    async def _on_error(self, failure) -> None:
        """Journalise les erreurs sans interrompre la collecte des autres URLs."""
        logger.error(f"Echec scraping : {failure.request.url} ({failure.value})")


class ScrapingCollector(BaseCollector):
    """Collecteur hybride RSS (decouverte d'URLs) + Scrapy/Playwright (HTML).

    Le flux RSS fournit la liste des articles recents, puis Scrapy telecharge
    le HTML de chaque article via Playwright et en extrait le contenu structure.
    Cette architecture respecte le critere de certification C1 : "telechargement
    de l'HTML d'une ou plusieurs pages web visees par une action de scraping".
    """

    def __init__(self) -> None:
        super().__init__(source_name="techcrunch")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        **kwargs: Any,
    ) -> CollectResult:
        """Orchestre la collecte : RSS -> liste d'URLs -> scraping HTML Scrapy.

        Args:
            keywords: Mots-cles pour filtrer les URLs avant scraping (optionnel).
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et nombre d'articles scrapes.
        """
        result = CollectResult(source_name=self.source_name)

        try:
            urls = await self._discover_urls_via_rss(keywords)
            if not urls:
                logger.warning("Aucune URL decouverte via le flux RSS TechCrunch")
                return result

            logger.info(f"Lancement du scraping HTML sur {len(urls)} articles...")
            articles = await self._scrape_html_pages(urls)

            if not articles:
                logger.warning("Aucun article extrait apres scraping HTML")
                return result

            raw_path = generate_raw_path("scraping", "techcrunch")
            payload = {
                "source": "techcrunch_climate_html_scraping",
                "scrape_date": datetime.now().isoformat(),
                "articles_count": len(articles),
                "articles": articles,
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
        """Recupere les URLs d'articles recents depuis le flux RSS officiel.

        Le flux RSS donne un listing stable et rapide, sans dependre de la
        structure HTML de la page d'index (qui peut changer). On filtre
        optionnellement par mots-cles dans le titre ou le resume.
        """
        headers = {
            "User-Agent": self.settings.scraping_user_agent,
            "Accept": "application/rss+xml, application/xml",
        }
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(TECHCRUNCH_CLIMATE_RSS, headers=headers)
            response.raise_for_status()

        feed = feedparser.parse(response.text)
        logger.info(f"Flux RSS : {len(feed.entries)} entrees decouvertes")

        urls: list[str] = []
        needles = [k.lower() for k in keywords] if keywords else []
        for entry in feed.entries[:MAX_ARTICLES]:
            url = entry.get("link", "").strip()
            if not url:
                continue
            if needles:
                haystack = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                if not any(n in haystack for n in needles):
                    continue
            urls.append(url)

        logger.info(f"{len(urls)} URLs retenues pour le scraping HTML")
        return urls

    async def _scrape_html_pages(self, urls: list[str]) -> list[dict[str, Any]]:
        """Lance le spider Scrapy + Playwright sur la liste d'URLs fournie.

        Scrapy est synchrone par nature ; on l'encapsule dans un executor pour
        ne pas bloquer l'event loop FastAPI.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._crawl_sync, urls)

    @staticmethod
    def _crawl_sync(urls: list[str]) -> list[dict[str, Any]]:
        """Execute le crawler Scrapy de maniere synchrone.

        Scrapy + Playwright partagent un seul reactor Twisted : on utilise un
        `CrawlerProcess` classique et on recupere les articles collectes via
        l'attribut `collected_articles` du spider apres la fin du process.
        """
        settings = get_project_settings()
        settings.update(
            {
                "DOWNLOAD_HANDLERS": {
                    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                },
                "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
                "PLAYWRIGHT_BROWSER_TYPE": "chromium",
                "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True},
                "PLAYWRIGHT_ABORT_REQUEST": _should_abort_request,
                "ROBOTSTXT_OBEY": True,
                "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
                "LOG_LEVEL": "WARNING",
                "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
            }
        )

        process = CrawlerProcess(settings, install_root_handler=False)
        crawler = process.create_crawler(TechCrunchArticleSpider)
        process.crawl(crawler, urls=urls)
        process.start(install_signal_handlers=False)

        return list(crawler.spider.collected_articles) if crawler.spider else []


async def run_scraping_collection() -> CollectResult:
    """Point d'entree principal pour le scraping hybride RSS + HTML.

    Returns:
        Resultat de la collecte avec nombre d'articles scrapes et chemins MinIO.
    """
    logger.info("=== Demarrage scraping hybride RSS + HTML (TechCrunch Climate) ===")
    collector = ScrapingCollector()
    return await collector.collect(keywords=[])


def _strip_html(text: str) -> str:
    """Nettoie basiquement un fragment HTML (suppression des balises)."""
    clean = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", clean).strip()


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    logging.getLogger("asyncio").addFilter(_IgnoreTaskDestroyedFilter())
    asyncio.run(run_scraping_collection())
