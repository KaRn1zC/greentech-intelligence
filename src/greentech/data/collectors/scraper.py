"""Module 2 : Scraping hybride via Scrapy + Playwright.

Cible le blog TechCrunch (section Climate) qui utilise un chargement
dynamique JavaScript (infinite scroll). Playwright est nécessaire
pour simuler le rendu navigateur et récupérer le contenu.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger
from scrapy import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from scrapy.utils.project import get_project_settings

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# URL cible pour le scraping
TECHCRUNCH_CLIMATE_URL = "https://techcrunch.com/category/climate/"

# Nombre maximum de pages/articles à scraper par session
MAX_ARTICLES = 30

# Délai minimum entre requêtes (respect du site cible)
DOWNLOAD_DELAY = 2.0


class TechCrunchSpider(Spider):
    """Spider Scrapy pour TechCrunch Climate avec rendu Playwright.

    Navigue sur la section Climate de TechCrunch, charge le contenu
    dynamique via Playwright et extrait les métadonnées des articles.
    """

    name = "techcrunch_climate"
    allowed_domains = ["techcrunch.com"]
    start_urls = [TECHCRUNCH_CLIMATE_URL]

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        settings = get_settings()
        self.user_agent = settings.scraping_user_agent
        self.collected_articles: list[dict[str, Any]] = []

    def start_requests(self):
        """Génère les requêtes initiales avec le rendu Playwright activé."""
        for url in self.start_urls:
            yield self._make_request(url)

    def _make_request(self, url: str):
        """Crée une requête Scrapy avec les métadonnées Playwright.

        Args:
            url: URL de la page à charger.
        """
        from scrapy import Request

        return Request(
            url,
            callback=self.parse,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_methods": [
                    # Attendre que le contenu principal soit chargé
                    {
                        "method": "wait_for_selector",
                        "args": ["article"],
                        "kwargs": {"timeout": 15000},
                    },
                    # Scroll pour déclencher le chargement dynamique
                    {
                        "method": "evaluate",
                        "args": ["window.scrollTo(0, document.body.scrollHeight)"],
                    },
                    {"method": "wait_for_timeout", "args": [3000]},
                ],
            },
            headers={"User-Agent": self.user_agent},
        )

    async def parse(self, response: Response) -> None:
        """Parse la page de listing des articles TechCrunch.

        Extrait les liens, titres, dates et auteurs depuis le HTML
        rendu par Playwright.

        Args:
            response: Réponse HTTP avec le HTML rendu.
        """
        page = response.meta.get("playwright_page")
        if page:
            await page.close()

        articles = response.css("article")
        logger.info(f"TechCrunch : {len(articles)} articles trouvés sur la page")

        for article in articles[:MAX_ARTICLES]:
            titre = article.css("h2 a::text, h3 a::text").get()
            url = article.css("h2 a::attr(href), h3 a::attr(href)").get()

            if not titre or not url:
                continue

            auteur = article.css(".post-block__author a::text, [class*='author'] a::text").get()
            date_str = article.css("time::attr(datetime)").get()

            self.collected_articles.append(
                {
                    "titre": titre.strip(),
                    "url": response.urljoin(url),
                    "auteur": auteur.strip() if auteur else None,
                    "date_publication": date_str,
                    "source_nom": "TechCrunch Climate",
                    "contenu_html": article.get(),
                }
            )

        logger.info(f"TechCrunch : {len(self.collected_articles)} articles extraits")


class ScrapingCollector(BaseCollector):
    """Collecteur de données via scraping hybride (Scrapy + Playwright).

    Lance un spider Scrapy avec Playwright pour charger les pages dynamiques
    et sauvegarde le HTML brut dans MinIO raw-data.
    """

    def __init__(self) -> None:
        super().__init__(source_name="techcrunch")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        **kwargs: Any,
    ) -> CollectResult:
        """Lance le scraping de TechCrunch Climate.

        Note : Les mots-clés sont utilisés pour le filtrage a posteriori,
        car TechCrunch Climate est une catégorie thématique fixe.

        Args:
            keywords: Mots-clés pour le filtrage (utilisés en post-traitement).
            **kwargs: Paramètres additionnels.

        Returns:
            Résultat de la collecte avec chemins MinIO.
        """
        result = CollectResult(source_name=self.source_name)

        try:
            articles = await self._run_spider()

            if not articles:
                logger.warning("Aucun article scrapé depuis TechCrunch")
                return result

            # Sauvegarde brute dans MinIO
            raw_path = generate_raw_path("scraping", "techcrunch")
            payload = {
                "source": "techcrunch_climate",
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

            logger.info(f"Scraping terminé : {len(articles)} articles -> {path}")

        except Exception as e:
            error_msg = f"Erreur scraping TechCrunch : {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    async def _run_spider(self) -> list[dict[str, Any]]:
        """Exécute le spider Scrapy dans un processus séparé.

        Returns:
            Liste des articles extraits.
        """
        spider = TechCrunchSpider

        # Exécuter Scrapy dans un thread séparé pour ne pas bloquer l'event loop
        loop = asyncio.get_event_loop()
        articles = await loop.run_in_executor(None, self._crawl_sync, spider)
        return articles

    @staticmethod
    def _crawl_sync(spider_class: type[Spider]) -> list[dict[str, Any]]:
        """Lance le crawler Scrapy de manière synchrone.

        Args:
            spider_class: Classe du spider à exécuter.

        Returns:
            Articles collectés par le spider.
        """
        collected: list[dict[str, Any]] = []

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
                "ROBOTSTXT_OBEY": True,
                "DOWNLOAD_DELAY": DOWNLOAD_DELAY,
                "LOG_LEVEL": "WARNING",
                "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
            }
        )

        process = CrawlerProcess(settings)
        crawler = process.create_crawler(spider_class)
        process.crawl(crawler)
        process.start()

        # Récupérer les articles collectés depuis l'instance du spider
        if crawler.spider:
            collected = crawler.spider.collected_articles

        return collected


async def run_scraping_collection() -> CollectResult:
    """Point d'entrée principal pour le scraping.

    Returns:
        Résultat de la collecte.
    """
    logger.info("=== Démarrage scraping (TechCrunch Climate) ===")

    collector = ScrapingCollector()
    return await collector.collect(keywords=[])


if __name__ == "__main__":
    asyncio.run(run_scraping_collection())
