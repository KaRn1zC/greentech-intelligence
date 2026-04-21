"""Orchestrator pour le scraping statique des 4 sites Green IT (B2.3).

Complementaire au scraper TechCrunch (``scraper.py``) qui utilise
Playwright : cet orchestrator se concentre sur les 4 sites identifies
en B2.1 comme etant en HTML statique (pas de JS critique). Il lance
les 4 spiders en un unique ``CrawlerProcess`` Scrapy pour :

- Minimiser l'overhead de demarrage Twisted (un seul reactor instancie)
- Partager les retries, le rate limit et la config ROBOTSTXT
- Aggreger les telemetries par spider dans un bilan global
- Garder une architecture coherente avec ``ScrapingCollector`` (TechCrunch)

Sites scrapes
-------------

Par ordre d'ajout au dataset (volume attendu / langue / specificite) :

1. ``GreenIT.fr`` (1 001 posts, FR) : blog francophone reference Green IT,
   contenu 100% Green IT positif.
2. ``Greensoftware.foundation`` (170 articles, EN) : fondation tech
   dediee au green software engineering.
3. ``Sustainablewebdesign.org`` (131 items, EN) : WSDG (posts + guidelines).
4. ``Climateaction.tech`` (71 posts, EN) : communaute tech pour le climat.

Volume total attendu : ~1 373 articles apres scraping, dont 95%+ sont
des candidats Green IT positifs (contenu ciblee).

Usage
-----

    uv run python -m greentech.data.collectors.static_scraping_collector

"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult
from greentech.data.collectors.spiders.base import StaticArticleSpider
from greentech.data.collectors.spiders.climate_action_tech_spider import (
    ClimateActionTechSpider,
)
from greentech.data.collectors.spiders.greenit_fr_spider import GreenItFrSpider
from greentech.data.collectors.spiders.greensoftware_spider import GreenSoftwareSpider
from greentech.data.collectors.spiders.sustainable_web_spider import (
    SustainableWebSpider,
)
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# Spiders par defaut lances ensemble. L'ordre n'a pas d'importance
# fonctionnelle (Scrapy ordonnance les requetes en interne) mais on
# liste du plus gros volume au plus petit pour que les logs soient
# lisibles (GreenIT.fr domine largement en nombre d'articles).
DEFAULT_SPIDERS: list[type[StaticArticleSpider]] = [
    GreenItFrSpider,
    GreenSoftwareSpider,
    SustainableWebSpider,
    ClimateActionTechSpider,
]

# Delai entre requetes, global. Les spiders peuvent surcharger via
# leur propre ``download_delay``, mais on fixe un plancher a 2s pour
# garantir une bonne citoyennete de scraping (conforme au referentiel
# C1 : "delais entre requetes pour un scraping ethique").
GLOBAL_DOWNLOAD_DELAY = 2.0


class StaticScrapingCollector(BaseCollector):
    """Orchestrator Scrapy pour les 4 sites Green IT statiques.

    Lance les spiders en un seul ``CrawlerProcess`` pour partager
    le reactor Twisted. Les resultats sont ensuite ventiles par
    site (chaque spider a son bucket MinIO dedie pour tracabilite).
    """

    def __init__(self) -> None:
        super().__init__(source_name="static_scraping")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str] | None = None,
        *,
        spider_classes: list[type[StaticArticleSpider]] | None = None,
        **kwargs: Any,
    ) -> CollectResult:
        """Orchestre le scraping des 4 sites Green IT.

        Args:
            keywords: Non utilise (les spiders scrapent l'integralite
                des sites sans filtrage par mots-cles, puisque le
                contenu est deja 100% Green IT). Conserve pour
                compatibilite avec l'interface BaseCollector.
            spider_classes: Liste optionnelle de spiders a lancer. Par
                defaut lance les 4 sites (``DEFAULT_SPIDERS``). Utile
                pour tester un site isole ou ajouter de nouveaux spiders
                sans modifier cette methode.
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            CollectResult avec le total d'articles collectes et la liste
            des chemins MinIO (un par site).
        """
        result = CollectResult(source_name=self.source_name)
        classes = spider_classes or DEFAULT_SPIDERS

        try:
            # Scrapy + Twisted sont synchrones par nature : on encapsule
            # dans un executor pour ne pas bloquer l'event loop FastAPI /
            # asyncio environnant.
            articles_by_spider, error_stats_by_spider = await self._run_spiders(classes)

            for spider_name, articles in articles_by_spider.items():
                if not articles:
                    logger.warning(f"Aucun article collecte pour spider '{spider_name}'")
                    continue

                raw_path = generate_raw_path("scraping", spider_name)
                payload = {
                    "source": spider_name,
                    "scrape_date": datetime.now().isoformat(),
                    "articles_count": len(articles),
                    "articles": articles,
                    "error_stats": error_stats_by_spider.get(spider_name, {}),
                }
                path = await upload_json_to_minio(
                    payload,
                    bucket=self.settings.minio_bucket_raw,
                    object_name=raw_path,
                )
                result.raw_paths.append(path)
                result.articles_count += len(articles)

                logger.info(f"[{spider_name}] {len(articles)} articles sauvegardes -> {path}")

            self._log_global_summary(articles_by_spider, error_stats_by_spider)

        except Exception as exc:
            error_msg = f"Erreur scraping statique multi-sites : {exc}"
            logger.exception(error_msg)
            result.errors.append(error_msg)

        return result

    async def _run_spiders(
        self,
        spider_classes: list[type[StaticArticleSpider]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
        """Execute les spiders dans un executor (Twisted = synchrone).

        Args:
            spider_classes: Classes des spiders a instancier.

        Returns:
            Tuple (articles_by_spider, error_stats_by_spider) ou chaque
            dict est indexe par nom du spider.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._crawl_sync, spider_classes)

    @staticmethod
    def _crawl_sync(
        spider_classes: list[type[StaticArticleSpider]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
        """Lance les spiders en un seul ``CrawlerProcess`` Scrapy.

        Un unique process partage les ressources (reactor Twisted, DNS
        resolver, robots.txt cache, ...). Les spiders tournent en
        parallele grace a l'ordonnancement interne Scrapy, avec le
        ``DOWNLOAD_DELAY`` applique par spider individuellement.

        Args:
            spider_classes: Classes des spiders a instancier.

        Returns:
            Tuple (articles_by_spider, error_stats_by_spider).
        """
        user_agent = get_settings().scraping_user_agent

        settings = get_project_settings()
        settings.update(
            {
                # --- Scrapy core ---
                "ROBOTSTXT_OBEY": True,
                "DOWNLOAD_DELAY": GLOBAL_DOWNLOAD_DELAY,
                "USER_AGENT": user_agent,
                "LOG_LEVEL": "WARNING",
                "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
                # Retries automatiques sur erreurs reseau temporaires.
                "RETRY_ENABLED": True,
                "RETRY_TIMES": 2,
                "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
                # Concurrency reduite pour rester courtois sur des sites
                # de petite taille (pas des CDN enterprise).
                "CONCURRENT_REQUESTS": 4,
                "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
            }
        )

        process = CrawlerProcess(settings, install_root_handler=False)
        crawlers = []

        for spider_cls in spider_classes:
            crawler = process.create_crawler(spider_cls)
            process.crawl(crawler)
            crawlers.append((spider_cls.name, crawler))

        # Bloque jusqu'a la fin de tous les spiders. Le reactor Twisted
        # s'arrete automatiquement quand le dernier spider termine.
        process.start(install_signal_handlers=False)

        articles_by_spider: dict[str, list[dict[str, Any]]] = {}
        error_stats_by_spider: dict[str, dict[str, int]] = {}

        for spider_name, crawler in crawlers:
            if crawler.spider is None:
                logger.warning(f"Spider '{spider_name}' n'a pas demarre")
                articles_by_spider[spider_name] = []
                error_stats_by_spider[spider_name] = {}
                continue

            spider = crawler.spider
            articles_by_spider[spider_name] = list(
                getattr(spider, "collected_articles", []),
            )
            error_stats_by_spider[spider_name] = dict(
                getattr(spider, "error_stats", {}),
            )

        return articles_by_spider, error_stats_by_spider

    @staticmethod
    def _log_global_summary(
        articles_by_spider: dict[str, list[dict[str, Any]]],
        error_stats_by_spider: dict[str, dict[str, int]],
    ) -> None:
        """Bilan global du run : articles par site + erreurs par categorie."""
        total_articles = sum(len(articles) for articles in articles_by_spider.values())
        logger.info("")
        logger.info("=" * 72)
        logger.info("  BILAN GLOBAL SCRAPING STATIQUE (4 sites Green IT)")
        logger.info("=" * 72)
        logger.info(f"  Articles collectes  : {total_articles}")
        logger.info("")

        for spider_name, articles in articles_by_spider.items():
            errors = error_stats_by_spider.get(spider_name, {})
            total_err = sum(errors.values())
            logger.info(f"  [{spider_name:<22}] {len(articles):>4} articles, {total_err:>3} echecs")
            if total_err > 0:
                for categorie, nb in errors.items():
                    if nb > 0:
                        logger.info(f"      - {categorie:<18}: {nb}")

        logger.info("=" * 72)


async def run_static_scraping() -> CollectResult:
    """Point d'entree principal pour le scraping statique multi-sites.

    Lance les 4 spiders Green IT (greenit.fr, greensoftware.foundation,
    sustainablewebdesign.org, climateaction.tech) et sauvegarde les
    articles dans MinIO par site.

    Returns:
        CollectResult avec le total d'articles et la liste des chemins MinIO.
    """
    logger.info("=== Demarrage scraping statique (4 sites Green IT) ===")
    collector = StaticScrapingCollector()
    return await collector.collect()


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_static_scraping())
