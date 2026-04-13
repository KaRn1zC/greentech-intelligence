"""Module 1 : Collecte de données via l'API NewsData.io.

Utilise httpx (client HTTP asynchrone) pour interroger l'API REST
et sauvegarde les réponses brutes dans le bucket MinIO raw-data.

"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

from .base import get_config_from_db

# URL de base de l'API NewsData.io
NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"

# Nombre maximum de résultats par requête (limite API gratuite)
MAX_RESULTS_PER_REQUEST = 10

# Timeout pour les requêtes HTTP (secondes)
HTTP_TIMEOUT = 30.0

# Rate limiting : NewsData.io plan gratuit = 30 requetes / 15 minutes.
# Un delai de 2s entre requetes respecte confortablement cette contrainte.
MIN_DELAY_BETWEEN_REQUESTS = 2.0

# Retry avec backoff exponentiel sur erreur 429 (Too Many Requests)
MAX_RETRIES_ON_RATE_LIMIT = 3
INITIAL_BACKOFF_SECONDS = 5.0
BACKOFF_MULTIPLIER = 2.0


class RateLimitExceededError(Exception):
    """Leve quand l'API refuse les requetes apres plusieurs retries avec backoff.

    Signale que le budget de requetes est definitivement epuise pour la fenetre
    en cours (ex. 30 req/15min sur NewsData.io gratuit). Le collecteur interrompt
    alors la boucle plutot que de continuer a hammerer l'API inutilement.
    """


class ApiCollector(BaseCollector):
    """Collecteur de données via l'API NewsData.io.

    Interroge l'API REST avec les mots-clés configurés dans search_config,
    filtre les résultats et les sauvegarde dans MinIO raw-data au format JSON.
    """

    def __init__(self) -> None:
        super().__init__(source_name="newsdata")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        **kwargs: Any,
    ) -> CollectResult:
        """Lance la collecte d'articles via l'API NewsData.io.

        Effectue une requête par mot-clé, filtre les résultats
        et stocke chaque lot dans MinIO.

        Args:
            keywords: Liste de mots-clés à rechercher.
            **kwargs: Paramètres additionnels (non utilisés).

        Returns:
            Résultat de la collecte avec chemins MinIO et compteurs.
        """
        result = CollectResult(source_name=self.source_name)

        if not self.settings.api_news_key or self.settings.api_news_key == "YOUR_NEWS_API_KEY_HERE":
            msg = "Clé API NewsData.io non configurée (API_NEWS_KEY dans .env)"
            logger.warning(msg)
            result.errors.append(msg)
            return result

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            for idx, keyword in enumerate(keywords):
                # Rate limit client-side pour respecter la limite NewsData.io
                # (30 req/15min = 1 req toutes les 30s max, on prend 2s de marge)
                if idx > 0:
                    await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

                try:
                    articles = await self._fetch_articles_with_retry(client, keyword)
                    if not articles:
                        logger.info(f"Aucun article trouvé pour '{keyword}'")
                        continue

                    # Sauvegarde brute dans MinIO
                    raw_path = generate_raw_path("api", "newsdata")
                    payload = {
                        "keyword": keyword,
                        "source": "newsdata.io",
                        "articles_count": len(articles),
                        "articles": articles,
                    }
                    path = await upload_json_to_minio(
                        payload,
                        bucket=self.settings.minio_bucket_raw,
                        object_name=raw_path,
                    )
                    result.raw_paths.append(path)
                    result.articles_count += len(articles)

                    logger.info(f"'{keyword}' : {len(articles)} articles collectés -> {path}")

                except RateLimitExceededError as e:
                    # Le budget NewsData est epuise : inutile de continuer les autres mots-cles
                    error_msg = (
                        f"Rate limit NewsData.io definitivement atteint pour '{keyword}' : {e}"
                    )
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    logger.warning(
                        "Arret de la collecte API : reessayez plus tard "
                        "(limite gratuite NewsData.io = 30 requetes / 15 min)"
                    )
                    break
                except Exception as e:
                    error_msg = f"Erreur collecte '{keyword}' : {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        logger.info(
            f"Collecte API terminée : {result.articles_count} articles, "
            f"{len(result.errors)} erreurs"
        )
        return result

    async def _fetch_articles_with_retry(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Appelle `_fetch_articles` avec retry exponentiel sur erreur 429.

        NewsData.io renvoie 429 quand la limite de requetes est atteinte. Dans
        ce cas, on patiente (backoff exponentiel) et on reessaie. Si plusieurs
        retries consecutifs echouent, on leve `RateLimitExceededError` pour
        interrompre proprement la collecte (continuer ne fera qu'aggraver).

        Args:
            client: Client HTTP asynchrone partage entre requetes.
            keyword: Mot-cle a rechercher.

        Returns:
            Liste des articles retournes par l'API.

        Raises:
            RateLimitExceededError: Si on atteint le nombre max de retries.
            httpx.HTTPStatusError: Sur toute autre erreur HTTP non-429.
        """
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES_ON_RATE_LIMIT + 1):
            try:
                return await self._fetch_articles(client, keyword)
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 429:
                    raise
                # NewsData.io peut fournir un header Retry-After indiquant la duree d'attente
                retry_after = e.response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else backoff
                logger.warning(
                    f"429 Too Many Requests sur '{keyword}' "
                    f"(tentative {attempt}/{MAX_RETRIES_ON_RATE_LIMIT}), "
                    f"attente {wait_seconds:.1f}s avant retry"
                )
                await asyncio.sleep(wait_seconds)
                backoff *= BACKOFF_MULTIPLIER

        msg = f"Rate limit persistant apres {MAX_RETRIES_ON_RATE_LIMIT} tentatives"
        raise RateLimitExceededError(msg)

    async def _fetch_articles(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Interroge l'API NewsData.io pour un mot-clé donné.

        Args:
            client: Client HTTP asynchrone.
            keyword: Mot-clé de recherche.

        Returns:
            Liste des articles bruts (dictionnaires JSON).

        Raises:
            httpx.HTTPStatusError: Si la réponse HTTP est une erreur 4xx/5xx.
        """
        params = {
            "apikey": self.settings.api_news_key,
            "q": keyword,
            "category": "technology",
            "language": "en",
            "size": MAX_RESULTS_PER_REQUEST,
        }

        logger.debug(f"Requête API : {keyword}")
        response = await client.get(NEWSDATA_BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if data.get("status") != "success":
            logger.warning(f"API status non-success pour '{keyword}' : {data.get('status')}")
            return []

        articles = data.get("results", [])
        return self._parse_articles(articles)

    def _parse_articles(self, raw_articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filtre et normalise les articles bruts de l'API.

        Extrait les champs pertinents et élimine les entrées incomplètes
        (sans titre ou sans lien).

        Args:
            raw_articles: Articles bruts depuis la réponse JSON de l'API.

        Returns:
            Articles filtrés avec les champs normalisés.
        """
        parsed = []
        for article in raw_articles:
            titre = article.get("title")
            url = article.get("link")

            if not titre or not url:
                logger.debug(
                    f"Article ignoré (titre ou URL manquant) : {article.get('title', 'N/A')}"
                )
                continue

            parsed.append(
                {
                    "titre": titre,
                    "url": url,
                    "description": article.get("description", ""),
                    "contenu": article.get("content", ""),
                    "date_publication": article.get("pubDate"),
                    "auteur": ", ".join(article.get("creator") or []) or None,
                    "source_nom": article.get("source_id", "newsdata"),
                    "langue": article.get("language", "en"),
                    "categorie": article.get("category", []),
                }
            )

        return parsed


async def run_api_collection() -> CollectResult:
    """Point d'entrée principal pour la collecte API.

    Charge les mots-clés depuis la base de données (search_config)
    et lance la collecte via l'API NewsData.io.

    Returns:
        Résultat de la collecte.
    """
    logger.info("=== Démarrage collecte API (NewsData.io) ===")

    # Charger les mots-clés depuis la configuration en base
    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="api")
        keywords = [cfg.mot_cle for cfg in configs]

    if not keywords:
        logger.warning("Aucun mot-clé configuré pour la collecte API")
        return CollectResult(source_name="newsdata")

    collector = ApiCollector()
    return await collector.collect(keywords)


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_api_collection())
