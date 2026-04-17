"""Collecteur REST/JSON : The Guardian Open Platform.

Remplace NewsData.io (desactive en avril 2026 a cause de la troncature
systematique du contenu en free tier). Le tier Developer du Guardian offre :

- 5000 requetes/jour, 12 requetes/seconde
- ``bodyText`` complet garanti (plusieurs milliers de caracteres)
- Sections natives adaptees au Green IT : ``environment``, ``technology``,
  ``sustainable-business``, ``green-living``
- Usage non-commercial, ce qui correspond au positionnement du projet
  (diplome / certification AI Engineering)

Architecture generale
---------------------

Le collecteur suit le meme pattern que les autres collecteurs du projet :

1. Charge les mots-cles depuis ``search_config`` (type_source = ``"guardian"``).
2. Interroge l'endpoint ``/search`` avec ``show-fields=bodyText,...`` pour
   recuperer l'integralite du corps de chaque article en une seule passe
   (pas de pattern liste + detail comme pour Dev.to).
3. Sauvegarde les reponses brutes dans MinIO (bucket raw-data).
4. Le nettoyage et l'ingestion SQL sont geres en aval par
   ``spark_cleaner`` et ``sql_ingester`` comme pour les autres sources.

Usage
-----

    uv run python -m greentech.data.collectors.guardian_collector

"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult, get_config_from_db
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# Endpoint de recherche Guardian Content API. La documentation officielle
# recommande d'utiliser HTTPS systematiquement (content.guardianapis.com).
GUARDIAN_SEARCH_URL = "https://content.guardianapis.com/search"

# Nombre d'articles par requete. Le maximum autorise par l'API est 200, mais
# on se limite a 50 pour :
# 1. Garder des payloads MinIO de taille raisonnable.
# 2. Permettre un echantillonnage large et varie sans saturer le quota.
# Avec ~15 mots-cles x 50 articles, on vise 500-750 articles par collecte.
PAGE_SIZE = 50

# Timeout HTTP : Guardian repond generalement en < 2s, on garde 30s pour les
# cas rares de latence cote CDN (Fastly sert l'API).
HTTP_TIMEOUT = 30.0

# Rate limit cote client : Guardian Developer autorise 12 req/s, on prend 1s
# entre chaque requete pour garder de la marge face aux burst et aux retries
# eventuels. Avec ~15 mots-cles, la collecte prend ~15 secondes, negligeable.
MIN_DELAY_BETWEEN_REQUESTS = 1.0

# Retry avec backoff exponentiel sur erreur 429 (Too Many Requests). Guardian
# renvoie rarement 429 si on respecte le 12 req/s, mais on couvre ce cas pour
# robustesse (et pour les futurs pics de pipeline CI/CD).
MAX_RETRIES_ON_RATE_LIMIT = 3
INITIAL_BACKOFF_SECONDS = 5.0
BACKOFF_MULTIPLIER = 2.0

# Champs demandes a l'API. ``bodyText`` est le champ cle : il contient
# l'integralite du corps de l'article en texte brut (sans HTML). Les autres
# champs enrichissent le stockage MinIO pour d'eventuelles analyses futures.
SHOW_FIELDS = "bodyText,headline,trailText,byline,publication,lastModified,lang"


class GuardianRateLimitExceededError(Exception):
    """Levee quand l'API Guardian refuse plusieurs retries consecutifs.

    Indique que le budget journalier de 5000 requetes est probablement
    epuise ou qu'un burst a declenche un rate limit persistant. Le collecteur
    interrompt alors la boucle pour eviter de saturer davantage l'API.
    """


class GuardianCollector(BaseCollector):
    """Collecteur REST/JSON pour Guardian Content API.

    Interroge l'API avec les mots-cles configures dans
    ``search_config`` (type_source = ``"guardian"``), filtre les resultats
    et sauvegarde chaque lot brut dans MinIO raw-data.
    """

    def __init__(self) -> None:
        super().__init__(source_name="guardian")
        self.settings = get_settings()

    async def collect(
        self,
        keywords: list[str],
        **kwargs: Any,
    ) -> CollectResult:
        """Lance la collecte d'articles via Guardian Content API.

        Une requete par mot-cle est emise ; chaque reponse contient jusqu'a
        ``PAGE_SIZE`` articles avec leur ``bodyText`` complet, prets pour
        l'etape de nettoyage Spark.

        Args:
            keywords: Liste de mots-cles a rechercher.
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et compteurs.
        """
        result = CollectResult(source_name=self.source_name)

        if (
            not self.settings.guardian_api_key
            or self.settings.guardian_api_key == "YOUR_GUARDIAN_API_KEY_HERE"
        ):
            msg = (
                "Cle API Guardian non configuree (GUARDIAN_API_KEY dans .env). "
                "Obtenir une cle gratuite : https://open-platform.theguardian.com/access/"
            )
            logger.warning(msg)
            result.errors.append(msg)
            return result

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            for idx, keyword in enumerate(keywords):
                if idx > 0:
                    await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

                try:
                    articles = await self._fetch_articles_with_retry(client, keyword)
                    if not articles:
                        logger.info(f"Aucun article Guardian trouve pour '{keyword}'")
                        continue

                    raw_path = generate_raw_path("api", "guardian")
                    payload = {
                        "keyword": keyword,
                        "source": "theguardian.com",
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

                    logger.info(
                        f"Guardian '{keyword}' : {len(articles)} articles collectes -> {path}"
                    )

                except GuardianRateLimitExceededError as exc:
                    error_msg = (
                        f"Rate limit Guardian definitivement atteint pour '{keyword}' : {exc}"
                    )
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    logger.warning(
                        "Arret de la collecte Guardian : reessayer plus tard "
                        "(quota journalier probablement epuise)"
                    )
                    break
                except Exception as exc:
                    error_msg = f"Erreur collecte Guardian '{keyword}' : {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        logger.info(
            f"Collecte Guardian terminee : {result.articles_count} articles, "
            f"{len(result.errors)} erreurs"
        )
        return result

    async def _fetch_articles_with_retry(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Appelle ``_fetch_articles`` avec backoff exponentiel sur 429.

        Args:
            client: Client HTTP asynchrone partage entre requetes.
            keyword: Mot-cle a rechercher.

        Returns:
            Liste des articles normalises retournes par l'API.

        Raises:
            GuardianRateLimitExceededError: Si le rate limit persiste apres
                ``MAX_RETRIES_ON_RATE_LIMIT`` tentatives.
            httpx.HTTPStatusError: Sur toute autre erreur HTTP non-429.
        """
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES_ON_RATE_LIMIT + 1):
            try:
                return await self._fetch_articles(client, keyword)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429:
                    raise
                retry_after = exc.response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else backoff
                logger.warning(
                    f"429 Too Many Requests Guardian sur '{keyword}' "
                    f"(tentative {attempt}/{MAX_RETRIES_ON_RATE_LIMIT}), "
                    f"attente {wait_seconds:.1f}s avant retry"
                )
                await asyncio.sleep(wait_seconds)
                backoff *= BACKOFF_MULTIPLIER

        msg = f"Rate limit Guardian persistant apres {MAX_RETRIES_ON_RATE_LIMIT} tentatives"
        raise GuardianRateLimitExceededError(msg)

    async def _fetch_articles(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Interroge Guardian Content API pour un mot-cle donne.

        Args:
            client: Client HTTP asynchrone.
            keyword: Terme de recherche.

        Returns:
            Liste des articles normalises (titre, URL, contenu complet,
            metadonnees). Un article est ignore s'il n'a ni ``bodyText``
            ni titre exploitable.

        Raises:
            httpx.HTTPStatusError: Si la reponse HTTP est une erreur 4xx/5xx.
        """
        params = {
            "api-key": self.settings.guardian_api_key,
            "q": keyword,
            "show-fields": SHOW_FIELDS,
            "page-size": PAGE_SIZE,
            "order-by": "newest",
            # On laisse l'API chercher dans toutes les sections : les mots-cles
            # Green IT (data center, carbon footprint, sustainable AI, ...) sont
            # assez specifiques pour ne pas ramener du bruit, et on veut couvrir
            # autant "environment" que "technology" sans doublonner les requetes.
        }

        logger.debug(f"Requete Guardian : {keyword}")
        response = await client.get(GUARDIAN_SEARCH_URL, params=params)
        response.raise_for_status()

        data = response.json()
        body = data.get("response", {})
        if body.get("status") != "ok":
            logger.warning(
                f"Guardian status non-ok pour '{keyword}' : {body.get('status')} "
                f"(message: {body.get('message', 'n/a')})"
            )
            return []

        raw_articles = body.get("results", [])
        return self._parse_articles(raw_articles)

    def _parse_articles(
        self, raw_articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalise les articles bruts Guardian au format commun du pipeline.

        Extrait les champs pertinents et elimine les entrees sans contenu
        exploitable (bodyText vide). Cette etape pre-filtre les articles
        avant meme l'upload MinIO pour eviter de polluer le bucket avec des
        articles qui seraient de toute facon rejetes par le spark_cleaner.

        Args:
            raw_articles: Articles bruts depuis la reponse JSON Guardian.

        Returns:
            Articles normalises avec les champs attendus par le pipeline.
        """
        parsed = []
        for article in raw_articles:
            titre = article.get("webTitle")
            url = article.get("webUrl")
            fields = article.get("fields") or {}
            body_text = fields.get("bodyText", "").strip()

            if not titre or not url:
                logger.debug(
                    f"Article Guardian ignore (titre/URL manquant) : {article.get('id', 'N/A')}"
                )
                continue

            # Si le bodyText est vide (cas rare mais possible sur certains
            # contenus multimedia ou live blogs), on n'envoie rien en aval.
            # Mieux vaut perdre l'article que polluer le dataset avec des
            # entrees a 0 chars de contenu.
            if len(body_text) < 50:
                logger.debug(
                    f"Article Guardian ignore (contenu trop court : {len(body_text)} chars) : {titre[:60]}"
                )
                continue

            parsed.append(
                {
                    "titre": titre,
                    "url": url,
                    "description": fields.get("trailText", ""),
                    "contenu": body_text,
                    "date_publication": article.get("webPublicationDate"),
                    "auteur": fields.get("byline") or None,
                    "source_nom": "theguardian.com",
                    "langue": fields.get("lang", "en"),
                    "categorie": [article.get("sectionName", "")]
                    if article.get("sectionName")
                    else [],
                    "guardian_id": article.get("id"),
                }
            )

        return parsed


async def run_guardian_collection() -> CollectResult:
    """Point d'entree pour la collecte Guardian.

    Charge les mots-cles depuis la table ``search_config`` (filtre
    ``type_source = 'guardian'``) et lance la collecte.

    Returns:
        Resultat complet de la collecte (chemins MinIO, compteurs, erreurs).
    """
    logger.info("=== Demarrage collecte REST/JSON (The Guardian) ===")

    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="guardian")
        keywords = [cfg.mot_cle for cfg in configs]

    if not keywords:
        logger.warning(
            "Aucun mot-cle configure pour Guardian (type_source='guardian' "
            "dans search_config). Lancer les scripts d'init pour les inserer."
        )
        return CollectResult(source_name="guardian")

    collector = GuardianCollector()
    return await collector.collect(keywords)


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_guardian_collection())
