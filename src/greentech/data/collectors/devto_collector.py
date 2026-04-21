"""Collecteur REST/JSON : Dev.to / Forem API.

Source complementaire a The Guardian pour diversifier les registres du
dataset : Dev.to heberge des articles ecrits par des developpeurs sur la
tech, avec un vocabulaire plus pratique et technique (outils, frameworks,
retours d'experience) la ou Guardian apporte un registre journalistique
plus narratif. Les tags ``greenit``, ``sustainability``, ``climatechange``,
``webperf`` et similaires sont tres pertinents pour le Green IT logiciel.

Architecture specifique : Dev.to impose un pattern a deux appels
-----------------------------------------------------------------

L'endpoint ``/api/articles?tag=X`` retourne une LISTE d'articles mais
**sans** le champ ``body_markdown``, qui contient le contenu integral. Ce
champ n'est disponible que via l'endpoint de detail ``/api/articles/{id}``.
Il faut donc :

1. Lister les articles par tag (pagination ``per_page=30`` par defaut).
2. Pour chaque article, faire un GET sur son id pour obtenir ``body_markdown``.
3. Nettoyer le markdown (images, code blocks, liens) avant stockage.

Ce pattern coute 1 + N requetes par tag, mais Dev.to n'impose pas de cle
API ni de rate limit dur en lecture publique : on se contente d'une petite
pause entre appels pour rester courtois.

Usage
-----

    uv run python -m greentech.data.collectors.devto_collector

"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings
from greentech.data.collectors.base import BaseCollector, CollectResult, get_config_from_db
from greentech.data.collectors.url_precheck import load_known_urls, url_is_known
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

# Endpoints Dev.to / Forem API v1. Les deux URL sont publiques et stables
# depuis plusieurs annees (l'API est celle qui alimente aussi les clients
# mobiles officiels).
DEVTO_LIST_URL = "https://dev.to/api/articles"
DEVTO_DETAIL_URL = "https://dev.to/api/articles/{id}"

# Nombre d'articles par page (max autorise = 1000, mais 30 suffit largement
# par tag tout en gardant des payloads MinIO de taille raisonnable).
PER_PAGE = 30

# Timeout HTTP. Dev.to est generalement tres rapide (< 500ms), 30s couvre
# les cas rares de latence.
HTTP_TIMEOUT = 30.0

# Pause courte entre appels pour rester courtois avec l'API publique. Dev.to
# n'affiche pas de rate limit dur, mais 0.3s laisse ~3 req/s par collecteur.
MIN_DELAY_BETWEEN_REQUESTS = 0.3

# Longueur minimale du body_markdown pour considerer l'article exploitable.
# En dessous, on estime que c'est trop peu pour en tirer un resume utile
# (tweet-like, quick tip, post vide).
MIN_BODY_LENGTH = 300

# Regex de nettoyage du markdown. On retire les artefacts qui introduisent
# du bruit dans le texte transmis au LLM de resume :
# - Blocs de code (`` ``` ``) qui sont rarement pertinents pour Green IT
#   et qui peuvent "polluer" le resume avec du syntaxe code
# - Images ``![alt](url)``
# - Liens ``[text](url)`` qu'on reduit a ``text``
# - Titres ``#``, ``##`` qu'on garde mais sans le markup
# - HTML residuel (quelques articles contiennent du HTML inline)
RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
RE_INLINE_CODE = re.compile(r"`([^`]+)`")
RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_HEADER_MARKUP = re.compile(r"^#{1,6}\s*", re.MULTILINE)
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
RE_MULTI_SPACES = re.compile(r"[ \t]{2,}")


class DevtoCollector(BaseCollector):
    """Collecteur REST/JSON pour Dev.to / Forem API.

    Interroge l'API publique avec les tags configures dans ``search_config``
    (``type_source = 'devto'``). Pour chaque article liste, recupere le corps
    complet (``body_markdown``) via un second appel, nettoie le markdown, et
    stocke le tout dans MinIO raw-data.
    """

    def __init__(self) -> None:
        super().__init__(source_name="devto")
        self.settings = get_settings()
        # URLs deja en BDD, chargees une seule fois au debut de collect().
        # Evite de fetcher le detail d'articles deja connus (pattern Dev.to
        # = 1 requete liste + N requetes detail, skip = economie grosse).
        self._known_urls: set[str] = set()
        self._skipped_existing: int = 0

    async def collect(
        self,
        keywords: list[str],
        *,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> CollectResult:
        """Collecte les articles Dev.to pour les tags donnes.

        Chaque mot-cle de ``keywords`` est interprete comme un tag Dev.to
        (sans le prefixe ``#``). Les tags valides pour Green IT incluent
        notamment : ``greenit``, ``sustainability``, ``climatechange``,
        ``webperf``, ``cleanenergy``.

        Args:
            keywords: Liste de tags Dev.to a interroger.
            skip_existing: Si True, pre-charge les URLs deja en BDD et
                saute les articles connus avant le fetch detail (grosse
                economie sur re-run). True par defaut.
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et compteurs.
        """
        result = CollectResult(source_name=self.source_name)

        if skip_existing:
            self._known_urls = await load_known_urls(self.source_name)

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": self.settings.scraping_user_agent},
        ) as client:
            for idx, tag in enumerate(keywords):
                if idx > 0:
                    await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

                try:
                    articles = await self._fetch_articles_with_bodies(client, tag)
                    if not articles:
                        logger.info(f"Aucun article Dev.to exploitable pour tag '{tag}'")
                        continue

                    raw_path = generate_raw_path("api", "devto")
                    payload = {
                        "tag": tag,
                        "source": "dev.to",
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
                        f"Dev.to tag '{tag}' : {len(articles)} articles collectes -> {path}"
                    )

                except Exception as exc:
                    error_msg = f"Erreur collecte Dev.to tag '{tag}' : {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        logger.info(
            f"Collecte Dev.to terminee : {result.articles_count} articles nouveaux, "
            f"{self._skipped_existing} deja en BDD (skip), "
            f"{len(result.errors)} erreurs"
        )
        return result

    async def _fetch_articles_with_bodies(
        self,
        client: httpx.AsyncClient,
        tag: str,
    ) -> list[dict[str, Any]]:
        """Liste les articles d'un tag puis fetch le body de chacun.

        Args:
            client: Client HTTP asynchrone partage entre requetes.
            tag: Tag Dev.to a interroger.

        Returns:
            Articles normalises avec body_markdown integral deja nettoye.
        """
        params = {"tag": tag, "per_page": PER_PAGE, "page": 1}
        logger.debug(f"Listing Dev.to : tag={tag}")

        response = await client.get(DEVTO_LIST_URL, params=params)
        response.raise_for_status()
        listed = response.json()

        if not isinstance(listed, list):
            logger.warning(f"Reponse Dev.to inattendue pour tag '{tag}' : {type(listed).__name__}")
            return []

        logger.debug(f"Dev.to tag={tag} : {len(listed)} articles listes, fetch des bodies")

        articles: list[dict[str, Any]] = []
        skipped_in_tag = 0
        for summary in listed:
            article_id = summary.get("id")
            if article_id is None:
                continue

            # Pre-check BDD avant le fetch detail. La liste Dev.to inclut
            # deja l'URL canonique dans ``summary["url"]``, donc on peut
            # skipper sans faire le detail HTTP si l'article est connu.
            # Comparaison normalisee (scheme http/https, trailing slash).
            summary_url = summary.get("url")
            if url_is_known(summary_url, self._known_urls):
                skipped_in_tag += 1
                self._skipped_existing += 1
                continue

            # On applique le delay APRES le skip pour ne pas penaliser les
            # re-runs qui sautent massivement : pas d'appel HTTP, pas de
            # delay necessaire. Le delay courtois s'applique uniquement
            # aux articles pour lesquels on fait un fetch detail reel.
            if articles:
                await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

            try:
                detail = await self._fetch_article_detail(client, int(article_id))
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    f"Echec fetch detail Dev.to id={article_id} : HTTP {exc.response.status_code}"
                )
                continue
            except Exception as exc:
                logger.warning(f"Echec fetch detail Dev.to id={article_id} : {exc}")
                continue

            normalized = self._parse_article(detail, summary, tag)
            if normalized is not None:
                articles.append(normalized)

        if skipped_in_tag > 0:
            logger.info(
                f"Dev.to tag='{tag}' : {skipped_in_tag} articles sautes (deja en BDD), "
                f"{len(articles)} nouveaux fetches"
            )
        return articles

    async def _fetch_article_detail(
        self,
        client: httpx.AsyncClient,
        article_id: int,
    ) -> dict[str, Any]:
        """Recupere le detail d'un article Dev.to (incluant body_markdown).

        Args:
            client: Client HTTP asynchrone.
            article_id: Identifiant numerique de l'article.

        Returns:
            Dictionnaire JSON complet de l'article.

        Raises:
            httpx.HTTPStatusError: Sur toute erreur HTTP 4xx/5xx.
        """
        response = await client.get(DEVTO_DETAIL_URL.format(id=article_id))
        response.raise_for_status()
        return response.json()

    def _parse_article(
        self,
        detail: dict[str, Any],
        summary: dict[str, Any],
        tag: str,
    ) -> dict[str, Any] | None:
        """Normalise un article Dev.to (detail + resume liste) au format commun.

        Combine les champs issus du detail (notamment ``body_markdown``) et
        ceux du resume de liste (``tag_list``, metadonnees) pour produire
        une entree compatible avec le pipeline Spark/SQL en aval.

        Args:
            detail: Reponse du GET ``/api/articles/{id}``.
            summary: Entree correspondante dans la reponse de listing.
            tag: Tag qui a decouvert cet article (pour la tracabilite).

        Returns:
            Article normalise, ou ``None`` si le contenu est trop court
            apres nettoyage.
        """
        titre = detail.get("title") or summary.get("title")
        url = detail.get("url") or summary.get("url")
        body_markdown = detail.get("body_markdown") or ""

        if not titre or not url:
            return None

        contenu = _clean_markdown(body_markdown)
        if len(contenu) < MIN_BODY_LENGTH:
            logger.debug(
                f"Article Dev.to ignore (corps trop court apres nettoyage : "
                f"{len(contenu)} chars) : {titre[:60]}"
            )
            return None

        user = detail.get("user") or summary.get("user") or {}
        auteur_parts = [user.get("name"), user.get("username")]
        auteur = next((p for p in auteur_parts if p), None)

        tag_list = detail.get("tag_list") or summary.get("tag_list") or []
        if isinstance(tag_list, str):
            tag_list = [t.strip() for t in tag_list.split(",") if t.strip()]

        return {
            "titre": titre,
            "url": url,
            "description": detail.get("description") or summary.get("description", ""),
            "contenu": contenu,
            "date_publication": detail.get("published_at") or summary.get("published_at"),
            "auteur": auteur,
            "source_nom": "dev.to",
            "langue": "en",
            "categorie": tag_list,
            "devto_id": detail.get("id"),
            "tag_recherche": tag,
        }


def _clean_markdown(markdown: str) -> str:
    """Nettoie un corps markdown pour le rendre exploitable par les LLM.

    Retire les blocs de code (rarement pertinents pour Green IT et
    polluants dans un resume), les images, reduit les liens a leur texte,
    retire le markup des titres, nettoie le HTML residuel et normalise
    les espaces. Le resultat est une prose quasi-plate, prete a etre
    tokenisee par le summarizer.

    Args:
        markdown: Contenu brut en markdown depuis ``body_markdown``.

    Returns:
        Texte nettoye, debarasse des artefacts markdown les plus gros.
    """
    if not markdown:
        return ""

    text = RE_CODE_BLOCK.sub(" ", markdown)
    text = RE_IMAGE.sub(" ", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_INLINE_CODE.sub(r"\1", text)
    text = RE_HEADER_MARKUP.sub("", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = RE_MULTI_NEWLINES.sub("\n\n", text)
    text = RE_MULTI_SPACES.sub(" ", text)
    return text.strip()


async def run_devto_collection() -> CollectResult:
    """Point d'entree pour la collecte Dev.to.

    Charge les tags depuis ``search_config`` (filtre
    ``type_source='devto'``) et lance la collecte.

    Returns:
        Resultat complet (chemins MinIO, compteurs, erreurs).
    """
    logger.info("=== Demarrage collecte REST/JSON (Dev.to) ===")

    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="devto")
        tags = [cfg.mot_cle for cfg in configs]

    if not tags:
        logger.warning(
            "Aucun tag configure pour Dev.to (type_source='devto' dans search_config). "
            "Lancer les scripts d'init pour inserer les tags par defaut."
        )
        return CollectResult(source_name="devto")

    collector = DevtoCollector()
    return await collector.collect(tags)


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_devto_collection())
