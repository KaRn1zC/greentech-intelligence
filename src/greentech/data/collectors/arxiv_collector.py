"""Collecteur REST/JSON : arXiv API (Atom XML).

Troisieme source REST/JSON du pipeline, complementaire a The Guardian
(presse generaliste) et Dev.to (vulgarisation developpeur). arXiv apporte
le registre academique : les abstracts de publications scientifiques qui
fondent l'etat de l'art du Green IT (efficient inference, model
compression, sustainable AI, carbon-aware computing, etc.).

La source existante ``arXiv Dataset`` (type=file) utilise le dump Kaggle
historique et couvre les categories cs.AI/cs.LG/cs.CL jusqu'a une date
figee. Ce collecteur API est vivant : il interroge l'index officiel en
quasi temps-reel, ce qui garantit un renouvellement regulier du dataset
d'entrainement au fil des publications Green IT.

Contraintes techniques
----------------------

L'API arXiv impose plusieurs regles de politesse documentees :

- ``max_results`` est techniquement limite a 30 000 par slice de 2 000,
  mais les requetes au-dela de 1 000 resultats sont decouragees. On se
  contente donc de la premiere tranche pertinente.
- Un delai de ``3 secondes`` est recommande entre requetes consecutives.
- Le service est expose en HTTPS via un redirect 301 depuis HTTP, il est
  donc imperatif de requeter directement ``https://export.arxiv.org``.
- Le format de reponse est Atom XML (non JSON) : feedparser fait le
  parsing transparent en dict Python.

Format des donnees
------------------

Chaque article arXiv fournit nativement un abstract (champ ``summary``
Atom) de 150-300 mots, qui joue le role de ``contenu`` dans le schema
BDD. Cette structure est coherente avec les articles ``arXiv Dataset``
deja ingeres : l'abstract est la feature principale utilisee par le
classifieur Qwen3-4B + LoRA via ``resume_classification``.

Usage
-----

    uv run python -m greentech.data.collectors.arxiv_collector

"""

from __future__ import annotations

import asyncio
from typing import Any

import feedparser
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

# Endpoint officiel de l'API arXiv. Toujours HTTPS : HTTP retourne un 301
# qui ajoute une latence inutile et casse le parsing feedparser sur la
# chaine redirigee si on oublie ``follow_redirects=True``.
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Nombre de resultats recuperes par requete. arXiv recommande des slices
# de 2 000 max mais, pour nos usages Green IT, 100 suffit largement et
# permet de rester sous le seuil de 1 000 resultats par query sans avoir
# a paginer. Si une query retourne plus, on fait plusieurs pages jusqu'a
# ``MAX_RESULTS_PER_KEYWORD``.
PAGE_SIZE = 100

# Plafond par mot-cle. Au-dela, on considere que la query est trop large
# et retourne trop de bruit. Les queries Green IT ciblees (ex: "carbon-
# aware computing") plafonnent naturellement bien en dessous.
MAX_RESULTS_PER_KEYWORD = 500

# Delai recommande par arXiv entre requetes consecutives. Respecte
# strictement pour eviter tout rate limit cote serveur et pour rester
# courtois avec une infrastructure scientifique publique. Le run du
# 2026-04-19 a revele que 3s ne suffit pas apres un smoke test recent :
# arXiv applique un cooldown persistant et renvoie 429 sur les 3-4
# premieres queries. 5s laisse une marge plus confortable.
MIN_DELAY_BETWEEN_REQUESTS = 5.0

# Timeout HTTP. arXiv peut etre lent sur de grosses queries ou en cas de
# latence intercontinentale (serveurs aux US / EU mirror), 60s couvre
# largement les pics.
HTTP_TIMEOUT = 60.0

# Retry avec backoff exponentiel sur 429 Too Many Requests. arXiv
# applique parfois un cooldown agressif meme avec le delai recommande,
# notamment apres un burst de requetes precedent (smoke tests, run
# interrompu). Trois tentatives avec des paliers de 10s -> 30s -> 90s
# couvrent la plupart des cooldowns observes.
MAX_RETRIES_ON_RATE_LIMIT = 3
INITIAL_BACKOFF_SECONDS = 10.0
BACKOFF_MULTIPLIER = 3.0

# Categories arXiv retenues pour le Green IT. On filtre apres fetch plutot
# que dans la query pour eviter la syntaxe ``cat:(A OR B OR ...)`` qui
# devient difficile a lire au-dela de 3-4 categories et pour garder la
# souplesse d'ajouter/retirer une categorie sans toucher aux queries.
# Prefixes : ``cs`` (Computer Science), ``eess`` (Electrical Engineering
# and Systems Science), ``stat.ML`` (Statistics & ML). Tout autre prefixe
# (math, physics, q-bio, etc.) est considere comme hors scope.
RELEVANT_CATEGORY_PREFIXES = ("cs.", "eess.", "stat.ML")

# Longueur minimale de l'abstract pour qu'un article soit conserve. Un
# abstract < 100 chars signale presque toujours un article incomplet ou
# un enregistrement errone dans l'index arXiv (ex: preprint retire ou
# duplique). On preserve ainsi la qualite du dataset d'entrainement.
MIN_ABSTRACT_LENGTH = 100


class ArxivCollector(BaseCollector):
    """Collecteur REST/JSON pour arXiv API.

    Interroge l'API arXiv avec les mots-cles configures dans
    ``search_config`` (``type_source = 'arxiv_api'``), filtre sur les
    categories Green IT pertinentes, et sauvegarde chaque lot brut dans
    MinIO raw-data. La source est identifiee ``source_name='arxiv_api'``
    pour la distinguer de la source ``arxiv`` (type=file) deja presente
    en BDD (import Kaggle historique).
    """

    def __init__(self) -> None:
        super().__init__(source_name="arxiv_api")
        self.settings = get_settings()
        # URLs deja en BDD, chargees au debut de collect() si skip_existing.
        # Applique lors du parsing de chaque entree : evite d'ajouter au
        # payload MinIO les articles deja connus (gain modeste, mais garde
        # le dataset MinIO propre et evite les re-ingestions inutiles).
        self._known_urls: set[str] = set()
        self._skipped_existing: int = 0

    async def collect(
        self,
        keywords: list[str],
        *,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> CollectResult:
        """Lance la collecte d'articles arXiv pour une liste de mots-cles.

        Pour chaque mot-cle, pagine jusqu'a ``MAX_RESULTS_PER_KEYWORD``
        resultats par slices de ``PAGE_SIZE``, filtre sur les categories
        tech pertinentes, et sauvegarde l'ensemble dans un seul objet
        MinIO par mot-cle. Le nom de l'objet inclut un timestamp pour
        garder l'historique des collectes.

        Args:
            keywords: Liste de mots-cles a rechercher dans le champ
                ``all`` d'arXiv (recherche dans titre + abstract +
                auteurs + commentaires).
            skip_existing: Si True, pre-charge les URLs deja en BDD et
                saute les articles connus pendant le parsing (gain
                modeste car la requete API batch renvoie les abstracts
                en une passe, mais evite la pollution MinIO et les
                re-ingestions).
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et compteurs.
        """
        result = CollectResult(source_name=self.source_name)

        if skip_existing:
            self._known_urls = await load_known_urls(self.source_name)

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": self.settings.scraping_user_agent},
        ) as client:
            for idx, keyword in enumerate(keywords):
                if idx > 0:
                    await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

                try:
                    articles = await self._fetch_paginated(client, keyword)
                    if not articles:
                        logger.info(f"Aucun article arXiv pour '{keyword}'")
                        continue

                    raw_path = generate_raw_path("api", "arxiv_api")
                    payload = {
                        "keyword": keyword,
                        "source": "arxiv.org",
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

                    logger.info(f"arXiv '{keyword}' : {len(articles)} articles collectes -> {path}")

                except Exception as exc:
                    error_msg = f"Erreur collecte arXiv '{keyword}' : {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        logger.info(
            f"Collecte arXiv terminee : {result.articles_count} articles, "
            f"{len(result.errors)} erreurs"
        )
        return result

    async def _fetch_paginated(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Fetch arXiv avec pagination jusqu'a ``MAX_RESULTS_PER_KEYWORD``.

        Chaque page est parsee individuellement pour pouvoir filtrer par
        categorie et s'arreter tot si le total de resultats est plus
        petit que la limite (economie de requetes).

        Args:
            client: Client HTTP asynchrone partage entre requetes.
            keyword: Mot-cle recherche dans le champ ``all`` d'arXiv.

        Returns:
            Liste des articles normalises, deduplique par arXiv id et
            filtre sur les categories pertinentes.
        """
        articles: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for page_idx, start in enumerate(
            range(0, MAX_RESULTS_PER_KEYWORD, PAGE_SIZE),
        ):
            if page_idx > 0:
                await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

            page_size = min(PAGE_SIZE, MAX_RESULTS_PER_KEYWORD - start)
            raw_entries, total_results = await self._fetch_page(
                client,
                keyword,
                start=start,
                max_results=page_size,
            )

            if not raw_entries:
                break

            for entry in raw_entries:
                parsed = self._parse_entry(entry, keyword)
                if parsed is None:
                    continue
                if parsed["arxiv_id"] in seen_ids:
                    continue
                seen_ids.add(parsed["arxiv_id"])
                articles.append(parsed)

            # arXiv retourne le total de resultats via opensearch:totalResults.
            # Si on l'a depasse, inutile de pousser plus loin la pagination.
            if start + page_size >= total_results:
                break

        return articles

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        keyword: str,
        *,
        start: int,
        max_results: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Recupere une page arXiv avec retry/backoff sur 429 Too Many Requests.

        Args:
            client: Client HTTP asynchrone.
            keyword: Mot-cle recherche dans le champ ``all``.
            start: Offset de pagination.
            max_results: Nombre de resultats a retourner pour cette page.

        Returns:
            Couple ``(entries, total_results)`` ou ``entries`` est la liste
            brute des entrees feedparser et ``total_results`` le nombre
            total d'articles correspondant a la query (tous pages).

        Raises:
            httpx.HTTPStatusError: Sur toute erreur HTTP 4xx/5xx autre que 429
                persistant apres epuisement des retries.
        """
        params = {
            "search_query": f'all:"{keyword}"',
            "start": start,
            "max_results": max_results,
        }
        logger.debug(
            f"Requete arXiv : keyword='{keyword}', start={start}, max_results={max_results}"
        )

        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(1, MAX_RETRIES_ON_RATE_LIMIT + 1):
            try:
                response = await client.get(ARXIV_API_URL, params=params)
                response.raise_for_status()

                feed = feedparser.parse(response.text)
                total_results = int(
                    feed.feed.get("opensearch_totalresults", 0) if hasattr(feed, "feed") else 0
                )
                entries = list(feed.entries) if hasattr(feed, "entries") else []
                return entries, total_results
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429:
                    raise
                retry_after = exc.response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else backoff
                logger.warning(
                    f"429 Too Many Requests arXiv sur '{keyword}' start={start} "
                    f"(tentative {attempt}/{MAX_RETRIES_ON_RATE_LIMIT}), "
                    f"attente {wait_seconds:.1f}s avant retry"
                )
                await asyncio.sleep(wait_seconds)
                backoff *= BACKOFF_MULTIPLIER

        # Tous les retries ont echoue : on remonte l'erreur pour que le
        # collecteur la logue et passe au mot-cle suivant.
        msg = f"Rate limit arXiv persistant apres {MAX_RETRIES_ON_RATE_LIMIT} tentatives"
        raise RuntimeError(msg)

    def _parse_entry(
        self,
        entry: dict[str, Any],
        keyword: str,
    ) -> dict[str, Any] | None:
        """Normalise une entree feedparser arXiv au format commun.

        Filtre sur les categories Green IT pertinentes et rejette les
        entrees sans abstract exploitable. Tous les champs attendus par
        le spark_cleaner sont presents : titre, url, contenu, auteur
        (liste separee par virgule), date_publication, langue.

        Args:
            entry: Entree Atom XML parsee par feedparser.
            keyword: Mot-cle qui a declenche cette requete (tracabilite).

        Returns:
            Article normalise, ou ``None`` si rejete (hors scope categorie
            ou abstract trop court).
        """
        # arXiv id format: http://arxiv.org/abs/1234.56789v1 ou similaire.
        # On preserve l'URL abs/ sans version pour la deduplication par DOI
        # arXiv, mais on garde l'id complet pour la tracabilite.
        arxiv_id = entry.get("id", "")
        if not arxiv_id:
            return None

        # Recuperation des categories (primary + autres). feedparser les
        # expose via tags (list de dict) ou arxiv_primary_category selon
        # la version. On accepte les deux formes pour robustesse.
        tags = entry.get("tags", []) or []
        categories: list[str] = []
        for tag in tags:
            term = tag.get("term") if isinstance(tag, dict) else None
            if term:
                categories.append(term)

        primary_category = None
        if "arxiv_primary_category" in entry:
            primary = entry["arxiv_primary_category"]
            if isinstance(primary, dict):
                primary_category = primary.get("term")

        # Filtre sur categories pertinentes. Au moins une categorie doit
        # commencer par un prefixe retenu pour que l'article soit accepte.
        relevant = any(cat.startswith(RELEVANT_CATEGORY_PREFIXES) for cat in categories)
        if not relevant:
            return None

        titre = (entry.get("title") or "").strip().replace("\n", " ")
        summary = (entry.get("summary") or "").strip().replace("\n", " ")

        if not titre or len(summary) < MIN_ABSTRACT_LENGTH:
            return None

        # URL canonique sans version (v1, v2, ...) pour faciliter la
        # deduplication : un meme papier publie en v1 puis v2 generera
        # la meme URL et sera rejete par la contrainte UNIQUE articles.url.
        canonical_url = arxiv_id.rstrip("/")
        if "v" in canonical_url.split("/")[-1]:
            canonical_url = canonical_url.rsplit("v", 1)[0]

        # Pre-check BDD (normalise) : saute les articles deja ingeres.
        # Comparaison tolerante au scheme (http vs https : crucial car le
        # dataset Kaggle historique stocke https:// et l'API retourne http://).
        if url_is_known(canonical_url, self._known_urls):
            self._skipped_existing += 1
            return None

        # Authors : liste feedparser -> string separe par virgule. Sera
        # anonymisee en aval par spark_cleaner (registre RGPD).
        authors = entry.get("authors", []) or []
        author_names = []
        for author in authors[:5]:  # Plafond a 5 pour limiter les noms lourds
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    author_names.append(name)
        auteur = ", ".join(author_names) if author_names else None

        return {
            "titre": titre[:500],  # Contrainte BDD
            "url": canonical_url,
            "description": summary[:500],  # Resume court pour affichage
            "contenu": summary,  # Abstract integral = contenu principal
            "date_publication": entry.get("published"),
            "auteur": auteur,
            "source_nom": "arxiv.org",
            "langue": "en",
            "categorie": categories,
            "categorie_principale": primary_category,
            "arxiv_id": canonical_url.rsplit("/", 1)[-1],
            "keyword_search": keyword,
        }


async def run_arxiv_collection() -> CollectResult:
    """Point d'entree pour la collecte arXiv API.

    Charge les mots-cles depuis la table ``search_config`` (filtre
    ``type_source = 'arxiv_api'``) et lance la collecte.

    Returns:
        Resultat complet de la collecte.
    """
    logger.info("=== Demarrage collecte REST/JSON (arXiv API) ===")

    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="arxiv_api")
        keywords = [cfg.mot_cle for cfg in configs]

    if not keywords:
        logger.warning(
            "Aucun mot-cle configure pour arXiv API (type_source='arxiv_api' "
            "dans search_config). Lancer les scripts d'init pour les inserer."
        )
        return CollectResult(source_name="arxiv_api")

    collector = ArxivCollector()
    return await collector.collect(keywords)


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_arxiv_collection())
