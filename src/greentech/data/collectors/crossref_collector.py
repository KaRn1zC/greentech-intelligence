"""Collecteur REST/JSON : Crossref API.

Quatrieme source REST/JSON du pipeline, complementaire a arXiv pour le
registre academique. La (seule vraie) difference : Crossref indexe les
publications **editoriales** (revues peer-reviewed, conference proceedings
publies par des editeurs comme Springer, Elsevier, IEEE, ACM) la ou arXiv
se concentre sur les preprints. Les deux sources se recoupent partiellement
mais apportent des signaux differents sur le Green IT :

- arXiv : recherche en train, souvent plus recente et libre d'acces.
- Crossref : publications validees par pairs, avec editeurs, en acces
  payant pour beaucoup (mais on ne lit que les abstracts publics).

Contraintes techniques
----------------------

L'API Crossref impose plusieurs bonnes pratiques documentees :

- **Polite Pool** : envoyer un ``mailto`` dans le User-Agent permet
  d'atterrir sur un pool prioritaire avec des latences plus stables. Le
  pool public marche aussi mais peut etre limite en cas de charge.
- ``rows`` maximum = 1000 par requete, 20 par defaut.
- Pagination profonde via cursor qui expire apres 5 minutes : on ne
  l'utilise pas (top-N suffit pour un dataset Green IT cible).
- Filtre crucial : ``has-abstract:true`` elimine les entrees sans
  abstract (env. 40 % du corpus Crossref). Sans lui, on accumule des
  references inutilisables (pas de feature d'entrainement).

Format des donnees
------------------

Les abstracts Crossref sont encadres par des balises JATS (Journal
Article Tag Suite), typiquement ``<jats:p>...</jats:p>``. On les strippe
au moment du parsing pour obtenir du texte plat compatible avec le
pipeline Spark/LLM downstream. L'URL canonique est ``https://doi.org/<DOI>``
qui redirige vers la page editeur (stable dans le temps, unique par DOI).

Usage
-----

    uv run python -m greentech.data.collectors.crossref_collector

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

# Endpoint racine de l'API Crossref. Le service est expose en HTTPS
# uniquement, sans redirect HTTP -> on attaque directement.
CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Nombre de resultats retournes par requete. On utilise le maximum officiel
# (1000) pour les queries les plus genereuses, mais le plafond effectif
# par mot-cle est fixe plus bas (``MAX_RESULTS_PER_KEYWORD``) parce que
# passe un certain rang la pertinence s'ecroule.
PAGE_SIZE = 200

# Plafond par mot-cle. Les top 200 resultats Crossref (tries par score
# de pertinence) couvrent largement les articles significatifs pour une
# query Green IT ciblee ; au-dela, on descend dans des papiers tangents
# ou faiblement relies au mot-cle.
MAX_RESULTS_PER_KEYWORD = 200

# Date minimum de publication. Les publications pre-2020 sont deja bien
# couvertes par le dataset arXiv Kaggle historique ; pour ce collecteur
# on se concentre sur les productions recentes qui enrichissent le
# corpus de facon incrementale.
MIN_PUBLICATION_YEAR = 2020

# Delai entre requetes pour rester courtois avec Crossref. Pas de rate
# limit officiel documente, mais 0.5s laisse ~2 req/s par collecteur
# ce qui est parfaitement raisonnable pour une API academique publique.
MIN_DELAY_BETWEEN_REQUESTS = 0.5

# Timeout HTTP. Crossref peut etre lent sur les requetes filtrees larges,
# 60s couvre les cas de latence.
HTTP_TIMEOUT = 60.0

# Longueur minimale de l'abstract net (apres strip JATS). En dessous, on
# considere l'entree comme exploitable uniquement a la marge.
MIN_ABSTRACT_LENGTH = 100

# Types d'oeuvres Crossref retenus. On ecarte les livres, chapitres,
# rapports gouvernementaux et autres formats qui ne matchent pas notre
# pattern d'entrainement (articles de recherche avec abstract court).
RELEVANT_TYPES = ("journal-article", "proceedings-article")

# Regex pour extraire le texte des balises JATS autour des abstracts.
# On garde le contenu interieur de toutes les balises et on rebalance
# les espaces a la fin pour obtenir du texte propre.
RE_JATS_TAG = re.compile(r"<[^>]+>")
RE_MULTI_SPACES = re.compile(r"\s+")


class CrossrefCollector(BaseCollector):
    """Collecteur REST/JSON pour Crossref API.

    Interroge Crossref avec les mots-cles configures dans ``search_config``
    (``type_source = 'crossref'``), en recherche par titre pour maximiser
    la precision, filtre sur les articles avec abstract, et sauvegarde
    chaque lot brut dans MinIO raw-data.
    """

    def __init__(self) -> None:
        super().__init__(source_name="crossref")
        self.settings = get_settings()
        # URLs deja en BDD, chargees au debut de collect() si skip_existing.
        self._known_urls: set[str] = set()
        self._skipped_existing: int = 0

    async def collect(
        self,
        keywords: list[str],
        *,
        skip_existing: bool = True,
        **kwargs: Any,
    ) -> CollectResult:
        """Lance la collecte Crossref pour une liste de mots-cles.

        Pour chaque mot-cle, recupere jusqu'a ``MAX_RESULTS_PER_KEYWORD``
        articles tries par pertinence (score Crossref), filtre sur les
        types pertinents et la presence d'abstract, et sauvegarde tout
        dans un unique objet MinIO.

        Args:
            keywords: Liste de mots-cles (recherche dans ``query.title``
                pour precision).
            skip_existing: Si True, pre-charge les URLs deja en BDD et
                saute les articles connus pendant le parsing.
            **kwargs: Parametres additionnels (non utilises).

        Returns:
            Resultat de la collecte avec chemins MinIO et compteurs.
        """
        result = CollectResult(source_name=self.source_name)
        user_agent = self._build_user_agent()

        if skip_existing:
            self._known_urls = await load_known_urls(self.source_name)

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": user_agent},
        ) as client:
            for idx, keyword in enumerate(keywords):
                if idx > 0:
                    await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

                try:
                    articles = await self._fetch_paginated(client, keyword)
                    if not articles:
                        logger.info(f"Aucun article Crossref pour '{keyword}'")
                        continue

                    raw_path = generate_raw_path("api", "crossref")
                    payload = {
                        "keyword": keyword,
                        "source": "api.crossref.org",
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
                        f"Crossref '{keyword}' : {len(articles)} articles collectes -> {path}"
                    )

                except Exception as exc:
                    error_msg = f"Erreur collecte Crossref '{keyword}' : {exc}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)

        logger.info(
            f"Collecte Crossref terminee : {result.articles_count} articles, "
            f"{len(result.errors)} erreurs"
        )
        return result

    def _build_user_agent(self) -> str:
        """Construit le User-Agent Crossref, avec mailto Polite Pool si dispo.

        Si ``settings.crossref_mailto`` est renseigne, on fournit l'email
        dans le User-Agent au format recommande par Crossref. Sinon, on
        se rabat sur le User-Agent generique du projet (pool public).

        Returns:
            String User-Agent prete a injecter dans les headers HTTP.
        """
        base = self.settings.scraping_user_agent
        mailto = self.settings.crossref_mailto.strip()
        if mailto:
            return f"{base} (mailto:{mailto})"
        return base

    async def _fetch_paginated(
        self,
        client: httpx.AsyncClient,
        keyword: str,
    ) -> list[dict[str, Any]]:
        """Fetch Crossref avec pagination jusqu'a ``MAX_RESULTS_PER_KEYWORD``.

        Args:
            client: Client HTTP asynchrone partage entre requetes.
            keyword: Mot-cle recherche dans ``query.title`` Crossref.

        Returns:
            Liste des articles normalises, deduplique par DOI et filtre
            sur abstract present + type de publication pertinent.
        """
        articles: list[dict[str, Any]] = []
        seen_dois: set[str] = set()

        for page_idx, offset in enumerate(
            range(0, MAX_RESULTS_PER_KEYWORD, PAGE_SIZE),
        ):
            if page_idx > 0:
                await asyncio.sleep(MIN_DELAY_BETWEEN_REQUESTS)

            page_size = min(PAGE_SIZE, MAX_RESULTS_PER_KEYWORD - offset)
            raw_items, total_results = await self._fetch_page(
                client,
                keyword,
                offset=offset,
                rows=page_size,
            )
            if not raw_items:
                break

            for item in raw_items:
                parsed = self._parse_item(item, keyword)
                if parsed is None:
                    continue
                if parsed["doi"] in seen_dois:
                    continue
                seen_dois.add(parsed["doi"])
                articles.append(parsed)

            if offset + page_size >= total_results:
                break

        return articles

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        keyword: str,
        *,
        offset: int,
        rows: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Recupere une page Crossref et retourne items + total-results.

        Args:
            client: Client HTTP asynchrone.
            keyword: Mot-cle recherche par ``query.title``.
            offset: Offset de pagination via ``offset`` param.
            rows: Nombre de resultats a retourner pour cette page.

        Returns:
            Couple ``(items, total_results)``.

        Raises:
            httpx.HTTPStatusError: Sur toute erreur HTTP 4xx/5xx.
        """
        params = {
            "query.title": keyword,
            "rows": rows,
            "offset": offset,
            "filter": (
                f"has-abstract:true,from-pub-date:{MIN_PUBLICATION_YEAR},type:journal-article"
            ),
            "sort": "relevance",
            "order": "desc",
        }
        logger.debug(f"Requete Crossref : keyword='{keyword}', offset={offset}, rows={rows}")

        response = await client.get(CROSSREF_WORKS_URL, params=params)
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        items = message.get("items", []) or []
        total_results = int(message.get("total-results", 0))

        return items, total_results

    def _parse_item(
        self,
        item: dict[str, Any],
        keyword: str,
    ) -> dict[str, Any] | None:
        """Normalise un item Crossref au format commun du pipeline.

        Les abstracts Crossref sont encadres par des balises JATS que l'on
        strippe ici pour obtenir du texte plat. On conserve le DOI dans
        un champ dedie (pour le versioning et la tracabilite) et on
        construit l'URL canonique ``https://doi.org/<DOI>`` qui sert de
        cle d'unicite pour la table ``articles``.

        Args:
            item: Entree brute depuis la reponse JSON Crossref.
            keyword: Mot-cle qui a declenche cette requete.

        Returns:
            Article normalise, ou ``None`` si rejete (type hors scope,
            pas d'abstract, titre vide, etc.).
        """
        work_type = item.get("type")
        if work_type not in RELEVANT_TYPES:
            return None

        doi = item.get("DOI")
        if not doi:
            return None

        titles = item.get("title") or []
        titre = (titles[0] if titles else "").strip().replace("\n", " ")
        if not titre:
            return None

        raw_abstract = item.get("abstract") or ""
        contenu = _strip_jats_tags(raw_abstract)
        if len(contenu) < MIN_ABSTRACT_LENGTH:
            return None

        # Auteurs : on concatene given + family dans l'ordre naturel et
        # on plafonne a 5 noms pour ne pas alourdir le champ ``auteur``.
        authors = item.get("author") or []
        author_names: list[str] = []
        for author in authors[:5]:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            full = " ".join(part for part in (given, family) if part)
            if full:
                author_names.append(full)
        auteur = ", ".join(author_names) if author_names else None

        # Date de publication : Crossref fournit plusieurs champs
        # (published-print, published-online, created, etc.). On prend
        # le premier disponible dans l'ordre le plus fiable.
        date_publication = _extract_date(item)

        # Langue : Crossref renvoie parfois le champ ``language`` (ISO).
        langue = (item.get("language") or "en").lower()[:10]

        # URL canonique via le resolveur DOI (stable dans le temps,
        # redirect vers l'editeur en vigueur).
        canonical_url = f"https://doi.org/{doi}"

        # Pre-check BDD (normalise) : saute les articles deja ingeres.
        if url_is_known(canonical_url, self._known_urls):
            self._skipped_existing += 1
            return None

        # Categorie / subject : Crossref a un champ ``subject`` parfois
        # rempli avec les categories Scopus / disciplines ACM.
        subjects = item.get("subject") or []

        return {
            "titre": titre[:500],  # Contrainte BDD
            "url": canonical_url,
            "description": contenu[:500],
            "contenu": contenu,
            "date_publication": date_publication,
            "auteur": auteur,
            "source_nom": "crossref.org",
            "langue": langue,
            "categorie": subjects[:10],
            "doi": doi,
            "type_crossref": work_type,
            "keyword_search": keyword,
        }


def _strip_jats_tags(raw: str) -> str:
    """Retire les balises JATS et normalise les espaces d'un abstract Crossref.

    Les abstracts Crossref sont frequemment structures avec des balises
    ``<jats:p>``, ``<jats:sec>`` etc. On les retire toutes pour obtenir
    du texte plat exploitable par le LLM de classification.

    Args:
        raw: Abstract brut avec balises JATS.

    Returns:
        Abstract nettoye, espaces normalises. Chaine vide si ``raw`` vide.
    """
    if not raw:
        return ""
    text = RE_JATS_TAG.sub(" ", raw)
    text = RE_MULTI_SPACES.sub(" ", text)
    return text.strip()


def _extract_date(item: dict[str, Any]) -> str | None:
    """Extrait la date de publication d'un item Crossref au format ISO.

    Crossref retourne les dates sous forme de ``date-parts`` (liste de
    listes d'entiers : ``[[2024, 3, 15]]``). On prend la premiere date
    valide dans l'ordre ``published-print -> published-online -> created``
    et on la formate en ISO 8601 string.

    Args:
        item: Entree Crossref complete.

    Returns:
        Date ISO 8601 (``YYYY-MM-DD``) si trouvee, sinon ``None``.
    """
    for key in ("published-print", "published-online", "issued", "created"):
        date_field = item.get(key)
        if not date_field:
            continue
        parts = date_field.get("date-parts") or []
        if not parts or not parts[0]:
            continue
        date_parts = parts[0]
        # On peut avoir [YYYY], [YYYY, M] ou [YYYY, M, D]. On completeavec des 1 pour former une date ISO valide.
        year = date_parts[0] if len(date_parts) > 0 else None
        if year is None:
            continue
        month = date_parts[1] if len(date_parts) > 1 else 1
        day = date_parts[2] if len(date_parts) > 2 else 1
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


async def run_crossref_collection() -> CollectResult:
    """Point d'entree pour la collecte Crossref.

    Charge les mots-cles depuis la table ``search_config`` (filtre
    ``type_source = 'crossref'``) et lance la collecte.

    Returns:
        Resultat complet de la collecte.
    """
    logger.info("=== Demarrage collecte REST/JSON (Crossref) ===")

    async with async_session_factory() as session:
        configs = await get_config_from_db(session, type_source="crossref")
        keywords = [cfg.mot_cle for cfg in configs]

    if not keywords:
        logger.warning(
            "Aucun mot-cle configure pour Crossref (type_source='crossref' "
            "dans search_config). Lancer les scripts d'init pour les inserer."
        )
        return CollectResult(source_name="crossref")

    collector = CrossrefCollector()
    return await collector.collect(keywords)


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_crossref_collection())
