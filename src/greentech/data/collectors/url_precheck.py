"""Pre-check BDD des URLs avant fetch : module partage entre tous les collecteurs.

Introduit en B2.3 pour les spiders statiques puis etendu a l'ensemble des
collecteurs (REST/JSON, scraping hybride, fichier). Le principe : au
demarrage d'une collecte, on charge toutes les URLs deja presentes dans
``articles.url`` en BDD dans un set Python, et on filtre les URLs a
fetcher contre ce set avant de faire l'appel reseau.

Normalisation pour eviter les faux negatifs
-------------------------------------------

Certaines sources stockent les URLs avec des variantes sans consequence
metier mais qui cassent la comparaison exacte :

- ``http://`` vs ``https://`` : le ``file_ingester`` (Kaggle arXiv) stocke
  ``https://arxiv.org/abs/...`` tandis que l'``arxiv_collector`` (API)
  recupere le champ Atom ``id`` qui est ``http://arxiv.org/abs/...``.
  Sans normalisation, un re-run arxiv_collector re-fetcherait tous les
  articles deja presents via file_ingester ET creerait des doublons BDD
  (scheme different -> URL differente -> pas de conflit UNIQUE).
- Trailing slash : ``/article/`` vs ``/article`` : differences cosmetiques
  sans difference semantique, mais cassent l'egalite stricte.

Le module applique une **normalisation conservative** : on force https,
on retire le trailing slash, on lowercase le scheme et le host. Cela
garantit que deux URLs semantiquement equivalentes sont vues comme
identiques par le pre-check, ce qui elimine :

1. Les faux negatifs (re-fetch inutile d'articles deja connus)
2. Les doublons BDD crees par les faux negatifs

La normalisation est appliquee des deux cotes :
- Sur les URLs chargees depuis la BDD (``load_known_urls``)
- Sur les URLs candidates (``url_is_known``)

Elle ne change pas l'URL inseree en BDD : les collecteurs continuent a
stocker l'URL brute (qui reste la cle UNIQUE). La normalisation est
purement une couche de comparaison.

Pourquoi un module partage plutot qu'une methode par collecteur ?
-----------------------------------------------------------------

- **DRY** : une seule query SQL, un seul code de conversion DSN asyncpg,
  un seul gestionnaire d'erreur pour le cas "BDD inaccessible".
- **Performance** : le set Python est immutable une fois charge, donc on
  peut le passer par reference a plusieurs collecteurs dans un pipeline
  orchestre (``retrain_pipeline.py collect``) sans requery.
- **Coherence** : les 4 collecteurs REST/JSON, les 2 scraping, et le file
  ingester appliquent la meme semantique "skip deja en BDD".

Strategie de gain par source
----------------------------

L'impact du pre-check depend du cout du fetch unitaire :

- **Dev.to** (gain ENORME) : le collecteur suit le pattern liste + detail.
  La liste retourne 30 articles en 1 requete, puis le detail demande 1
  requete par article. Skipper un article connu economise donc 1 HTTP
  complet (fetch + parsing) par match.
- **TechCrunch scraping** (gain ENORME) : RSS retourne les URLs, puis
  Playwright charge chaque page (Chromium, 5-10s par page). Skipper un
  article connu economise un fetch Playwright complet.
- **Static scraping B2.3** (gain ENORME) : sitemap donne les URLs, puis 1
  HTTP par article. Skip = economie HTTP.
- **Guardian / arXiv / Crossref** (gain MODESTE) : 1 requete API retourne
  un batch de 50-200 articles AVEC leur contenu. Le fetch est deja paye
  avant le pre-check, seul le parsing et l'upload MinIO sont evites.
  Utile surtout pour eviter la pollution des JSON bruts MinIO et garder
  le dataset historique propre (pas d'articles fantomes non refetchables).
- **File ingester** (gain FAIBLE) : lecture disque locale, bon marche,
  mais evite de pousser des duplicatas vers MinIO.

Usage type
----------

::

    from greentech.data.collectors.url_precheck import load_known_urls

    async def collect(self, ...):
        known_urls = await load_known_urls(self.source_name)
        for url in discovered_urls:
            if url in known_urls:
                # skip : evite fetch + parsing
                continue
            # ... fetch et parsing normal

"""

from __future__ import annotations

from loguru import logger

# ---------------------------------------------------------------------------
# Normalisation URL (comparaison tolerante)
# ---------------------------------------------------------------------------


def normalize_url(url: str | None) -> str:
    """Normalise une URL pour permettre une comparaison tolerante.

    Regles appliquees (idempotentes) :

    1. Trim whitespace
    2. Scheme force a ``https://`` (``http://`` -> ``https://``)
    3. Host lowercase : ``Example.COM`` -> ``example.com``
    4. Trailing slash retire : ``/foo/`` -> ``/foo``

    Ces regles couvrent les divergences observees en production :
    - ``http`` vs ``https`` entre ``file_ingester`` (Kaggle arXiv, https) et
      ``arxiv_collector`` API (http).
    - Variations de casse cote host (RFC 3986 dit le host est case-insensitive).
    - Trailing slash optionnel selon les conventions des CMS.

    On NE normalise PAS :
    - Le path au-dela du trailing slash (la casse path est significative
      sur certains serveurs).
    - Les query strings (certains CMS encodent l'identifiant de l'article
      via ``?p=123``, ne pas les perdre).
    - Les fragments (``#section``), qui sont toujours cote client.

    Args:
        url: URL a normaliser. ``None`` ou chaine vide retourne chaine vide.

    Returns:
        URL normalisee, ou chaine vide si l'entree etait vide.
    """
    if not url:
        return ""
    url = url.strip()
    if not url:
        return ""

    # Scheme -> https (insensible a la casse, ex: "HTTP://")
    lowered = url.lower()
    if lowered.startswith("http://"):
        url = "https://" + url[len("http://") :]
    elif lowered.startswith("https://"):
        url = "https://" + url[len("https://") :]

    # Lowercase du host uniquement (pas du path qui peut etre case-sensitive)
    # On split apres le scheme pour isoler host + path.
    if url.startswith("https://"):
        after_scheme = url[len("https://") :]
        # Host = jusqu'au premier "/"
        slash_idx = after_scheme.find("/")
        if slash_idx == -1:
            # Pas de path, tout est host
            host = after_scheme
            path = ""
        else:
            host = after_scheme[:slash_idx]
            path = after_scheme[slash_idx:]
        url = "https://" + host.lower() + path

    # Trailing slash : retire uniquement s'il y a un path non-vide apres le host
    # (on ne touche pas a "https://example.com/" qui est la racine).
    if url.count("/") > 3 and url.endswith("/"):
        url = url[:-1]

    return url


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


async def load_known_urls(source_name: str = "collector") -> set[str]:
    """Charge toutes les URLs deja presentes en table ``articles``.

    Utilise asyncpg directement (plutot que SQLAlchemy) pour minimiser
    l'overhead : une simple query indexee sur la colonne UNIQUE
    ``articles.url`` retourne en < 500 ms meme sur une table de 50 000
    articles.

    Cette fonction est safe a appeler meme si la BDD est indisponible :
    en cas d'erreur, elle retourne un set vide et logge un warning. Le
    collecteur continue alors sans filtrage, avec le garde-fou final
    ``ON CONFLICT (url) DO NOTHING`` cote SQL ingester comme derniere
    ligne de defense.

    Args:
        source_name: Nom de la source appelante (purement pour les logs,
            pour faciliter le debug multi-collecteurs).

    Returns:
        Set des URLs deja connues. Vide si la BDD est inaccessible.

    Example:
        >>> known = await load_known_urls("devto")
        >>> len(known)
        5412
        >>> "https://dev.to/foo/bar" in known
        True
    """
    try:
        import asyncpg

        from greentech.config import get_settings

        settings = get_settings()
        # SQLAlchemy utilise ``postgresql+asyncpg://`` ; asyncpg pur attend
        # ``postgresql://``. On convertit avant le connect().
        dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetch(
                "SELECT url FROM articles WHERE url IS NOT NULL",
            )
        finally:
            await conn.close()

        # Normalisation des URLs stockees : scheme force a https, trailing
        # slash retire, host lowercase. Voir ``normalize_url`` pour le detail
        # des regles. Sans cette normalisation, un re-run sur des sources
        # utilisant des schemes differents (ex: file_ingester/arxiv_collector
        # http vs https) re-fetcherait les articles deja connus.
        urls = {normalize_url(row["url"]) for row in rows if row["url"]}
        logger.info(
            f"[{source_name}] Pre-check BDD : {len(urls)} URLs deja connues (skip avant fetch)"
        )
        return urls
    except Exception as exc:
        logger.warning(
            f"[{source_name}] Pre-check BDD echec ({exc}) : "
            "on continue sans filtrage, le skip s'appliquera in fine "
            "via ON CONFLICT sur l'ingestion SQL"
        )
        return set()


def url_is_known(url: str | None, known_urls: set[str]) -> bool:
    """Teste si une URL candidate est deja connue via comparaison normalisee.

    Helper a appeler par chaque collecteur. Normalise l'URL candidate via
    ``normalize_url`` puis teste son appartenance au set ``known_urls``
    (qui doit etre le retour de ``load_known_urls``, deja normalise).

    Args:
        url: URL candidate a tester. ``None`` ou vide -> False.
        known_urls: Set retourne par ``load_known_urls()``, URLs normalisees.

    Returns:
        True si l'URL normalisee est dans le set, False sinon.

    Example:
        >>> known = await load_known_urls()
        >>> # Meme si l'URL candidate est "http://..." et que la BDD a "https://..."
        >>> url_is_known("http://arxiv.org/abs/1234.5678", known)
        True
    """
    if not url:
        return False
    return normalize_url(url) in known_urls


def coerce_bool(value: object) -> bool:
    """Convertit une valeur arbitraire en bool, tolerante aux strings CLI.

    Les collecteurs recoivent parfois des flags via CLI (``-a``, ``--arg``)
    ou des variables d'environnement, ou sont pilotes par un orchestrateur
    Python. Cette fonction normalise l'ensemble en bool Python. Les
    strings ``"false"``, ``"0"``, ``"no"``, ``"off"`` (casse insensible) et
    la chaine vide sont interpretees comme False ; tout le reste comme
    True.

    Args:
        value: Valeur a convertir (str, bool, int, float, ...).

    Returns:
        Bool equivalent.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"false", "0", "no", "off", ""}
