"""Peuple une base PostgreSQL distante (ex. Render) avec un echantillon
representatif d'articles, afin de disposer d'une demonstration non vide.

Le script LIT la base locale du projet (source de verite, deja peuplee par le
pipeline de collecte) et ECRIT un echantillon dans la base cible designee par la
variable d'environnement ``RENDER_DATABASE_URL``. Il est idempotent : les
articles deja presents (meme ``url``) et les sources deja presentes (meme
``nom``) sont ignores, donc il peut etre relance sans creer de doublons.

Prerequis
---------
- La base locale doit etre demarree et peuplee (``docker compose up -d postgres``
  puis le pipeline de collecte). Le script lit ``settings.database_url``.
- La base cible doit exister. Son schema est cree par l'API au demarrage
  (``create_all``) ; le script reapplique ``create_all`` par securite, ce qui
  est sans effet si les tables existent deja.

Recuperation de l'URL cible
---------------------------
Dans le dashboard Render : service ``greentech-db`` -> onglet ``Connect`` ->
``External Database URL`` (format ``postgresql://user:mdp@host/db``).

Usage (PowerShell)
------------------
    $env:RENDER_DATABASE_URL = "postgresql://user:mdp@host/db"
    uv run python scripts/seed_render_db.py --limit 400 --green-ratio 0.2

Usage (bash)
-----------
    RENDER_DATABASE_URL="postgresql://user:mdp@host/db" \
        uv run python scripts/seed_render_db.py --limit 400
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from greentech.config import get_settings
from greentech.data.storage.models import Article, Base, Source

# Colonnes d'article copiees telles quelles. On exclut la cle primaire
# (id_article), la cle etrangere (id_source, re-mappee par nom de source), l'uuid
# (regenere cote cible) et les timestamps a server_default (date_creation /
# date_modification, positionnes par la base cible).
_ARTICLE_COPY_COLS: tuple[str, ...] = (
    "titre",
    "url",
    "contenu",
    "resume",
    "resume_ecologique",
    "auteur",
    "date_publication",
    "langue",
    "est_green_it",
    "score_confiance",
    "modele_classification",
    "raison_llm_judge",
    "annotation_source",
    "annotated_at",
    "annotated_by",
    "chemin_donnees_brutes",
    "date_analyse",
)


def _to_async_url(raw: str) -> str:
    """Normalise une URL PostgreSQL pour le driver asyncpg.

    Render fournit une URL ``postgres://`` ou ``postgresql://``, parfois suffixee
    de ``?sslmode=require``. asyncpg ne comprend pas ``sslmode`` dans le DSN (le
    SSL passe par ``connect_args``), et SQLAlchemy attend le prefixe
    ``postgresql+asyncpg://``. On corrige les deux aspects.

    Args:
        raw: URL telle que fournie par Render.

    Returns:
        URL compatible ``create_async_engine`` (driver asyncpg).
    """
    url = raw.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    return re.sub(r"[?&]sslmode=[^&]+", "", url)


async def _fetch_sample(engine: AsyncEngine, *, limit: int, green_ratio: float) -> list[dict]:
    """Lit un echantillon equilibre d'articles classifies depuis la base source.

    Ne retient que les articles classifies (``est_green_it`` non nul) disposant
    d'un ``resume``, pour que le tableau de bord affiche des cartes et une liste
    « Derniers articles » realistes.

    Args:
        engine: Moteur asynchrone connecte a la base source (locale).
        limit: Nombre total d'articles vises.
        green_ratio: Proportion d'articles Green IT dans l'echantillon.

    Returns:
        Liste de dictionnaires (une entree par article, colonnes d'article +
        metadonnees de source prefixees ``src_``).
    """
    n_green = max(0, int(limit * green_ratio))
    n_other = max(0, limit - n_green)
    art = Article.__table__
    src = Source.__table__
    select_cols = [art.c[name] for name in _ARTICLE_COPY_COLS] + [
        src.c.nom.label("src_nom"),
        src.c.type.label("src_type"),
        src.c.url_base.label("src_url_base"),
        src.c.description.label("src_description"),
    ]
    base = (
        select(*select_cols)
        .select_from(art.outerjoin(src, art.c.id_source == src.c.id_source))
        .where(art.c.resume.isnot(None))
    )

    async with engine.connect() as conn:
        green = (
            (await conn.execute(base.where(art.c.est_green_it.is_(True)).limit(n_green)))
            .mappings()
            .all()
        )
        other = (
            (await conn.execute(base.where(art.c.est_green_it.is_(False)).limit(n_other)))
            .mappings()
            .all()
        )

    logger.info(f"Echantillon source : {len(green)} Green IT + {len(other)} non Green IT")
    return [dict(row) for row in (*green, *other)]


async def _seed_target(engine: AsyncEngine, sample: list[dict]) -> tuple[int, int]:
    """Ecrit les sources puis les articles dans la base cible (idempotent).

    Args:
        engine: Moteur asynchrone connecte a la base cible (Render).
        sample: Echantillon retourne par :func:`_fetch_sample`.

    Returns:
        Tuple ``(nombre_de_sources, articles_inseres)``.
    """
    art = Article.__table__
    src = Source.__table__

    # 1) Schema (sans effet si les tables existent deja).
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2) Sources distinctes referencees par l'echantillon.
    sources: dict[str, dict] = {}
    for row in sample:
        nom = row.get("src_nom")
        if nom and nom not in sources:
            sources[nom] = {
                "nom": nom,
                "type": row.get("src_type") or "inconnu",
                "url_base": row.get("src_url_base"),
                "description": row.get("src_description"),
            }

    async with engine.begin() as conn:
        if sources:
            await conn.execute(
                pg_insert(src)
                .values(list(sources.values()))
                .on_conflict_do_nothing(index_elements=["nom"])
            )
        rows = (await conn.execute(select(src.c.id_source, src.c.nom))).all()
        nom_to_id = {nom: id_source for id_source, nom in rows}

    # 3) Articles (id_source re-mappe par nom), par lots de 100, idempotents sur url.
    payload = [
        {
            **{col: row[col] for col in _ARTICLE_COPY_COLS},
            "id_source": nom_to_id.get(row.get("src_nom")),
        }
        for row in sample
    ]

    inserted = 0
    async with engine.begin() as conn:
        for start in range(0, len(payload), 100):
            chunk = payload[start : start + 100]
            result = await conn.execute(
                pg_insert(art).values(chunk).on_conflict_do_nothing(index_elements=["url"])
            )
            inserted += result.rowcount or 0

    return len(sources), inserted


async def _run(limit: int, green_ratio: float) -> int:
    """Orchestre la lecture source -> ecriture cible. Retourne un code de sortie."""
    target_raw = os.environ.get("RENDER_DATABASE_URL", "").strip()
    if not target_raw:
        logger.error(
            "Variable RENDER_DATABASE_URL absente. Recupere l'External Database "
            "URL dans Render (greentech-db -> Connect), exporte-la, puis relance."
        )
        return 1

    source_engine = create_async_engine(get_settings().database_url, pool_pre_ping=True)
    target_engine = create_async_engine(
        _to_async_url(target_raw),
        pool_pre_ping=True,
        # Render impose le SSL ; "require" chiffre sans verification stricte du
        # certificat (equivalent libpq sslmode=require), robuste cote client.
        connect_args={"ssl": "require"},
    )
    try:
        sample = await _fetch_sample(source_engine, limit=limit, green_ratio=green_ratio)
        if not sample:
            logger.error(
                "Aucun article classifie avec resume dans la base locale. "
                "Verifie que la base locale est demarree et peuplee."
            )
            return 1
        n_sources, n_inserted = await _seed_target(target_engine, sample)
    finally:
        await source_engine.dispose()
        await target_engine.dispose()

    logger.success(
        f"Seed termine : {n_inserted} article(s) insere(s), {n_sources} source(s) "
        "cote cible (doublons url/nom ignores)."
    )
    return 0


def main() -> None:
    """Point d'entree CLI."""
    parser = argparse.ArgumentParser(
        description="Copie un echantillon d'articles de la base locale vers la base Render."
    )
    parser.add_argument(
        "--limit", type=int, default=400, help="Nombre d'articles a copier (defaut : 400)."
    )
    parser.add_argument(
        "--green-ratio",
        type=float,
        default=0.2,
        help="Proportion d'articles Green IT dans l'echantillon (defaut : 0.2).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.limit, args.green_ratio)))


if __name__ == "__main__":
    main()
