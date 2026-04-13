"""Verification LLM des articles candidats (etage 2 du pipeline de classification).

Ce script lit en base tous les articles marques `CANDIDATE` par le pre-filtre
mots-cles (`modele_classification = "keyword_filter"` et `est_green_it IS NULL`),
les soumet au LLM judge (`Qwen/Qwen2.5-7B-Instruct` via HF Serverless) et ecrit
le verdict final en base :

- `est_green_it` : True ou False selon la decision du LLM
- `modele_classification` : "keyword_filter+qwen_llm_judge"
- `score_confiance` : confiance rapportee par le LLM

Usage
-----

    uv run python scripts/classify_candidates.py

Le script respecte un delai de 0.5s entre chaque appel pour rester dans le
fair-use HF Serverless. Avec un corpus de 500 candidats, le traitement prend
environ 15 a 20 minutes.

En cas d'echec API ponctuel, l'article reste en l'etat (est_green_it=NULL)
et sera re-tente au prochain passage du script.

"""

from __future__ import annotations

import asyncio

from loguru import logger
from sqlalchemy import and_, select, update

from greentech.ai.services.classifier_llm import (
    ClassifierVerdict,
    verify_green_it_batch,
)
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

# Marqueur final pose en base apres verification LLM reussie
FINAL_MODELE_TAG = "keyword_filter+qwen_llm_judge"
# Marqueur produit par le pre-filtre : articles en attente de LLM
PREFILTER_MODELE_TAG = "keyword_filter"


async def fetch_candidates(limit: int | None = None) -> list[tuple[int, str, str]]:
    """Recupere les articles candidats en attente de verification LLM.

    Args:
        limit: Nombre maximum d'articles a recuperer (None = tous).

    Returns:
        Liste de tuples (id_article, titre, contenu).
    """
    async with async_session_factory() as session:
        stmt = (
            select(Article.id_article, Article.titre, Article.contenu)
            .where(
                and_(
                    Article.modele_classification == PREFILTER_MODELE_TAG,
                    Article.est_green_it.is_(None),
                    Article.contenu.isnot(None),
                )
            )
            .order_by(Article.id_article)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return [(row[0], row[1], row[2] or "") for row in result.all()]


async def apply_verdicts(verdicts: dict[int, ClassifierVerdict]) -> tuple[int, int]:
    """Ecrit les verdicts du LLM en base de donnees.

    Args:
        verdicts: Dictionnaire {id_article: ClassifierVerdict}.

    Returns:
        Tuple (nb_succes, nb_echecs). Les echecs ne modifient pas la DB
        (l'article reste en attente pour un prochain run).
    """
    succes = 0
    echecs = 0
    async with async_session_factory() as session:
        for id_article, verdict in verdicts.items():
            if not verdict.succes or verdict.est_green_it is None:
                echecs += 1
                continue

            await session.execute(
                update(Article)
                .where(Article.id_article == id_article)
                .values(
                    est_green_it=verdict.est_green_it,
                    modele_classification=FINAL_MODELE_TAG,
                    score_confiance=verdict.confiance,
                )
            )
            succes += 1

        await session.commit()
    return succes, echecs


async def classify_all_candidates(
    *,
    limit: int | None = None,
    delay_seconds: float = 1.0,
) -> dict[str, int]:
    """Orchestre l'etage 2 : LLM judge sur tous les candidats.

    Args:
        limit: Plafond optionnel pour un run partiel (debug ou test).
        delay_seconds: Pause entre deux appels LLM pour respecter le fair-use HF.

    Returns:
        Statistiques {total, green_it, non_green_it, echecs}.
    """
    candidates = await fetch_candidates(limit=limit)
    stats = {"total": len(candidates), "green_it": 0, "non_green_it": 0, "echecs": 0}

    if not candidates:
        logger.info("Aucun candidat a verifier - etage 2 inactif")
        return stats

    logger.info(f"{len(candidates)} candidats a verifier via LLM judge")
    verdicts = await verify_green_it_batch(candidates, delay_seconds=delay_seconds)

    for verdict in verdicts.values():
        if not verdict.succes or verdict.est_green_it is None:
            stats["echecs"] += 1
        elif verdict.est_green_it:
            stats["green_it"] += 1
        else:
            stats["non_green_it"] += 1

    succes_ecrit, echecs_ecrit = await apply_verdicts(verdicts)
    logger.info(
        f"Verdicts appliques : {succes_ecrit} ecrits, {echecs_ecrit} echoues"
    )
    logger.info("Bilan etage 2 :")
    logger.info(f"  Total candidats : {stats['total']}")
    logger.info(f"  Green IT        : {stats['green_it']}")
    logger.info(f"  Non Green IT    : {stats['non_green_it']}")
    logger.info(f"  Echecs (retry)  : {stats['echecs']}")
    return stats


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(classify_all_candidates())
