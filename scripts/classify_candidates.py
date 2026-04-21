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

Resilience aux interruptions (kill-safe)
----------------------------------------

Le script traite et **commit les verdicts par batchs** de ``BATCH_COMMIT_SIZE``
articles. Sur une interruption (kill, coupure, shutdown), au maximum
``BATCH_COMMIT_SIZE`` verdicts calcules mais non persistes sont perdus
(~10 min de travail a 5 articles/min cote Qwen local). Le script est
idempotent : un re-run relit uniquement les articles encore NULL.

Avec un corpus de 500 candidats, le traitement prend environ 15 a 20
minutes via HF Serverless, 1 a 2 heures via Qwen local (GPU AMD ROCm).

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

# Taille des batchs de commit : trade-off entre pertes sur kill et
# overhead DB. 50 articles = ~5-10 min de travail LLM local au pire,
# acceptable comme perte maximale sur interruption.
BATCH_COMMIT_SIZE = 50


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
    batch_size: int = BATCH_COMMIT_SIZE,
) -> dict[str, int]:
    """Orchestre l'etage 2 : LLM judge sur tous les candidats.

    Traite les candidats par batchs de ``batch_size`` pour limiter la
    perte sur interruption. Apres chaque batch, les verdicts sont
    persistes (``apply_verdicts`` -> commit) et le prochain batch est
    lance. Sur kill/shutdown entre deux commits, on perd au maximum
    ``batch_size`` verdicts deja calcules mais non persistes.

    Au re-run, ``fetch_candidates`` ne renvoie que les articles encore
    ``est_green_it IS NULL``, donc le script reprend la ou il s'etait
    arrete sans re-traiter les articles deja decides.

    Args:
        limit: Plafond optionnel pour un run partiel (debug ou test).
        delay_seconds: Pause entre deux appels LLM pour respecter le
            fair-use HF. Ignoree en mode local (GPU saturation naturelle).
        batch_size: Nombre d'articles par batch de commit. Plus petit =
            moins de perte sur kill, plus gros = moins d'overhead DB.

    Returns:
        Statistiques {total, green_it, non_green_it, echecs}.
    """
    candidates = await fetch_candidates(limit=limit)
    stats = {"total": len(candidates), "green_it": 0, "non_green_it": 0, "echecs": 0}

    if not candidates:
        logger.info("Aucun candidat a verifier - etage 2 inactif")
        return stats

    logger.info(
        f"{len(candidates)} candidats a verifier via LLM judge "
        f"(batch_size={batch_size}, commit apres chaque batch)"
    )

    # Decoupage en batchs. Chaque batch : LLM judge -> apply_verdicts ->
    # commit. Kill-safe : perte max = batch_size verdicts.
    total = len(candidates)
    total_succes_ecrit = 0
    total_echecs_ecrit = 0
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = candidates[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"[Batch {batch_num}/{total_batches}] "
            f"Traitement articles {batch_start + 1}..{batch_end} ({len(batch)} candidats)"
        )

        verdicts = await verify_green_it_batch(batch, delay_seconds=delay_seconds)

        # Mise a jour stats globales pour ce batch
        batch_green = 0
        batch_non_green = 0
        batch_echecs = 0
        for verdict in verdicts.values():
            if not verdict.succes or verdict.est_green_it is None:
                batch_echecs += 1
                stats["echecs"] += 1
            elif verdict.est_green_it:
                batch_green += 1
                stats["green_it"] += 1
            else:
                batch_non_green += 1
                stats["non_green_it"] += 1

        # Commit immediat du batch : kill-safe.
        succes_ecrit, echecs_ecrit = await apply_verdicts(verdicts)
        total_succes_ecrit += succes_ecrit
        total_echecs_ecrit += echecs_ecrit

        logger.info(
            f"[Batch {batch_num}/{total_batches}] "
            f"Commit : {succes_ecrit} ecrits, {echecs_ecrit} echoues | "
            f"batch (green={batch_green}, non_green={batch_non_green}, echecs={batch_echecs}) | "
            f"cumul ({batch_end}/{total} articles, green={stats['green_it']}, "
            f"non_green={stats['non_green_it']}, echecs={stats['echecs']})"
        )

    logger.info(f"Verdicts appliques : {total_succes_ecrit} ecrits, {total_echecs_ecrit} echoues")
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
