"""Auto-correction des articles borderline pour les sources 100 % Green IT (B2.10).

Ce script applique une heuristique source-pure : pour les sources verifiees en
phase B2.1 comme etant entierement specialisees Green IT (GreenIT.fr, Green
Software Foundation, Sustainable Web Design, Climate Action Tech), les articles
borderline (score_confiance entre 0.3 et 0.7) classes Non Green IT par le LLM
judge sont automatiquement reclasses Green IT. Cette decision repose sur :

1. La verification editoriale prealable B2.1 : ces 4 sites ont ete audites
   manuellement et confirmes comme 100 % thematiques Green IT.
2. L'observation empirique du premier batch d'annotation manuelle (B2.10) :
   sur 30 articles GreenIT.fr annotes a la main, 29 (96.7 %) sont en fait
   Green IT. Le LLM judge presente donc un biais systematique sur ces sources,
   probablement du a la dominance de l'anglais dans son corpus d'entrainement.

La traceabilite est preservee :
- ``annotation_source = 'auto_source_pure'`` (distinct de 'manual' et 'llm_judge')
- ``annotated_at = now()`` (UTC)
- ``annotated_by = 'system_heuristic_b2.10'``
- ``score_confiance`` et ``raison_llm_judge`` sont conserves intacts

Les articles deja annotes manuellement (``annotation_source = 'manual'``) ne
sont JAMAIS ecrases - leur decision humaine prime sur l'heuristique.

Usage
-----

Preview (dry-run, aucune modification BDD) ::

    uv run python scripts/auto_correct_pure_sources.py --dry-run

Application reelle ::

    uv run python scripts/auto_correct_pure_sources.py

Rollback (annule toutes les corrections automatiques) ::

    uv run python scripts/auto_correct_pure_sources.py --rollback

"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import and_, func, or_, select, update

from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

# Sources 100 % Green IT verifiees lors de la phase B2.1 (avril 2026). Chaque
# entree correspond au champ ``sources.nom`` insere par les migrations 002 et 003.
PURE_GREEN_IT_SOURCES: tuple[str, ...] = (
    "GreenIT.fr",
    "Green Software Foundation",
    "Sustainable Web Design",
    "Climate Action Tech",
)

# Fenetre de score consideree borderline (alignee sur B2.10).
SCORE_MIN = 0.3
SCORE_MAX = 0.7

# Marqueurs poses en base par cette correction automatique.
ANNOTATION_SOURCE_TAG = "auto_source_pure"
ANNOTATED_BY_TAG = "system_heuristic_b2.10"


async def _get_pure_source_ids(session) -> dict[str, int]:
    """Resout les noms de sources vers leurs ids en base.

    Returns:
        Dictionnaire {nom_source: id_source} pour les sources trouvees.
    """
    stmt = select(Source.id_source, Source.nom).where(Source.nom.in_(PURE_GREEN_IT_SOURCES))
    result = await session.execute(stmt)
    return {row.nom: row.id_source for row in result.all()}


async def _preview_corrections(session, source_ids: dict[str, int]) -> dict[str, int]:
    """Compte les articles eligibles a la correction, par source.

    Un article est eligible si :
    - Il appartient a une source pure
    - Son score_confiance est dans la fenetre borderline [0.3, 0.7]
    - Son annotation_source n'est PAS 'manual' (on respecte les annotations humaines)
    """
    stats: dict[str, int] = {}
    for nom, id_source in source_ids.items():
        stmt = (
            select(func.count(Article.id_article))
            .where(
                and_(
                    Article.id_source == id_source,
                    Article.score_confiance >= SCORE_MIN,
                    Article.score_confiance <= SCORE_MAX,
                    or_(
                        Article.annotation_source != "manual",
                        Article.annotation_source.is_(None),
                    ),
                )
            )
        )
        result = await session.execute(stmt)
        stats[nom] = int(result.scalar_one())
    return stats


async def _apply_corrections(session, source_ids: dict[str, int]) -> int:
    """Applique l'UPDATE en masse pour reclasser les borderline en Green IT.

    Returns:
        Nombre de lignes effectivement modifiees.
    """
    now_utc = datetime.now(UTC)
    stmt = (
        update(Article)
        .where(
            and_(
                Article.id_source.in_(list(source_ids.values())),
                Article.score_confiance >= SCORE_MIN,
                Article.score_confiance <= SCORE_MAX,
                or_(
                    Article.annotation_source != "manual",
                    Article.annotation_source.is_(None),
                ),
            )
        )
        .values(
            est_green_it=True,
            annotation_source=ANNOTATION_SOURCE_TAG,
            annotated_at=now_utc,
            annotated_by=ANNOTATED_BY_TAG,
        )
    )
    result = await session.execute(stmt)
    return result.rowcount or 0


async def _rollback_corrections(session) -> int:
    """Annule toutes les corrections automatiques posees par ce script.

    Restaure ``est_green_it = False`` et ``annotation_source = 'llm_judge'``
    pour les articles marques ``annotation_source = 'auto_source_pure'``.

    Returns:
        Nombre de lignes restaurees.
    """
    stmt = (
        update(Article)
        .where(Article.annotation_source == ANNOTATION_SOURCE_TAG)
        .values(
            est_green_it=False,
            annotation_source="llm_judge",
            annotated_at=None,
            annotated_by=None,
        )
    )
    result = await session.execute(stmt)
    return result.rowcount or 0


async def run(dry_run: bool, rollback: bool) -> None:
    """Point d'entree principal du script.

    Args:
        dry_run: Si True, affiche les stats sans modifier la BDD.
        rollback: Si True, annule les corrections automatiques precedentes.
    """
    if rollback:
        logger.warning("Mode ROLLBACK : annulation des corrections auto_source_pure")
        async with async_session_factory() as session:
            n_restored = await _rollback_corrections(session)
            await session.commit()
            logger.success(f"{n_restored} articles restaures (est_green_it=False, source=llm_judge)")
        return

    async with async_session_factory() as session:
        source_ids = await _get_pure_source_ids(session)
        missing = set(PURE_GREEN_IT_SOURCES) - set(source_ids.keys())
        if missing:
            logger.error(f"Sources introuvables en BDD : {missing}")
            logger.error("Verifie que les migrations 002 et 003 ont bien ete appliquees.")
            return

        stats = await _preview_corrections(session, source_ids)
        total = sum(stats.values())

        logger.info("=" * 70)
        logger.info("Auto-correction des sources 100 % Green IT (heuristique B2.10)")
        logger.info("=" * 70)
        for nom, count in stats.items():
            logger.info(f"  {nom:35s} {count:>5d} articles a corriger")
        logger.info(f"  {'TOTAL':35s} {total:>5d} articles")
        logger.info("=" * 70)

        if dry_run:
            logger.warning("Mode DRY-RUN : aucune modification appliquee")
            logger.info("Relance sans --dry-run pour appliquer les corrections")
            return

        if total == 0:
            logger.info("Rien a faire : aucun article borderline restant sur les sources pures")
            return

        n_updated = await _apply_corrections(session, source_ids)
        await session.commit()
        logger.success(f"{n_updated} articles reclasses en Green IT")
        logger.info(
            f"Traceabilite : annotation_source='{ANNOTATION_SOURCE_TAG}', "
            f"annotated_by='{ANNOTATED_BY_TAG}'"
        )
        logger.info(
            "Pour annuler en cas de probleme : "
            "uv run python scripts/auto_correct_pure_sources.py --rollback"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-correction des borderline pour les sources 100 % Green IT (B2.10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Aper�u sans modification de la BDD (compte les articles eligibles).",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Annule toutes les corrections precedemment posees par ce script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run and args.rollback:
        logger.error("--dry-run et --rollback sont mutuellement exclusifs.")
        raise SystemExit(1)
    asyncio.run(run(dry_run=args.dry_run, rollback=args.rollback))
