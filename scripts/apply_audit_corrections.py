"""Application des corrections issues de l'audit multi-agents GreenIT.fr (B2.10).

Ce script lit la liste des faux positifs identifies par l'audit multi-agents
(``audit_greenit/false_positives.json``) et repasse ces articles en Non Green IT
en base de donnees. Cas d'usage : un article a ete auto-classifie Green IT par
``auto_correct_pure_sources.py`` (heuristique source-pure sur GreenIT.fr) mais
relit a posteriori, il ne porte pas d'angle IT/numerique substantiel.

Tracabilite des corrections :

- ``est_green_it = False``
- ``annotation_source = 'auto_audit'`` (distinct des autres marqueurs)
- ``annotated_at = now()`` (UTC)
- ``annotated_by = 'multi_agent_audit_b2.10'``
- ``score_confiance`` et ``raison_llm_judge`` restent intacts (audit traceable)

Le fichier d'audit est immuable et versionne : il documente quels articles ont
ete corriges, quand, et pourquoi. Le script est idempotent et reversible
(via ``--rollback``).

Usage
-----

Preview (dry-run) ::

    uv run python scripts/apply_audit_corrections.py --dry-run

Application reelle ::

    uv run python scripts/apply_audit_corrections.py

Rollback (re-classe les articles en Green IT auto_source_pure) ::

    uv run python scripts/apply_audit_corrections.py --rollback

"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from sqlalchemy import update

from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

AUDIT_FILE = Path(__file__).resolve().parent.parent / "audit_greenit" / "false_positives.json"

ANNOTATION_SOURCE_TAG = "auto_audit"
ANNOTATED_BY_TAG = "multi_agent_audit_b2.10"


def _load_audit() -> list[dict]:
    """Charge la liste des faux positifs depuis le fichier d'audit."""
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(f"Fichier d'audit introuvable : {AUDIT_FILE}")
    with AUDIT_FILE.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["false_positives"]


async def _apply_corrections(ids: list[int]) -> int:
    """Repasse les articles cibles en Non Green IT."""
    now_utc = datetime.now(UTC)
    async with async_session_factory() as session:
        stmt = (
            update(Article)
            .where(Article.id_article.in_(ids))
            .values(
                est_green_it=False,
                annotation_source=ANNOTATION_SOURCE_TAG,
                annotated_at=now_utc,
                annotated_by=ANNOTATED_BY_TAG,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0


async def _rollback() -> int:
    """Restaure les articles ``auto_audit`` en Green IT auto_source_pure."""
    async with async_session_factory() as session:
        stmt = (
            update(Article)
            .where(Article.annotation_source == ANNOTATION_SOURCE_TAG)
            .values(
                est_green_it=True,
                annotation_source="auto_source_pure",
                annotated_at=None,
                annotated_by=None,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0


async def run(dry_run: bool, rollback: bool) -> None:
    if rollback:
        logger.warning("Mode ROLLBACK : restauration des articles audites en Green IT")
        n = await _rollback()
        logger.success(f"{n} articles restaures en annotation_source='auto_source_pure'")
        return

    audit = _load_audit()
    ids = [item["id_article"] for item in audit]

    logger.info("=" * 70)
    logger.info("Application des corrections d'audit multi-agents GreenIT.fr")
    logger.info("=" * 70)
    logger.info(f"  Fichier d'audit : {AUDIT_FILE.relative_to(AUDIT_FILE.parents[1])}")
    logger.info(f"  Faux positifs a corriger : {len(ids)}")
    logger.info("=" * 70)

    # Apercu : 5 premiers et 5 derniers articles concernes
    for item in audit[:5]:
        logger.info(f"  - id={item['id_article']} : {item['titre'][:70]}")
    if len(audit) > 10:
        logger.info(f"  ... ({len(audit) - 10} autres) ...")
    for item in audit[-5:]:
        logger.info(f"  - id={item['id_article']} : {item['titre'][:70]}")

    if dry_run:
        logger.warning("Mode DRY-RUN : aucune modification BDD")
        return

    n_updated = await _apply_corrections(ids)
    logger.success(f"{n_updated} articles repasses en Non Green IT")
    logger.info(
        f"Tracabilite : annotation_source='{ANNOTATION_SOURCE_TAG}', "
        f"annotated_by='{ANNOTATED_BY_TAG}'"
    )
    logger.info(
        "Pour annuler : uv run python scripts/apply_audit_corrections.py --rollback"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Application des corrections issues de l'audit multi-agents GreenIT.fr (B2.10)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Apercu sans modification.")
    parser.add_argument("--rollback", action="store_true", help="Restaure les articles audites.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run and args.rollback:
        logger.error("--dry-run et --rollback sont mutuellement exclusifs.")
        raise SystemExit(1)
    asyncio.run(run(dry_run=args.dry_run, rollback=args.rollback))
