"""Application des classifications produites par agents LLM sur les borderline.

Ce script generique applique les decisions Green IT / Non Green IT produites par
des agents LLM (audit multi-agents) sur n'importe quelle source. Il lit un
fichier JSON de classifications et applique les decisions en BDD avec une
traceabilite complete.

Format attendu du fichier JSON :

.. code-block:: json

    {
      "_meta": {
        "annotation_source_tag": "auto_agent_audit",
        "annotated_by_tag": "multi_agent_classifier_b2.10",
        ...
      },
      "classifications": [
        {"id_article": 12345, "decision": "g", "raison": "..."},
        {"id_article": 67890, "decision": "n", "raison": "..."}
      ]
    }

Les decisions ``"g"`` et ``"n"`` se traduisent respectivement en
``est_green_it=True`` et ``est_green_it=False``.

Usage
-----

Preview (dry-run) ::

    uv run python scripts/apply_agent_classifications.py audit_crossref/classifications.json --dry-run

Application reelle ::

    uv run python scripts/apply_agent_classifications.py audit_crossref/classifications.json

Rollback (re-passe en NULL et annotation_source=NULL) ::

    uv run python scripts/apply_agent_classifications.py audit_crossref/classifications.json --rollback

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

DEFAULT_ANNOTATION_SOURCE = "auto_agent_audit"
DEFAULT_ANNOTATED_BY = "multi_agent_classifier_b2.10"


def _load_classifications(path: Path) -> tuple[list[dict], dict]:
    """Charge le fichier JSON et separe meta + classifications."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("_meta", {})
    classifications = payload["classifications"]
    return classifications, meta


async def _apply(
    classifications: list[dict],
    annotation_source_tag: str,
    annotated_by_tag: str,
) -> tuple[int, int]:
    """Applique les decisions en BDD. Retourne (n_green, n_non_green)."""
    now_utc = datetime.now(UTC)
    green_ids = [c["id_article"] for c in classifications if c["decision"] == "g"]
    non_green_ids = [c["id_article"] for c in classifications if c["decision"] == "n"]

    async with async_session_factory() as session:
        n_green = 0
        if green_ids:
            stmt = (
                update(Article)
                .where(Article.id_article.in_(green_ids))
                .values(
                    est_green_it=True,
                    annotation_source=annotation_source_tag,
                    annotated_at=now_utc,
                    annotated_by=annotated_by_tag,
                )
            )
            result = await session.execute(stmt)
            n_green = result.rowcount or 0

        n_non_green = 0
        if non_green_ids:
            stmt = (
                update(Article)
                .where(Article.id_article.in_(non_green_ids))
                .values(
                    est_green_it=False,
                    annotation_source=annotation_source_tag,
                    annotated_at=now_utc,
                    annotated_by=annotated_by_tag,
                )
            )
            result = await session.execute(stmt)
            n_non_green = result.rowcount or 0

        await session.commit()
        return n_green, n_non_green


async def _rollback(annotation_source_tag: str) -> int:
    """Annule toutes les decisions posees avec ce tag."""
    async with async_session_factory() as session:
        stmt = (
            update(Article)
            .where(Article.annotation_source == annotation_source_tag)
            .values(
                est_green_it=None,
                annotation_source=None,
                annotated_at=None,
                annotated_by=None,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0


async def run(json_path: Path, dry_run: bool, rollback: bool) -> None:
    classifications, meta = _load_classifications(json_path)
    annotation_source_tag = meta.get("annotation_source_tag", DEFAULT_ANNOTATION_SOURCE)
    annotated_by_tag = meta.get("annotated_by_tag", DEFAULT_ANNOTATED_BY)

    if rollback:
        logger.warning(f"Mode ROLLBACK : annulation des decisions taggees '{annotation_source_tag}'")
        n = await _rollback(annotation_source_tag)
        logger.success(f"{n} articles restaures (est_green_it=NULL, annotation_source=NULL)")
        return

    n_green = sum(1 for c in classifications if c["decision"] == "g")
    n_non_green = sum(1 for c in classifications if c["decision"] == "n")

    logger.info("=" * 70)
    logger.info("Application des classifications multi-agents")
    logger.info("=" * 70)
    logger.info(f"  Fichier              : {json_path.name}")
    logger.info(f"  Source du dataset    : {meta.get('source', 'inconnue')}")
    logger.info(f"  Methode              : {meta.get('audit_method', 'inconnue')}")
    logger.info(f"  Total classifies     : {len(classifications)}")
    logger.info(f"  Green IT             : {n_green}")
    logger.info(f"  Non Green IT         : {n_non_green}")
    logger.info(f"  Tag annotation_source: '{annotation_source_tag}'")
    logger.info(f"  Tag annotated_by     : '{annotated_by_tag}'")
    logger.info("=" * 70)

    if dry_run:
        logger.warning("Mode DRY-RUN : aucune modification BDD")
        return

    applied_green, applied_non_green = await _apply(
        classifications, annotation_source_tag, annotated_by_tag
    )
    logger.success(
        f"Applique : {applied_green} Green IT + {applied_non_green} Non Green IT "
        f"= {applied_green + applied_non_green} articles modifies"
    )
    logger.info(
        f"Pour annuler : uv run python {Path(__file__).name} {json_path} --rollback"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Applique les classifications produites par des agents LLM sur les articles."
    )
    parser.add_argument("json_path", type=Path, help="Chemin du fichier JSON de classifications.")
    parser.add_argument("--dry-run", action="store_true", help="Apercu sans modification BDD.")
    parser.add_argument("--rollback", action="store_true", help="Annule les decisions taggees.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run and args.rollback:
        logger.error("--dry-run et --rollback sont mutuellement exclusifs.")
        raise SystemExit(1)
    asyncio.run(run(json_path=args.json_path, dry_run=args.dry_run, rollback=args.rollback))
