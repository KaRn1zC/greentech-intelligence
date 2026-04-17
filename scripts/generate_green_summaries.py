"""Genere en batch les resumes ecologiques pour les articles Green IT.

Ce script peuple la colonne ``articles.resume_ecologique`` en appelant le
LLM instructif via le prompt specialise Green IT defini dans
``summarizer.py``. Contrairement au resume de classification, ce resume
n'est genere **que** pour les articles confirmes Green IT
(``est_green_it = True``). Il est affiche dans la page detail de l'UI.

Utilisation
-----------

Lancer la generation complete (ecrase les resumes ecologiques existants) ::

    uv run python scripts/generate_green_summaries.py --force

Reprendre un run interrompu sans ecraser les resumes deja produits ::

    uv run python scripts/generate_green_summaries.py

Tester sur un echantillon limite ::

    uv run python scripts/generate_green_summaries.py --limit 10 --force

Le script est idempotent par defaut : sans ``--force``, il saute les
articles dont le ``resume_ecologique`` est deja renseigne.

"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from greentech.ai.services.summarizer import summarize_green_only
from greentech.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande.

    Returns:
        Namespace avec les champs ``force``, ``limit`` et ``delay``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genere les resumes ecologiques (colonne articles.resume_ecologique) "
            "pour les articles confirmes Green IT via le LLM Qwen."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenere egalement les resumes deja presents (defaut : non).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Plafond sur le nombre d'articles Green IT a traiter.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Pause en secondes entre deux articles (defaut : 0.3 s).",
    )
    return parser.parse_args()


async def main() -> None:
    """Execute le batch green et affiche les statistiques finales."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  GENERATION BATCH DES RESUMES ECOLOGIQUES (Green IT)")
    logger.info("=" * 70)
    logger.info(f"  force : {args.force}")
    logger.info(f"  limit : {args.limit if args.limit is not None else 'aucune (integralite)'}")
    logger.info(f"  delay : {args.delay}s entre articles")
    logger.info("=" * 70)

    stats = await summarize_green_only(
        limit=args.limit,
        force=args.force,
        delay_seconds=args.delay,
    )

    logger.info("")
    logger.info("Bilan final :")
    logger.info(f"  Articles Green IT traites : {stats['total']}")
    logger.info(f"  Succes                    : {stats['succes']}")
    logger.info(f"  Echecs                    : {stats['echecs']}")

    if stats["echecs"] > 0:
        logger.warning(
            f"{stats['echecs']} echec(s) detecte(s). Verifier les logs ci-dessus "
            "puis relancer sans --force pour reprendre uniquement les articles manquants."
        )


if __name__ == "__main__":
    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(main())
