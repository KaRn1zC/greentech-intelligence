"""Genere en batch les resumes de classification pour tous les articles.

Ce script peuple la colonne ``articles.resume`` en appelant le LLM
instructif (HF Serverless, avec fallback local Qwen sur GPU AMD ROCm) via
le prompt centralise dans ``classification_summarizer.py``. C'est cette
colonne qui sert ensuite d'entree canonique au classifieur Qwen3-4B + LoRA.

Utilisation
-----------

Lancer une regeneration complete du dataset (ecrase les resumes existants) ::

    uv run python scripts/generate_classification_summaries.py --force

Tester le prompt sur un echantillon limite avant de lancer le run
complet (~5-9 h sur le fallback local) ::

    uv run python scripts/generate_classification_summaries.py --limit 20 --force

Reprendre un run interrompu sans ecraser les resumes deja produits ::

    uv run python scripts/generate_classification_summaries.py

Le script est idempotent par defaut : sans ``--force``, il saute les
articles dont le ``resume`` est deja renseigne. Cela permet de relancer
sans risque apres une interruption.

"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from greentech.ai.services.summarizer import summarize_all_articles_for_classification
from greentech.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande.

    Returns:
        Namespace avec les champs ``force``, ``limit`` et ``delay``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Genere les resumes de classification (colonne articles.resume) "
            "pour tous les articles en base via le LLM Qwen (HF ou fallback local)."
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
        help=(
            "Plafond sur le nombre d'articles a traiter (utile pour tester le "
            "prompt sur un echantillon avant le run complet)."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help=(
            "Pause en secondes entre deux articles pour respecter le fair-use "
            "HF Serverless (defaut : 0.3 s)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help=(
            "Selectionne les articles dans un ordre aleatoire au lieu de l'ordre "
            "d'insertion (id_article). Utile en smoke test pour echantillonner "
            "toutes les sources au lieu des premiers inseres (dominante arXiv)."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    """Execute le batch et affiche les statistiques finales."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  GENERATION BATCH DES RESUMES DE CLASSIFICATION")
    logger.info("=" * 70)
    logger.info(f"  force   : {args.force}")
    logger.info(f"  limit   : {args.limit if args.limit is not None else 'aucune (integralite)'}")
    logger.info(f"  delay   : {args.delay}s entre articles")
    logger.info(f"  shuffle : {args.shuffle}")
    logger.info("=" * 70)

    stats = await summarize_all_articles_for_classification(
        limit=args.limit,
        force=args.force,
        delay_seconds=args.delay,
        shuffle=args.shuffle,
    )

    logger.info("")
    logger.info("Bilan final :")
    logger.info(f"  Articles traites  : {stats['total']}")
    logger.info(f"  Succes            : {stats['succes']}")
    logger.info(f"  Echecs            : {stats['echecs']}")

    if stats["echecs"] > 0:
        logger.warning(
            f"{stats['echecs']} echec(s) detecte(s). Verifier les logs ci-dessus "
            "puis relancer sans --force pour reprendre uniquement les articles sans resume."
        )


if __name__ == "__main__":
    # Logging persistant console + fichier (sans Loki pour ne pas bloquer si
    # la stack monitoring n'est pas up).
    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(main())
