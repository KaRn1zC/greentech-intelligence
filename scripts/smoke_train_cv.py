"""Mini-run de validation du protocole unifie B3 avant P4.4 full.

Lance un train_with_unified_protocol minimaliste (n_splits=2, n_seeds=1
= 2 trainings au lieu de 15) sur le dataset complet pour valider que :

1. rsLoRA s'active correctement (Qwen3)
2. TIES-merging s'execute sans erreur (Qwen3, fusion des 2 folds)
3. SWACallback collecte les snapshots et applique la moyenne en fin
4. La calibration post-fold (Temperature Scaling + threshold) tourne
5. L'ensemble final est ecrit dans models/<model>/

Duree estimee :
- mDeBERTa : ~6-12 min (2 trainings full fine-tune 278M params)
- Qwen3-4B : ~15-25 min (2 trainings LoRA all-linear)

Si le mini-run OK, on peut lancer P4.4 (train-cv-both, K=5 x 3 seeds = 15
trainings par modele) en confiance.

Usage
-----

::

    # mDeBERTa d'abord (plus rapide a valider)
    uv run python scripts/smoke_train_cv.py --model mdeberta

    # Qwen3 ensuite
    uv run python scripts/smoke_train_cv.py --model qwen3

"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

# Force stdout/stderr en UTF-8 sur Windows pour eviter UnicodeEncodeError sur
# les emojis que MLflow imprime (ex. \U0001f3c3 = 🏃 dans "View run ..."). Sans
# cela, le codec charmap (cp1252) Windows crashe a la fin de chaque run MLflow.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from loguru import logger


async def run(model_type: str) -> int:
    """Lance le mini-run train_with_unified_protocol."""
    from greentech.ai.models.training import train_with_unified_protocol

    logger.info("#" * 78)
    logger.info(f"#  SMOKE TRAIN-CV : {model_type.upper()}")
    logger.info("#  Mode : 2 folds x 1 seed = 2 trainings (vs 15 en P4.4 full)")
    logger.info(f"#  Date : {datetime.now(UTC).isoformat()}")
    logger.info("#" * 78)

    try:
        result = await train_with_unified_protocol(
            model_type=model_type,
            n_splits=2,
            n_seeds=1,
            base_random_state=42,
            strict_stratification=False,
        )
    except Exception as exc:
        logger.exception(f"Smoke train-cv {model_type} echoue : {exc}")
        return 1

    logger.info("")
    logger.info("=" * 78)
    logger.info(f"  SMOKE TRAIN-CV {model_type.upper()} OK")
    logger.info(f"  Resultat : {result}")
    logger.info("=" * 78)
    return 0


def main() -> None:
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=True)

    parser = argparse.ArgumentParser(
        description="Mini-run de validation du protocole unifie B3 (n_splits=2, n_seeds=1)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mdeberta", "qwen3"],
        help="Modele a tester : mdeberta (~6-12 min) ou qwen3 (~15-25 min).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.model)))


if __name__ == "__main__":
    main()
