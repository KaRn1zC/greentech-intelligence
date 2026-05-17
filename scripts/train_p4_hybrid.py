"""P4.4 hybride : mDeBERTa K=5x3 (~1h45) puis Qwen3-4B K=3x2 reduit (~6h).

Strategie validee le 2026-05-17 face a la lenteur intrinseque de Qwen3-4B
sur RX 7900 XTX (10 sec/optimizer-step => K=5x3 prendrait ~30h, inacceptable).

Sequence d'execution :

1. **mDeBERTa-v3-base** (278M params, full fine-tune) :
   - K=5 folds x 3 seeds = 15 trainings
   - 5 epochs, bf16, batch effectif 32, gradient_checkpointing
   - Calibration + SWA + ensemble logit-average
   - Duree estimee : ~1h45 cumulees
   - Sortie : models/mdeberta/folds/, models/mdeberta/ensemble_config.json

2. **Qwen3-4B + LoRA** (hyperparams reduits) :
   - K=3 folds x 2 seeds = 6 trainings (vs 15 historique)
   - **2 epochs** au lieu de 3 (Unsloth recommande 1-3 pour LoRA classif)
   - **r=16, alpha=32** au lieu de r=32, alpha=64 (sweet spot Unsloth 2026)
   - all-linear (7 modules) + rsLoRA + gradient_checkpointing maintenus
   - Calibration + SWA + TIES-merging maintenus
   - Duree estimee : ~6h cumulees
   - Sortie : models/qwen3/folds/, models/qwen3/merged/, models/qwen3/ensemble_config.json

3. **Bilan** ecrit dans models/p4_hybrid_summary.json pour P5.1.

Usage
-----

::

    # Stratégie hybride complete (~8h en background)
    uv run python scripts/train_p4_hybrid.py

    # mDeBERTa uniquement (~1h45, pour valider d'abord)
    uv run python scripts/train_p4_hybrid.py --mdeberta-only

    # Qwen3 uniquement (~6h, apres validation mDeBERTa OK)
    uv run python scripts/train_p4_hybrid.py --qwen3-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Force stdout/stderr en UTF-8 sur Windows pour eviter UnicodeEncodeError sur
# les emojis que MLflow imprime (ex. \U0001f3c3 = 🏃 dans "View run ...").
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
SUMMARY_PATH = BASE_DIR / "models" / "p4_hybrid_summary.json"


async def train_mdeberta() -> dict:
    """Entraine mDeBERTa avec K=5x3 (protocole unifie complet)."""
    from greentech.ai.models.training import train_with_unified_protocol

    logger.info("#" * 78)
    logger.info("#  STAGE 1/2 : mDeBERTa-v3-base K=5 folds x 3 seeds")
    logger.info("#  Hyperparams : 5 epochs, batch effectif 32, bf16, gc=True")
    logger.info("#  Duree estimee : ~1h45")
    logger.info("#" * 78)

    t0 = time.perf_counter()
    result = await train_with_unified_protocol(
        model_type="mdeberta",
        n_splits=5,
        n_seeds=3,
        base_random_state=42,
        strict_stratification=False,
    )
    duration = time.perf_counter() - t0
    logger.info(f"mDeBERTa K=5x3 termine en {duration / 60:.1f} min")
    return {"model_type": "mdeberta", "n_splits": 5, "n_seeds": 3, "duration_seconds": duration, "result": result}


async def train_qwen3() -> dict:
    """Entraine Qwen3-4B avec K=3x2 reduit (hyperparams adaptes : r=16, alpha=32, 2 epochs)."""
    from greentech.ai.models.training import train_with_unified_protocol

    logger.info("#" * 78)
    logger.info("#  STAGE 2/2 : Qwen3-4B + LoRA K=3 folds x 2 seeds (reduit)")
    logger.info("#  Hyperparams : 2 epochs, r=16 alpha=32, all-linear, rsLoRA, gc=True")
    logger.info("#  Duree estimee : ~6h")
    logger.info("#" * 78)

    t0 = time.perf_counter()
    result = await train_with_unified_protocol(
        model_type="qwen3",
        n_splits=3,
        n_seeds=2,
        base_random_state=42,
        strict_stratification=False,
    )
    duration = time.perf_counter() - t0
    logger.info(f"Qwen3 K=3x2 termine en {duration / 60:.1f} min")
    return {"model_type": "qwen3", "n_splits": 3, "n_seeds": 2, "duration_seconds": duration, "result": result}


async def run(do_mdeberta: bool, do_qwen3: bool) -> int:
    """Orchestre l'entrainement hybride P4.4."""
    summary: dict = {
        "date_start": datetime.now(UTC).isoformat(),
        "strategy": "hybride_p4_4_2026_05_17",
        "stages": [],
    }

    if do_mdeberta:
        try:
            mdeberta_result = await train_mdeberta()
            summary["stages"].append(mdeberta_result)
        except Exception as exc:
            logger.exception(f"Echec entrainement mDeBERTa : {exc}")
            summary["stages"].append({"model_type": "mdeberta", "error": str(exc)})
            if do_qwen3:
                logger.warning("mDeBERTa a echoue, on continue tout de meme avec Qwen3.")

    if do_qwen3:
        try:
            qwen3_result = await train_qwen3()
            summary["stages"].append(qwen3_result)
        except Exception as exc:
            logger.exception(f"Echec entrainement Qwen3 : {exc}")
            summary["stages"].append({"model_type": "qwen3", "error": str(exc)})

    summary["date_end"] = datetime.now(UTC).isoformat()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    logger.info(f"Bilan P4.4 hybride persiste : {SUMMARY_PATH}")
    return 0


def main() -> None:
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=True)

    parser = argparse.ArgumentParser(
        description="P4.4 hybride : mDeBERTa K=5x3 + Qwen3 K=3x2 reduit (~8h total)."
    )
    parser.add_argument(
        "--mdeberta-only",
        action="store_true",
        help="Lance UNIQUEMENT mDeBERTa K=5x3 (~1h45).",
    )
    parser.add_argument(
        "--qwen3-only",
        action="store_true",
        help="Lance UNIQUEMENT Qwen3 K=3x2 reduit (~6h).",
    )
    args = parser.parse_args()

    if args.mdeberta_only and args.qwen3_only:
        logger.error("--mdeberta-only et --qwen3-only sont mutuellement exclusifs.")
        sys.exit(1)

    do_mdeberta = not args.qwen3_only
    do_qwen3 = not args.mdeberta_only
    sys.exit(asyncio.run(run(do_mdeberta, do_qwen3)))


if __name__ == "__main__":
    main()
