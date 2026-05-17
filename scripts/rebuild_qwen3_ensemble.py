"""Reconstruit l'ensemble Qwen3 à partir des 6 folds K=3×2 existants.

Contexte
--------

L'entraînement P4.4 hybride du 2026-05-17 a produit 6 adapters Qwen3-4B + LoRA
sains dans ``models/qwen3/folds/`` mais le TIES-merging final a planté sur un
bug PEFT 0.19.0 (``add_weighted_adapter`` refusé en présence de
``modules_to_save`` + crash ``TypeError: unhashable type: 'list'`` au
``merge_and_unload``). Depuis P4.15, ``training._merge_lora_adapters``
implémente TIES manuellement sur les safetensors (pas de dépendance PEFT
sur ``add_weighted_adapter``). Ce script ré-exerce le pipeline sur les 6
folds existants sans aucun re-entraînement.

Étapes
------

1. Lit les MCC/F1/T°/seuil de chaque fold depuis ``optimal_threshold.json`` et
   ``temperature.json``.
2. Sélectionne le top-1 seed par fold par MCC (3 adapters retenus sur 6).
3. Lance ``_merge_lora_adapters`` patché → ``models/qwen3/merged/``.
4. Écrit ``models/qwen3/ensemble_config.json`` au format identique à mDeBERTa.
5. Met à jour ``models/p4_hybrid_summary.json`` avec le bilan Qwen3 complet.

Usage
-----

::

    uv run python scripts/rebuild_qwen3_ensemble.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
QWEN3_ROOT = BASE_DIR / "models" / "qwen3"
FOLDS_ROOT = QWEN3_ROOT / "folds"
MERGED_DIR = QWEN3_ROOT / "merged"
ENSEMBLE_PATH = QWEN3_ROOT / "ensemble_config.json"
SUMMARY_PATH = BASE_DIR / "models" / "p4_hybrid_summary.json"

N_FOLDS = 3
N_SEEDS = 2


def load_fold_metrics() -> list[dict]:
    """Lit les MCC/T°/seuil des 6 folds depuis leurs JSON locaux."""
    metrics: list[dict] = []
    for fold in range(1, N_FOLDS + 1):
        for seed_idx in range(1, N_SEEDS + 1):
            fold_dir = FOLDS_ROOT / f"fold_{fold}_seed_{seed_idx}"
            if not fold_dir.exists():
                msg = f"Fold manquant : {fold_dir}"
                raise FileNotFoundError(msg)

            threshold_data = json.loads(
                (fold_dir / "optimal_threshold.json").read_text(encoding="utf-8")
            )
            temperature_data = json.loads(
                (fold_dir / "temperature.json").read_text(encoding="utf-8")
            )

            metrics.append({
                "fold": fold,
                "seed_idx": seed_idx,
                "seed": 41 + seed_idx,
                "mcc": float(threshold_data["value"]),
                "threshold": float(threshold_data["threshold"]),
                "temperature": float(temperature_data["temperature"]),
                "checkpoint_path": str(fold_dir),
            })
    return metrics


def select_best_seed_per_fold(metrics: list[dict]) -> list[dict]:
    """Pour chaque fold, garde le seed avec le meilleur MCC."""
    best: dict[int, dict] = {}
    for entry in metrics:
        fold = entry["fold"]
        if fold not in best or entry["mcc"] > best[fold]["mcc"]:
            best[fold] = entry
    return [best[fold] for fold in sorted(best.keys())]


def main() -> int:
    from greentech.ai.models.training import _merge_lora_adapters
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)

    logger.info("#" * 78)
    logger.info("#  Rebuild Qwen3 ensemble (fix TIES-merging post P4.4)")
    logger.info("#" * 78)

    metrics = load_fold_metrics()
    mcc_values = [m["mcc"] for m in metrics]
    mcc_mean = sum(mcc_values) / len(mcc_values)
    logger.info(f"6 folds chargés : MCC mean={mcc_mean:.4f}, min={min(mcc_values):.4f}, max={max(mcc_values):.4f}")

    best_per_fold = select_best_seed_per_fold(metrics)
    selected_paths = [Path(b["checkpoint_path"]) for b in best_per_fold]
    logger.info(f"Top-1 par fold sélectionnés : {[b['fold'] for b in best_per_fold]} (MCC moyenne {sum(b['mcc'] for b in best_per_fold) / 3:.4f})")
    for b in best_per_fold:
        logger.info(f"  Fold {b['fold']} : seed_{b['seed_idx']} MCC={b['mcc']:.4f} T={b['temperature']:.3f} seuil={b['threshold']:.2f}")

    logger.info("Lancement TIES-merging manuel (Yadav 2023, density=0.5)...")
    _merge_lora_adapters(
        fold_checkpoints=selected_paths,
        output_dir=MERGED_DIR,
        density=0.5,
    )
    logger.info(f"TIES-merging OK : {MERGED_DIR}")

    # _merge_lora_adapters supprime puis recree output_dir, donc les calibrations
    # potentiellement copiees auparavant ont disparu. On les recopie depuis la
    # racine du modele (qui contient les T/seuil moyens K-fold persistes par
    # train_with_unified_protocol). Cela aligne sur ce que ``_build_ensemble``
    # fait dans le pipeline natif.
    import shutil as _shutil
    for _cfile in ("temperature.json", "optimal_threshold.json"):
        _src = QWEN3_ROOT / _cfile
        if _src.exists():
            _shutil.copy(_src, MERGED_DIR / _cfile)
            logger.info(f"Calibration recopiée dans merged/ : {_cfile}")

    folds_info = [
        {
            "fold": b["fold"],
            "seed_idx": b["seed_idx"],
            "seed": b["seed"],
            "mcc": b["mcc"],
            "temperature": b["temperature"],
            "threshold": b["threshold"],
            "checkpoint_path": b["checkpoint_path"],
        }
        for b in best_per_fold
    ]

    mean_temperature = sum(b["temperature"] for b in best_per_fold) / 3
    mean_threshold = sum(b["threshold"] for b in best_per_fold) / 3

    ensemble_info = {
        "strategy": "ties_manual",
        "model_type": "qwen3",
        "folds": folds_info,
        "inference_model_path": str(MERGED_DIR),
        "calibration": {
            "temperature": mean_temperature,
            "threshold": mean_threshold,
        },
        "metadata": {
            "built_at": datetime.now(UTC).isoformat(),
            "n_folds": 3,
            "n_seeds_per_fold": 2,
            "cv_mcc_mean": mcc_mean,
            "cv_mcc_per_run": mcc_values,
            "post_hoc_rebuild": True,
            "ties_density": 0.5,
            "ties_reference": "Yadav et al. NeurIPS 2023, arXiv:2306.01708",
        },
    }

    ENSEMBLE_PATH.write_text(json.dumps(ensemble_info, indent=2, ensure_ascii=False))
    logger.info(f"ensemble_config.json écrit : {ENSEMBLE_PATH}")

    # Patch p4_hybrid_summary.json
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        for stage in summary.get("stages", []):
            if stage.get("model_type") == "qwen3":
                stage.pop("error", None)
                stage["n_splits"] = 3
                stage["n_seeds"] = 2
                stage["mcc_mean"] = mcc_mean
                stage["mcc_per_run"] = mcc_values
                stage["ensemble"] = {
                    "strategy": "merge_lora",
                    "n_folds": 3,
                    "merged_path": str(MERGED_DIR),
                    "rebuilt_at": datetime.now(UTC).isoformat(),
                }
                break
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        logger.info(f"Bilan P4.4 mis à jour : {SUMMARY_PATH}")

    logger.info("=" * 78)
    logger.info(f"  Rebuild OK : Qwen3 MCC mean={mcc_mean:.4f} ({len(metrics)} folds)")
    logger.info(f"  merged/ : {MERGED_DIR}")
    logger.info(f"  ensemble_config : {ENSEMBLE_PATH}")
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
