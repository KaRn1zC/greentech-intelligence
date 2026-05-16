"""Baseline rigoureuse : linear probing + zero-shot bonus (P4.6).

Remplace la baseline "tete random" historique (methodologiquement faible) par
quatre baselines academiquement defendables, executees sequentiellement :

1. **Linear probing Qwen3-4B** (backbone frozen, LogReg sur embeddings mean-pool)
2. **Linear probing mDeBERTa-v3-base** (idem, methode standard SSL)
3. **Zero-shot NLI mDeBERTa-v3-base** (pipeline ``zero-shot-classification``)
4. **Zero-shot prompt Qwen3-Instruct** (LLM avec prompt + parse OUI/NON)

Toutes les baselines partagent le meme jeu de metriques (MCC, F1, Precision,
Recall, latence) pour permettre la comparaison directe avec le futur
benchmark post-entrainement (P5.1).

Sortie consolidee : ``models/baseline_comparison_2026-04.json`` (mise a jour,
ecrase la version "tete random" precedente).

Usage
-----

::

    # Lance les 4 baselines rigoureuses (cumul ~45-60 min sur RX 7900 XTX)
    uv run python scripts/benchmark_baseline_rigorous.py

    # Lance seulement le linear probing (le plus important, ~15-20 min)
    uv run python scripts/benchmark_baseline_rigorous.py --only-linear-probing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
COMPARE_RUN_TAG = "baseline-comparison-2026-04"
COMPARE_OUTPUT_FILE = BASE_DIR / "models" / "baseline_comparison_2026-04.json"


def _log_metrics_short(method: str, model: str, metrics: dict) -> None:
    """Affiche les metriques cle d'une baseline en format compact."""
    logger.info("")
    logger.info(f"  >> {method} | {model}")
    logger.info(
        f"     MCC={metrics['mcc']:.4f}  F1={metrics['f1']:.4f}  "
        f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}"
    )
    logger.info(
        f"     Accuracy={metrics['accuracy']:.4f}  "
        f"BalancedAcc={metrics['balanced_accuracy']:.4f}  "
        f"Specificite={metrics['specificite']:.4f}"
    )
    cm = (
        f"TP={metrics['vrais_positifs']:>5d}  FN={metrics['faux_negatifs']:>5d}  "
        f"FP={metrics['faux_positifs']:>5d}  TN={metrics['vrais_negatifs']:>5d}"
    )
    logger.info(f"     Confusion: {cm}")
    if "latence_moyenne_ms" in metrics:
        logger.info(
            f"     Latence moy={metrics['latence_moyenne_ms']:.2f}ms  "
            f"p95={metrics.get('latence_p95_ms', 0):.2f}ms"
        )


def _result_to_dict(result, dataset_signature: str | None) -> dict:
    """Serialise un RigorousBaselineResult en dict JSON-compatible."""
    return {
        "model": result.model_name,
        "method": result.method,
        "n_articles": result.n_articles,
        "duration_seconds": result.duration_seconds,
        "dataset_signature": dataset_signature,
        "metrics": dict(result.metrics),
    }


async def run(only_linear_probing: bool = False) -> int:
    """Orchestre les 4 baselines rigoureuses et persiste le rapport."""
    from greentech.ai.mlops.baseline_rigorous import (
        run_linear_probing,
        run_zero_shot_nli,
        run_zero_shot_prompt,
        to_baseline_result,
    )
    from greentech.ai.mlops.baseline_tracking import (
        compute_dataset_signature,
        log_baseline_to_mlflow,
        push_baseline_to_prometheus,
    )
    from greentech.ai.models.training import load_full_dataset
    from greentech.config import get_settings

    settings = get_settings()
    dataset_path = BASE_DIR / "data" / "golden_dataset.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset introuvable : {dataset_path}")
        logger.info("Executer d'abord : uv run python scripts/export_golden_dataset.py")
        return 1

    logger.info("#" * 78)
    logger.info("#  BASELINES RIGOUREUSES (P4.6)")
    logger.info(f"#  Date : {datetime.now(UTC).isoformat()}")
    logger.info(f"#  Mode : {'linear probing seul' if only_linear_probing else 'complet (4 baselines)'}")
    logger.info("#" * 78)

    texts, labels = load_full_dataset(dataset_path)
    signature = compute_dataset_signature(dataset_path)
    logger.info(f"Dataset : {len(texts)} articles, signature={signature}")

    results: dict[str, dict] = {}
    qwen_model = settings.huggingface_model_baseline  # Qwen/Qwen3-4B
    mdeberta_model = settings.huggingface_model_encoder_base  # microsoft/mdeberta-v3-base
    qwen_instruct = settings.huggingface_model_local_fallback  # Qwen2.5-3B-Instruct

    # === 1) Linear probing Qwen3-4B (decoder, pooling="last_token") ===
    logger.info("")
    logger.info(">>> [1/4] Linear probing Qwen3-4B")
    qwen_lp = run_linear_probing(qwen_model, texts, labels, pooling="last_token")
    _log_metrics_short("linear_probing", qwen_model, qwen_lp.metrics)
    results["qwen3_linear_probing"] = _result_to_dict(qwen_lp, signature)
    try:
        log_baseline_to_mlflow(
            to_baseline_result(qwen_lp), dataset_signature=signature
        )
        push_baseline_to_prometheus(to_baseline_result(qwen_lp))
    except Exception as exc:
        logger.warning(f"Tracking MLflow/Prometheus echoue : {exc}")

    # === 2) Linear probing mDeBERTa-v3-base (encoder, pooling="mean") ===
    logger.info("")
    logger.info(">>> [2/4] Linear probing mDeBERTa-v3-base")
    mdeberta_lp = run_linear_probing(mdeberta_model, texts, labels, pooling="mean")
    _log_metrics_short("linear_probing", mdeberta_model, mdeberta_lp.metrics)
    results["mdeberta_linear_probing"] = _result_to_dict(mdeberta_lp, signature)
    try:
        log_baseline_to_mlflow(
            to_baseline_result(mdeberta_lp), dataset_signature=signature
        )
        push_baseline_to_prometheus(to_baseline_result(mdeberta_lp))
    except Exception as exc:
        logger.warning(f"Tracking MLflow/Prometheus echoue : {exc}")

    if only_linear_probing:
        logger.info("")
        logger.info("Mode 'only-linear-probing' : sauter les baselines zero-shot bonus.")
        _write_consolidated_json(results, signature, mode="linear_probing_only")
        return 0

    # === 3) Zero-shot NLI mDeBERTa-v3-base ===
    logger.info("")
    logger.info(">>> [3/4] Zero-shot NLI mDeBERTa-v3-base")
    try:
        mdeberta_zs = run_zero_shot_nli(mdeberta_model, texts, labels)
        _log_metrics_short("zero_shot_nli", mdeberta_model, mdeberta_zs.metrics)
        results["mdeberta_zero_shot_nli"] = _result_to_dict(mdeberta_zs, signature)
        try:
            log_baseline_to_mlflow(
                to_baseline_result(mdeberta_zs), dataset_signature=signature
            )
            push_baseline_to_prometheus(to_baseline_result(mdeberta_zs))
        except Exception as exc:
            logger.warning(f"Tracking MLflow/Prometheus echoue : {exc}")
    except Exception as exc:
        logger.exception(f"Zero-shot NLI mDeBERTa echoue : {exc}")
        results["mdeberta_zero_shot_nli"] = {"error": str(exc)}

    # === 4) Zero-shot prompt Qwen3-Instruct (via LocalQwenClient) ===
    logger.info("")
    logger.info(">>> [4/4] Zero-shot prompt Qwen-Instruct (LLM local)")
    try:
        qwen_zs = await run_zero_shot_prompt(qwen_instruct, texts, labels)
        _log_metrics_short("zero_shot_prompt", qwen_instruct, qwen_zs.metrics)
        results["qwen_instruct_zero_shot_prompt"] = _result_to_dict(qwen_zs, signature)
        try:
            log_baseline_to_mlflow(
                to_baseline_result(qwen_zs), dataset_signature=signature
            )
            push_baseline_to_prometheus(to_baseline_result(qwen_zs))
        except Exception as exc:
            logger.warning(f"Tracking MLflow/Prometheus echoue : {exc}")
    except Exception as exc:
        logger.exception(f"Zero-shot prompt Qwen-Instruct echoue : {exc}")
        results["qwen_instruct_zero_shot_prompt"] = {"error": str(exc)}

    _write_consolidated_json(results, signature, mode="full")
    return 0


def _write_consolidated_json(
    results: dict[str, dict], signature: str | None, *, mode: str
) -> None:
    """Persiste le rapport JSON consolide (ecrase l'ancien tete-random)."""
    payload = {
        "compare_run": COMPARE_RUN_TAG,
        "date": datetime.now(UTC).isoformat(),
        "phase": "B4.2 (rigorous re-baseline, P4.6)",
        "mode": mode,
        "dataset_signature": signature,
        "purpose": (
            "Baseline rigoureuse remplacant la version tete-random initiale. "
            "Methodes : linear probing (Qwen3, mDeBERTa) + zero-shot NLI (mDeBERTa) "
            "+ zero-shot prompt (Qwen Instruct). Sert de point zero comparable "
            "avec le futur benchmark post-entrainement (P5.1)."
        ),
        "models": results,
    }
    COMPARE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"  Rapport consolide persiste : {COMPARE_OUTPUT_FILE}")
    logger.info("  Etape suivante : actualiser docs/BENCHMARK_BRUT_2026-04.md (P4.3)")
    logger.info("=" * 78)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline rigoureuse (linear probing + zero-shot bonus) - P4.6."
    )
    parser.add_argument(
        "--only-linear-probing",
        action="store_true",
        help=(
            "Lance uniquement les 2 linear probings (Qwen3 + mDeBERTa), saute les "
            "baselines zero-shot bonus. ~15-20 min total."
        ),
    )
    return parser.parse_args()


def main() -> None:
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=True)
    args = _parse_args()
    exit_code = asyncio.run(run(only_linear_probing=args.only_linear_probing))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
