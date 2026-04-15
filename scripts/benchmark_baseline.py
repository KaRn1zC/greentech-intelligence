"""Evalue la baseline d'un modele pre-entraine sur le Golden Dataset.

Script dedie a la mesure des metriques de reference d'un LLM SANS
fine-tuning, avec tracking MLflow complet. Utile avant chaque cycle
d'entrainement LoRA K-fold pour disposer d'une reference solide contre
laquelle comparer le gain apporte par le fine-tuning.

Par defaut, evalue `Qwen/Qwen3-4B` (valeur de
``settings.huggingface_model_baseline``). Un autre modele peut etre passe
en argument pour evaluer la baseline d'une variante.

Usage:
    # Evaluation du modele par defaut (Qwen3-4B)
    uv run python scripts/benchmark_baseline.py

    # Evaluation d'un autre modele
    uv run python scripts/benchmark_baseline.py Qwen/Qwen3-8B

Artefacts produits:
    - `models/baseline_metrics.json` : metriques + metadata (date, modele, portee)
    - Run MLflow : `baseline-<nom_modele>` dans l'experience `greentech-classification`
    - Emissions CO2 (via CodeCarbon, integre dans `tracked_experiment`)
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# Racine du projet (scripts/ est au niveau de la racine)
BASE_DIR = Path(__file__).resolve().parent.parent


def _log_metrics_report(metrics: dict, title: str) -> None:
    """Affiche un rapport lisible des metriques dans les logs."""
    logger.info("")
    logger.info("-" * 72)
    logger.info(f"  {title}")
    logger.info("-" * 72)
    logger.info(f"  MCC (critere principal) : {metrics['mcc']:.4f}")
    logger.info(f"  F1-score                : {metrics['f1']:.4f}")
    logger.info(
        f"  Accuracy                : {metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )
    logger.info(
        f"  Balanced accuracy       : {metrics['balanced_accuracy']:.4f} "
        f"({metrics['balanced_accuracy'] * 100:.2f}%)"
    )
    logger.info(f"  Precision               : {metrics['precision']:.4f}")
    logger.info(f"  Recall Green IT         : {metrics['recall']:.4f}")
    logger.info(f"  Specificite             : {metrics['specificite']:.4f}")
    logger.info("")
    logger.info("  Matrice de confusion :")
    logger.info(
        f"    TP = {metrics['vrais_positifs']:>5d}  |  FN = {metrics['faux_negatifs']:>5d}"
    )
    logger.info(
        f"    FP = {metrics['faux_positifs']:>5d}  |  TN = {metrics['vrais_negatifs']:>5d}"
    )
    logger.info("")
    logger.info("  Distribution :")
    logger.info(
        f"    Reels   : Green IT = {metrics['nb_reels_green_it']:>5d}  |  "
        f"Non Green IT = {metrics['nb_reels_non_green_it']:>5d}"
    )
    logger.info(
        f"    Predits : Green IT = {metrics['nb_predictions_green_it']:>5d}  |  "
        f"Non Green IT = {metrics['nb_predictions_non_green_it']:>5d}"
    )
    if "latence_moyenne_ms" in metrics:
        logger.info("")
        logger.info(f"  Latence moyenne : {metrics['latence_moyenne_ms']:.2f} ms")
    if "latence_p95_ms" in metrics:
        logger.info(f"  Latence p95     : {metrics['latence_p95_ms']:.2f} ms")
    logger.info("-" * 72)


async def run_baseline(model_name: str | None = None) -> int:
    """Orchestre l'evaluation baseline : chargement, inference, tracking.

    Args:
        model_name: Identifiant HF du modele a evaluer. Si ``None``, utilise
            ``settings.huggingface_model_baseline``.

    Returns:
        Code de sortie shell (0 = succes, 1 = echec).
    """
    from greentech.ai.models.baseline import evaluate_baseline
    from greentech.config import get_settings

    settings = get_settings()
    resolved_model = model_name or settings.huggingface_model_baseline

    logger.info("")
    logger.info("#" * 70)
    logger.info("#  BASELINE - EVALUATION D'UN MODELE PRE-ENTRAINE")
    logger.info(f"#  Modele : {resolved_model}")
    logger.info(f"#  Date   : {datetime.now(UTC).isoformat()}")
    logger.info("#" * 70)

    try:
        result = evaluate_baseline(model_name=resolved_model)
    except FileNotFoundError as exc:
        logger.error(f"Dataset introuvable : {exc}")
        logger.info(
            "Executer d'abord : uv run python scripts/export_golden_dataset.py"
        )
        return 1
    except Exception as exc:
        logger.exception(f"Baseline echouee : {exc}")
        return 1

    _log_metrics_report(
        result.metrics,
        f"Baseline : {result.model_name} (zero-shot, {result.n_articles} articles)",
    )

    from greentech.ai.mlops.baseline_tracking import track_baseline

    track_baseline(result, BASE_DIR / "models" / "baseline_metrics.json")

    return 0


def main() -> None:
    """Point d'entree CLI."""
    from greentech.utils.logger import setup_logging

    # Loki actif : le run baseline est visible dans Grafana Explore au meme
    # titre que les runs d'entrainement, pour une tracabilite uniforme.
    setup_logging(level="INFO", enable_loki=True)

    args = sys.argv[1:]
    model_name = args[0] if args else None

    exit_code = asyncio.run(run_baseline(model_name))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
