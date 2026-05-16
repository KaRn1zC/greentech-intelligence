"""Evalue la baseline d'un modele pre-entraine sur le Golden Dataset.

Script dedie a la mesure des metriques de reference d'un LLM SANS
fine-tuning, avec tracking MLflow complet. Utile avant chaque cycle
d'entrainement LoRA K-fold pour disposer d'une reference solide contre
laquelle comparer le gain apporte par le fine-tuning.

Modes
-----

1. **Mode simple** (defaut) : evalue un seul modele.
   Par defaut, le modele est ``settings.huggingface_model_baseline`` (Qwen3-4B).
   Un autre modele peut etre passe en argument positionnel.

2. **Mode --compare** (P4.2 - benchmark equitable) : evalue
   sequentiellement Qwen3-4B et mDeBERTa-v3-base sur le meme dataset,
   genere un rapport comparatif JSON et logge les deux runs MLflow sous
   le tag commun ``compare_run=baseline-comparison-2026-04``. Sert de
   point zero avant l'entrainement K-fold B4.

Usage
-----

::

    # Evaluation du modele par defaut (Qwen3-4B)
    uv run python scripts/benchmark_baseline.py

    # Evaluation d'un autre modele
    uv run python scripts/benchmark_baseline.py Qwen/Qwen3-8B

    # Benchmark comparatif Qwen3-4B vs mDeBERTa-v3-base (P4.2)
    uv run python scripts/benchmark_baseline.py --compare

Artefacts produits
------------------

Mode simple :
    - ``models/baseline_metrics.json`` : metriques + metadata (date, modele, portee)
    - Run MLflow : ``baseline-<nom_modele>`` dans l'experience ``greentech-classification``
    - Emissions CO2 (via CodeCarbon, integre dans ``tracked_experiment``)

Mode --compare :
    - ``models/baseline_metrics.json`` : metriques du dernier modele evalue
      (Qwen3 par defaut, ecrase par mDeBERTa si execute apres)
    - ``models/baseline_comparison_2026-04.json`` : rapport JSON consolide
      avec les 2 jeux de metriques cote a cote, pret a etre cite dans
      ``docs/BENCHMARK_BRUT_2026-04.md``
    - 2 runs MLflow distincts taggues ``compare_run=baseline-comparison-2026-04``
    - Emissions CO2 pour chaque modele
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# Racine du projet (scripts/ est au niveau de la racine)
BASE_DIR = Path(__file__).resolve().parent.parent

# Nom du run de comparaison utilise comme tag MLflow et dans le fichier consolide.
# Aligne avec la checklist P4.2 et le futur docs/BENCHMARK_BRUT_2026-04.md.
COMPARE_RUN_TAG = "baseline-comparison-2026-04"
COMPARE_OUTPUT_FILE = BASE_DIR / "models" / "baseline_comparison_2026-04.json"


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


def _log_comparison_table(results: dict[str, dict]) -> None:
    """Affiche un tableau comparatif des metriques cle pour les 2 modeles."""
    logger.info("")
    logger.info("=" * 78)
    logger.info("  COMPARAISON BASELINE (Qwen3-4B vs mDeBERTa-v3-base)")
    logger.info("=" * 78)
    headers = ["Metrique", "Qwen3-4B (decoder)", "mDeBERTa-v3-base (encoder)"]
    logger.info(f"  {headers[0]:<22s} | {headers[1]:>22s} | {headers[2]:>26s}")
    logger.info("  " + "-" * 76)

    key_metrics = [
        ("MCC", "mcc"),
        ("F1-score", "f1"),
        ("Precision", "precision"),
        ("Recall Green IT", "recall"),
        ("Accuracy", "accuracy"),
        ("Balanced acc.", "balanced_accuracy"),
        ("Specificite", "specificite"),
        ("Latence moy. (ms)", "latence_moyenne_ms"),
        ("Latence p95 (ms)", "latence_p95_ms"),
    ]
    qwen_metrics = results.get("qwen3", {}).get("metrics", {})
    mdeberta_metrics = results.get("mdeberta", {}).get("metrics", {})
    for label, key in key_metrics:
        qval = qwen_metrics.get(key)
        mval = mdeberta_metrics.get(key)
        qstr = f"{qval:>22.4f}" if isinstance(qval, (int, float)) else f"{'-':>22s}"
        mstr = f"{mval:>26.4f}" if isinstance(mval, (int, float)) else f"{'-':>26s}"
        logger.info(f"  {label:<22s} | {qstr} | {mstr}")
    logger.info("=" * 78)


def _write_comparison_json(results: dict[str, dict]) -> None:
    """Persiste le rapport JSON consolide pour reutilisation dans la doc P4.3."""
    COMPARE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "compare_run": COMPARE_RUN_TAG,
        "date": datetime.now(UTC).isoformat(),
        "phase": "B4.2",
        "purpose": (
            "Benchmark BRUT (zero-shot, sans fine-tuning) pour disposer "
            "d'un point zero comparable entre Qwen3-4B (decoder) et "
            "mDeBERTa-v3-base (encoder) avant l'entrainement K-fold unifie B4.3."
        ),
        "models": results,
    }
    COMPARE_OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info(f"Rapport comparatif persiste : {COMPARE_OUTPUT_FILE}")


def _result_to_dict(result, dataset_signature: str | None) -> dict:
    """Serialise un BaselineResult en dict pret pour JSON."""
    payload = {
        "model": result.model_name,
        "n_articles": result.n_articles,
        "duration_seconds": result.duration_seconds,
        "dataset_signature": dataset_signature,
        "metrics": dict(result.metrics),
    }
    # Inclure les predictions et latences uniquement si l'usage en aval le demande.
    # Par defaut, on les omet pour garder le JSON lisible (sinon ~6000 entrees).
    return payload


async def _evaluate_one(model_name: str, *, compare_tag: str | None = None) -> tuple[object, str | None]:
    """Evalue un seul modele et retourne (result, signature).

    Args:
        model_name: Identifiant HF du modele a evaluer.
        compare_tag: Si fourni, tag MLflow ``compare_run=<tag>`` ajoute au run.

    Returns:
        Tuple ``(BaselineResult, dataset_signature)``.
    """
    from greentech.ai.mlops.baseline_tracking import (
        compute_dataset_signature,
        log_baseline_to_mlflow,
        push_baseline_to_prometheus,
        save_baseline_metrics_json,
    )
    from greentech.ai.models.baseline import evaluate_baseline

    result = evaluate_baseline(model_name=model_name)
    _log_metrics_report(
        result.metrics,
        f"Baseline : {result.model_name} (zero-shot, {result.n_articles} articles)",
    )

    signature = compute_dataset_signature(BASE_DIR / "data" / "golden_dataset.csv")

    # Persistance triple (JSON + MLflow + Prometheus) avec le tag compare si fourni.
    save_baseline_metrics_json(
        result, BASE_DIR / "models" / "baseline_metrics.json", dataset_signature=signature
    )
    try:
        log_baseline_to_mlflow(result, dataset_signature=signature)
    except Exception as exc:
        logger.warning(f"MLflow indisponible, baseline non trackee : {exc}")
    push_baseline_to_prometheus(result)

    # Pour le mode --compare, on logue en plus un tag MLflow pour retrouver les
    # runs ensemble dans l'UI. Le tag est applique en post-traitement sur le
    # run actif s'il existe encore (sinon, tag-only via search_runs).
    if compare_tag is not None:
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("greentech-classification")
            if exp is not None:
                runs = client.search_runs(
                    [exp.experiment_id],
                    filter_string=f"tags.modele = '{model_name}' and tags.type = 'baseline'",
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if runs:
                    client.set_tag(runs[0].info.run_id, "compare_run", compare_tag)
                    logger.info(
                        f"Tag MLflow ajoute : compare_run={compare_tag} "
                        f"sur run {runs[0].info.run_id}"
                    )
        except Exception as exc:
            logger.warning(f"Tag compare_run non applique : {exc}")

    return result, signature


async def run_baseline(model_name: str | None = None) -> int:
    """Orchestre l'evaluation baseline d'UN modele.

    Args:
        model_name: Identifiant HF du modele a evaluer. Si ``None``, utilise
            ``settings.huggingface_model_baseline``.

    Returns:
        Code de sortie shell (0 = succes, 1 = echec).
    """
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
        await _evaluate_one(resolved_model)
    except FileNotFoundError as exc:
        logger.error(f"Dataset introuvable : {exc}")
        logger.info(
            "Executer d'abord : uv run python scripts/export_golden_dataset.py"
        )
        return 1
    except Exception as exc:
        logger.exception(f"Baseline echouee : {exc}")
        return 1

    return 0


async def run_compare() -> int:
    """Orchestre le benchmark comparatif Qwen3-4B vs mDeBERTa-v3-base (P4.2).

    Evalue les deux modeles sequentiellement (le GPU ne peut pas heberger
    les deux en meme temps), persiste un rapport JSON consolide
    ``models/baseline_comparison_2026-04.json`` et tagge les 2 runs MLflow
    sous ``compare_run=baseline-comparison-2026-04``.

    Returns:
        Code de sortie shell (0 = succes, 1 si un des modeles echoue).
    """
    from greentech.config import get_settings

    settings = get_settings()
    qwen_model = settings.huggingface_model_baseline
    mdeberta_model = settings.huggingface_model_encoder_base

    logger.info("")
    logger.info("#" * 78)
    logger.info("#  BENCHMARK COMPARATIF BRUT (P4.2 - zero-shot)")
    logger.info(f"#  Decoder  : {qwen_model}")
    logger.info(f"#  Encoder  : {mdeberta_model}")
    logger.info(f"#  Date     : {datetime.now(UTC).isoformat()}")
    logger.info(f"#  Tag run  : {COMPARE_RUN_TAG}")
    logger.info("#" * 78)

    results: dict[str, dict] = {}

    # 1) Qwen3-4B (decoder generatif)
    try:
        logger.info("")
        logger.info(">>> [1/2] Evaluation Qwen3-4B (decoder)")
        qwen_result, signature = await _evaluate_one(qwen_model, compare_tag=COMPARE_RUN_TAG)
        results["qwen3"] = _result_to_dict(qwen_result, signature)
        results["qwen3"]["architecture"] = "decoder (causal LM avec tete classification num_labels=2)"
    except Exception as exc:
        logger.exception(f"Echec evaluation Qwen3-4B : {exc}")
        return 1

    # 2) mDeBERTa-v3-base (encoder)
    try:
        logger.info("")
        logger.info(">>> [2/2] Evaluation mDeBERTa-v3-base (encoder)")
        mdeberta_result, signature = await _evaluate_one(
            mdeberta_model, compare_tag=COMPARE_RUN_TAG
        )
        results["mdeberta"] = _result_to_dict(mdeberta_result, signature)
        results["mdeberta"]["architecture"] = (
            "encoder bidirectionnel (DisentangledSelfAttention, multilingue 100 langues)"
        )
    except Exception as exc:
        logger.exception(f"Echec evaluation mDeBERTa-v3-base : {exc}")
        return 1

    # 3) Rapport consolide
    _log_comparison_table(results)
    _write_comparison_json(results)

    logger.info("")
    logger.info("=" * 78)
    logger.info("  BENCHMARK COMPARATIF TERMINE")
    logger.info("  Etape suivante : rediger docs/BENCHMARK_BRUT_2026-04.md (P4.3)")
    logger.info("=" * 78)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalue la baseline zero-shot d'un modele HF sur le Golden Dataset. "
            "Utiliser --compare pour le benchmark Qwen3-4B vs mDeBERTa-v3-base (P4.2)."
        )
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help=(
            "Identifiant HF du modele a evaluer (mode simple). Ignore en mode "
            "--compare."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Lance le benchmark comparatif Qwen3-4B vs mDeBERTa-v3-base. "
            "Genere models/baseline_comparison_2026-04.json + 2 runs MLflow."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entree CLI."""
    from greentech.utils.logger import setup_logging

    # Loki actif : le run baseline est visible dans Grafana Explore au meme
    # titre que les runs d'entrainement, pour une tracabilite uniforme.
    setup_logging(level="INFO", enable_loki=True)

    args = _parse_args()

    if args.compare:
        exit_code = asyncio.run(run_compare())
    else:
        exit_code = asyncio.run(run_baseline(args.model_name))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
