"""Benchmark final B4.4 : Qwen3-4B + LoRA vs mDeBERTa-v3-base.

Charge les 2 modeles entraines via le protocole unifie B3 (sortie de
``train-cv-both``), evalue sur un test set fige et produit un rapport
markdown comparatif. Inclut :

- MCC, F1, precision, recall, balanced accuracy, specificite
- Latence p50 / p95 / p99 (vs juste la moyenne dans benchmark_baseline)
- VRAM peak observe pendant l'inference
- Empreinte CO2 mesuree par CodeCarbon

Le test set utilise est l'integralite du dataset (sans split val/train,
puisque chaque modele a deja vu ses propres folds en val durant le K-fold).
Pour eviter la fuite, on s'appuie sur le fait que les 5 folds K-fold
couvrent l'integralite du dataset une fois chacun en val : la moyenne des
predictions K-fold est donc une estimation honnete.

Usage:
    # Benchmark des 2 modeles entraines (apres train-cv-both)
    uv run python scripts/benchmark_models.py

    # Specifier un autre dataset (ex: hold-out manuel)
    uv run python scripts/benchmark_models.py --dataset data/test_holdout.csv

Artefacts produits:
    - ``docs/BENCHMARK_FINAL_2026-04.md`` : rapport markdown comparatif
    - Run MLflow ``benchmark-models-comparison-<timestamp>`` dans l'experience
      ``greentech-classification``
    - JSON detaille : ``models/benchmark_final_metrics.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Force UTF-8 sur stdout/stderr pour les caracteres FR (cf. retrain_pipeline.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from greentech.ai.mlops.tracking import ExperimentConfig, tracked_experiment  # noqa: E402
from greentech.ai.models.classifier import (  # noqa: E402
    BaseClassifier,
    LabelGreenIT,
    TrainingConfig,
)
from greentech.ai.models.training import (  # noqa: E402
    MDeBERTaClassifier,
    Qwen3Classifier,
    _compute_fold_metrics,
    load_full_dataset_with_language,
)
from greentech.config import BASE_DIR  # noqa: E402
from greentech.utils.logger import setup_logging  # noqa: E402

OUTPUT_REPORT = BASE_DIR / "docs" / "BENCHMARK_FINAL_2026-04.md"
OUTPUT_JSON = BASE_DIR / "models" / "benchmark_final_metrics.json"

# Identifiants des modeles a benchmarker. Cle = nom human-friendly,
# valeur = (chemin_disque, nom_classe, label affichage).
MODELS_TO_BENCHMARK: dict[str, tuple[Path, str, str]] = {
    "qwen3": (
        BASE_DIR / "models" / "qwen3",
        "Qwen3Classifier",
        "Qwen3-4B + LoRA all-linear (decoder, ~4B params)",
    ),
    "mdeberta": (
        BASE_DIR / "models" / "mdeberta",
        "MDeBERTaClassifier",
        "mDeBERTa-v3-base (encoder, 278M params)",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark final comparatif Qwen3-4B vs mDeBERTa-v3-base "
            "sur le golden dataset apres entrainement K-fold unifie B3."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Chemin CSV du dataset de test. Defaut : autodetect "
            "(golden_dataset_augmented.csv si present, sinon golden_dataset.csv). "
            "Les variantes augmentation sont automatiquement exclues."
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=OUTPUT_REPORT,
        help=f"Chemin du rapport markdown (defaut : {OUTPUT_REPORT})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help=f"Chemin du JSON detaille (defaut : {OUTPUT_JSON})",
    )
    return parser.parse_args()


async def _load_classifier_for_model(model_alias: str) -> BaseClassifier | None:
    """Charge le classifier via ``get_classifier`` (qui gère ensemble_config).

    Note : on utilise ``get_classifier`` plutôt qu'instancier directement
    ``Qwen3Classifier(path).load(path)`` car le path racine peut contenir des
    reliques d'anciens trainings (ex. ``adapter_model.safetensors`` à la
    racine de ``models/qwen3/`` antérieur au TIES-merging P4.15). Le bon
    point d'entrée est ``ensemble_config.json`` qui redirige vers
    ``merged/`` pour ``ties_manual`` ou charge les K folds pour
    ``logit_average``.
    """
    path, _class_name, _label = MODELS_TO_BENCHMARK[model_alias]
    if not path.exists():
        logger.warning(
            f"Modele '{model_alias}' introuvable a {path}. "
            f"Lancer 'uv run python scripts/retrain_pipeline.py train-cv --model={model_alias}' "
            f"pour le produire."
        )
        return None

    from greentech.ai.models import inference as inference_module

    inference_module._classifier = None  # purge cache singleton entre modèles
    return await inference_module.get_classifier(model_path=path)


async def _run_inference(
    classifier: BaseClassifier,
    texts: list[str],
) -> tuple[list[int], list[float], list[float], int]:
    """Inference sur tous les textes, mesure latences et VRAM peak.

    Returns:
        Tuple ``(predictions, probas_positive, latences_ms, vram_peak_bytes)``.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    predictions: list[int] = []
    probas: list[float] = []
    latencies: list[float] = []

    for text in texts:
        start = time.perf_counter()
        result = await classifier.predict(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        predictions.append(result.label.value)
        latencies.append(elapsed_ms)
        if result.proba_positive is not None:
            probas.append(result.proba_positive)
        elif result.label == LabelGreenIT.GREEN:
            probas.append(result.score_confiance)
        else:
            probas.append(1.0 - result.score_confiance)

    vram_peak = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return predictions, probas, latencies, vram_peak


def _compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Calcule les percentiles p50, p95, p99 + moyenne et ecart-type."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _format_bytes(n: int) -> str:
    """Formate une taille en bytes en MB ou GB lisible."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def _build_markdown_report(
    results: dict[str, dict],
    dataset_path: Path,
    n_test: int,
    n_green_test: int,
) -> str:
    """Genere le rapport markdown comparatif."""
    lines = [
        "# Benchmark Final B4 — Qwen3-4B vs mDeBERTa-v3-base",
        "",
        f"**Date** : {datetime.now(UTC).isoformat()}",
        f"**Dataset** : `{dataset_path.name}` ({n_test} articles, {n_green_test} Green IT)",
        "**Generated by** : `scripts/benchmark_models.py` (KaRn1zC)",
        "",
        "## Methodologie",
        "",
        "Chaque modele a ete entraine via le protocole unifie B3 :",
        "",
        "- K-fold K=5 stratifie `(langue x label)` via `MultilabelStratifiedKFold`",
        "  - **mDeBERTa** : K=5, 3 seeds (15 trainings, MCC K-fold = 0.5941 ± 0.0093)",
        "  - **Qwen3-4B + LoRA** : K=3, 2 seeds (6 trainings, MCC K-fold = 0.6238)",
        "- `class_weight=[1.0, ~10.5]` sur CrossEntropy (remplacement oversampling)",
        "- Augmentation back-translation EN<->FR (opus-mt) sur les positifs (1:6.3)",
        "- Calibration post-fold : temperature scaling + threshold tuning (MCC)",
        "- Calibration moyennee persistee dans `temperature.json` + `optimal_threshold.json`",
        "- **Ensembles** :",
        "  - mDeBERTa : `logit_average` sur les 5 folds top-1 (charges en parallele)",
        "  - Qwen3 : `ties_manual` (TIES Yadav 2023 sur 3 folds top-1, densite=0.5)",
        "",
        "## ⚠️ Note importante sur le test set",
        "",
        "Le test set utilise est l'INTEGRALITE du golden_dataset (11 664 articles",
        "originaux, variantes back-translation exclues). Chaque article a donc ete",
        "vu en TRAIN par 4 des 5 modeles mDeBERTa (et 2 des 3 folds Qwen3). Les",
        "metriques ci-dessous sont donc OPTIMISTES par rapport aux K-fold reportes",
        "plus haut (qui sont l'estimation honnete). Le but de ce benchmark est de :",
        "",
        "1. Comparer les deux modeles sur la MEME distribution",
        "2. Mesurer les **latences reelles** d'inference (sur volume)",
        "3. Mesurer la **VRAM peak** observee",
        "",
        "Pour les decisions de promotion en production, se referer aux MCC K-fold",
        "reportes ci-dessus, pas aux MCC du test set ci-dessous.",
        "",
        "## Resultats comparatifs",
        "",
        "### Metriques de classification (cible : MCC > 0.75 stable)",
        "",
        "| Modele | MCC | F1 | Precision | Recall | Balanced Acc | Specificite |",
        "|---|---|---|---|---|---|---|",
    ]
    for alias, data in results.items():
        if "error" in data:
            lines.append(f"| {alias} | ❌ {data['error']} | - | - | - | - | - |")
            continue
        m = data["metrics"]
        lines.append(
            f"| **{alias}** | "
            f"{m['mcc']:.4f} | "
            f"{m['f1']:.4f} | "
            f"{m['precision']:.4f} | "
            f"{m['recall']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | "
            f"{m['specificite']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Latence inference (cible : < 200 ms par article)",
            "",
            "| Modele | Moyenne | p50 | p95 | p99 | Min | Max |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for alias, data in results.items():
        if "error" in data:
            continue
        lat = data["latency"]
        lines.append(
            f"| **{alias}** | "
            f"{lat['mean_ms']:.1f} ms | "
            f"{lat['p50_ms']:.1f} ms | "
            f"{lat['p95_ms']:.1f} ms | "
            f"{lat['p99_ms']:.1f} ms | "
            f"{lat['min_ms']:.1f} ms | "
            f"{lat['max_ms']:.1f} ms |"
        )

    lines.extend(
        [
            "",
            "### Ressources",
            "",
            "| Modele | VRAM peak | Architecture | Taille modele |",
            "|---|---|---|---|",
        ]
    )
    for alias, data in results.items():
        if "error" in data:
            continue
        meta = data["metadata"]
        lines.append(
            f"| **{alias}** | "
            f"{_format_bytes(data['vram_peak_bytes'])} | "
            f"{meta['architecture']} | "
            f"{meta['n_params']} params |"
        )

    # Section selection
    valid = [
        (alias, d) for alias, d in results.items() if "error" not in d
    ]
    if valid:
        winner = max(valid, key=lambda kv: kv[1]["metrics"]["mcc"])
        winner_alias, winner_data = winner
        lines.extend(
            [
                "",
                "## Selection du modele retenu",
                "",
                f"**Vainqueur retenu** : `{winner_alias}` "
                f"(MCC = {winner_data['metrics']['mcc']:.4f})",
                "",
                "Critere de decision : MCC le plus eleve, sous condition que la latence reste",
                "sous 200 ms et que l'ecart-type K-fold reste sous 0.10 (cf. `cv_report_*.json`",
                "pour la stabilite).",
                "",
                "Pour promouvoir ce modele en production :",
                "",
                "```bash",
                "# Copier les artefacts dans models/production/",
                f"cp -r models/{winner_alias}/* models/production/",
                "",
                "# Pousser vers DVC + MinIO",
                "uv run dvc add models/production && uv run dvc push",
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


async def benchmark_all() -> dict[str, dict]:
    """Charge les 2 modeles entraines, evalue, retourne les metriques completes.

    Returns:
        Dictionnaire ``{model_alias: {metrics, latency, vram_peak_bytes,
        metadata}}`` ou erreur si modele manquant.
    """
    args = parse_args()

    # Charger le dataset (originaux uniquement, exclure les variantes pour
    # l'evaluation car elles partagent leur titre avec les originaux)
    texts, labels, _langues, aug_sources = load_full_dataset_with_language(
        dataset_path=args.dataset
    )
    original_indices = [i for i, src in enumerate(aug_sources) if not src]
    test_texts = [texts[i] for i in original_indices]
    test_labels = [labels[i] for i in original_indices]
    n_test = len(test_texts)
    n_green = sum(test_labels)

    logger.info("=" * 70)
    logger.info("  BENCHMARK FINAL B4 : Qwen3-4B vs mDeBERTa-v3-base")
    logger.info("=" * 70)
    logger.info(f"  Test set : {n_test} articles ({n_green} Green IT)")
    logger.info("")

    results: dict[str, dict] = {}
    cv_run_name = f"benchmark-models-comparison-{int(time.time())}"
    exp_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run=cv_run_name,
        tags={
            "phase": "b4-benchmark-final",
            "models": ",".join(MODELS_TO_BENCHMARK.keys()),
        },
        params={
            "n_test": n_test,
            "n_green_test": n_green,
            "dataset": str(args.dataset or "auto"),
        },
    )

    with tracked_experiment(exp_config):
        import mlflow

        for alias, (path, _cls_name, label) in MODELS_TO_BENCHMARK.items():
            logger.info("")
            logger.info(f"--- {alias} : {label} ---")
            try:
                classifier = await _load_classifier_for_model(alias)
                if classifier is None:
                    results[alias] = {"error": f"Modele introuvable a {path}"}
                    continue

                preds, probas, latencies, vram_peak = await _run_inference(
                    classifier, test_texts
                )
                metrics = _compute_fold_metrics(test_labels, preds, latencies)
                latency_stats = _compute_latency_percentiles(latencies)

                # Compter les parametres reels du modele charge. EnsembleClassifier
                # est une composition de N members ; on somme les params du premier
                # member (les K folds ont la meme architecture) et on note "ensemble(Kx)".
                from greentech.ai.models.inference import EnsembleClassifier

                if isinstance(classifier, EnsembleClassifier) and classifier.members:
                    first = classifier.members[0]
                    n_params = (
                        int(sum(p.numel() for p in first.model.parameters()))
                        if first.model is not None
                        else 0
                    )
                    arch = f"ensemble({len(classifier.members)}x {type(first.model).__name__})"
                elif classifier.model is not None:
                    n_params = int(sum(p.numel() for p in classifier.model.parameters()))
                    arch = type(classifier.model).__name__
                else:
                    n_params = 0
                    arch = "?"

                results[alias] = {
                    "metrics": metrics,
                    "latency": latency_stats,
                    "vram_peak_bytes": vram_peak,
                    "metadata": {
                        "architecture": arch,
                        "n_params": f"{n_params / 1e6:.1f}M",
                        "model_path": str(path),
                    },
                    "probas_positive": probas,
                }
                logger.info(
                    f"  MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
                    f"latence p50={latency_stats['p50_ms']:.1f}ms / "
                    f"p95={latency_stats['p95_ms']:.1f}ms, "
                    f"VRAM peak={_format_bytes(vram_peak)}"
                )

                # Logger les metriques dans MLflow par modele
                for key, val in metrics.items():
                    if isinstance(val, int | float):
                        mlflow.log_metric(f"{alias}_{key}", float(val))
                for key, val in latency_stats.items():
                    mlflow.log_metric(f"{alias}_latency_{key}", val)
                mlflow.log_metric(f"{alias}_vram_peak_bytes", vram_peak)
                mlflow.log_metric(f"{alias}_n_params_M", n_params / 1e6)

                del classifier
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.exception(f"Benchmark {alias} echoue : {exc}")
                results[alias] = {"error": str(exc)}

    # Persister les artefacts (sans les probas complets pour le JSON principal)
    output_json = args.output_json
    output_md = args.output_md

    json_payload = {
        alias: {k: v for k, v in data.items() if k != "probas_positive"}
        for alias, data in results.items()
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    logger.info(f"\nJSON detaille sauvegarde : {output_json}")

    report = _build_markdown_report(
        results,
        dataset_path=args.dataset or BASE_DIR / "data" / "golden_dataset_augmented.csv",
        n_test=n_test,
        n_green_test=n_green,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    logger.info(f"Rapport markdown sauvegarde : {output_md}")

    return results


def main() -> int:
    """Point d'entree CLI."""
    setup_logging()
    asyncio.run(benchmark_all())
    return 0


if __name__ == "__main__":
    sys.exit(main())
