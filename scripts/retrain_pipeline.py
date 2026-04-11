"""Pipeline complet de re-collecte, re-annotation et re-entrainement.

Orchestre toutes les etapes pour agrandir le corpus, re-annoter le dataset,
re-entrainer le modele Llama 3.2 3B + LoRA, benchmarker la nouvelle version
contre la meilleure version historique et le modele de base, puis promouvoir
automatiquement en production UNIQUEMENT si la nouvelle version est meilleure.

Le pipeline garantit que l'application utilise toujours la meilleure version
entrainee du modele, jamais une regression.

Usage:
    # Pipeline complet (collecte, annotation, entrainement, benchmark, promotion auto)
    uv run python scripts/retrain_pipeline.py

    # Etapes individuelles
    uv run python scripts/retrain_pipeline.py collect       # Collecte depuis les sources
    uv run python scripts/retrain_pipeline.py annotate      # Re-annotation du dataset
    uv run python scripts/retrain_pipeline.py train         # Re-entrainement Llama
    uv run python scripts/retrain_pipeline.py benchmark     # Benchmark nouveau vs production vs baseline
    uv run python scripts/retrain_pipeline.py promote       # Promotion manuelle (force)
    uv run python scripts/retrain_pipeline.py baseline      # Calculer les metriques du modele de base

    # Ingerer un fichier manuellement
    uv run python scripts/retrain_pipeline.py ingest-file data/mon_fichier.json

    # Combiner des etapes
    uv run python scripts/retrain_pipeline.py collect annotate train

"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers de modeles
MODELS_DIR = BASE_DIR / "models"
PRODUCTION_DIR = MODELS_DIR / "production"
VERSIONS_DIR = MODELS_DIR / "versions"
LLAMA_TRAIN_DIR = MODELS_DIR / "challenger-llama"

# Fichiers de reference pour les metriques
BEST_METRICS_FILE = MODELS_DIR / "best_metrics.json"
BASELINE_METRICS_FILE = MODELS_DIR / "baseline_metrics.json"


# =============================================================================
# UTILITAIRES
# =============================================================================


def _run_module(module: str, *args: str) -> bool:
    """Execute un module Python via uv run.

    Args:
        module: Chemin du module Python.
        *args: Arguments supplementaires.

    Returns:
        True si l'execution a reussi.
    """
    cmd = ["uv", "run", "python", "-m", module, *args]
    logger.info(f"Execution : {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=False)
    if result.returncode != 0:
        logger.error(f"Echec de {module} (code {result.returncode})")
        return False
    return True


def _run_script(script: str, *args: str) -> bool:
    """Execute un script Python via uv run.

    Args:
        script: Chemin relatif du script.
        *args: Arguments supplementaires.

    Returns:
        True si l'execution a reussi.
    """
    cmd = ["uv", "run", "python", str(BASE_DIR / script), *args]
    logger.info(f"Execution : {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=False)
    if result.returncode != 0:
        logger.error(f"Echec de {script} (code {result.returncode})")
        return False
    return True


def _generate_version_tag() -> str:
    """Genere un tag de version horodate."""
    return datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict | None:
    """Charge un fichier JSON s'il existe."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_best_metrics(metrics: dict, version_tag: str) -> None:
    """Sauvegarde les metriques de la meilleure version connue."""
    data = {
        "version": version_tag,
        "date": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    BEST_METRICS_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"Meilleur modele enregistre : {version_tag} (F1={metrics['f1']:.4f})")


# =============================================================================
# COLLECTE
# =============================================================================


def step_collect() -> bool:
    """Re-collecte des donnees depuis les sources configurees."""
    logger.info("=" * 70)
    logger.info("  COLLECTE DES DONNEES")
    logger.info("=" * 70)

    ok = True

    logger.info("\n--- Collecte API (NewsData.io) ---")
    if not _run_module("greentech.data.collectors.api_collector"):
        logger.warning("Collecte API echouee (cle API manquante ?), on continue")

    logger.info("\n--- Scraping web (TechCrunch) ---")
    if not _run_module("greentech.data.collectors.scraper"):
        logger.warning("Scraping echoue, on continue")

    logger.info("\n--- Nettoyage Spark ---")
    if not _run_module("greentech.data.processors.spark_cleaner"):
        logger.error("Nettoyage Spark echoue")
        ok = False

    if ok:
        logger.info("\n--- Ingestion PostgreSQL ---")
        if not _run_module("greentech.data.storage.sql_ingester"):
            logger.error("Ingestion SQL echouee")
            ok = False

    return ok


def step_ingest_file(file_path: str) -> bool:
    """Ingere un fichier local puis nettoie et ingere dans PostgreSQL."""
    logger.info("=" * 70)
    logger.info(f"  INGESTION FICHIER : {file_path}")
    logger.info("=" * 70)

    path = Path(file_path)
    if not path.exists():
        logger.error(f"Fichier introuvable : {file_path}")
        return False

    if not _run_module("greentech.data.collectors.file_ingester", str(path)):
        return False

    if not _run_module("greentech.data.processors.spark_cleaner"):
        return False
    return _run_module("greentech.data.storage.sql_ingester")


# =============================================================================
# ANNOTATION
# =============================================================================


def step_annotate() -> bool:
    """Re-annote l'integralite du corpus et regenere golden_dataset.csv."""
    logger.info("=" * 70)
    logger.info("  RE-ANNOTATION DU GOLDEN DATASET")
    logger.info("=" * 70)
    return _run_script("scripts/auto_annotate_dataset.py")


# =============================================================================
# ARCHIVAGE
# =============================================================================


def step_archive_current_production() -> str | None:
    """Archive le modele de production actuel dans models/versions/."""
    if not PRODUCTION_DIR.exists() or not (PRODUCTION_DIR / "adapter_config.json").exists():
        logger.info("Pas de modele de production a archiver")
        return None

    version_tag = _generate_version_tag()
    archive_dir = VERSIONS_DIR / version_tag
    archive_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Archivage du modele actuel vers {archive_dir}")
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, archive_dir / f.name)

    best = _load_json(BEST_METRICS_FILE)
    metadata = {
        "version": version_tag,
        "date_archive": datetime.now(UTC).isoformat(),
        "fichiers": [f.name for f in archive_dir.iterdir() if f.is_file()],
        "metrics": best["metrics"] if best else None,
    }
    (archive_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"Archive creee : {version_tag}")
    return version_tag


# =============================================================================
# ENTRAINEMENT
# =============================================================================


def step_train() -> bool:
    """Re-entraine Llama 3.2 3B + LoRA sur le dataset courant."""
    logger.info("=" * 70)
    logger.info("  RE-ENTRAINEMENT LLAMA 3.2 3B + LoRA")
    logger.info("=" * 70)
    return _run_module("greentech.ai.models.training", "challenger-llama")


# =============================================================================
# BASELINE (modele de base sans fine-tuning)
# =============================================================================


async def step_baseline() -> dict[str, float]:
    """Calcule les metriques du modele de base SANS fine-tuning (reference permanente)."""
    existing = _load_json(BASELINE_METRICS_FILE)
    if existing:
        logger.info(f"Baseline deja calculee (F1={existing['metrics']['f1']:.4f}), reutilisation")
        return existing["metrics"]

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from greentech.ai.models.training import load_golden_dataset
    from greentech.config import get_settings

    logger.info("=" * 70)
    logger.info("  CALCUL BASELINE : Llama 3.2 3B SANS fine-tuning")
    logger.info("=" * 70)

    _, _, test_texts, test_labels = load_golden_dataset(oversample=False)
    hf_token = get_settings().huggingface_token or None
    model_name = "meta-llama/Llama-3.2-3B"

    logger.info(f"Chargement de {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds = []
    latencies = []
    for text in test_texts:
        start = time.perf_counter()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        preds.append(torch.softmax(outputs.logits, dim=-1).argmax(dim=-1).item())
        latencies.append(int((time.perf_counter() - start) * 1000))

    metrics = {
        "f1": f1_score(test_labels, preds, average="binary", zero_division=0),
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, average="binary", zero_division=0),
        "recall": recall_score(test_labels, preds, average="binary", zero_division=0),
        "latence_moyenne_ms": float(np.mean(latencies)),
    }

    data = {"model": model_name, "date": datetime.now(UTC).isoformat(), "metrics": metrics}
    BASELINE_METRICS_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"Baseline sauvegardee : F1={metrics['f1']:.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# =============================================================================
# BENCHMARK (nouveau vs meilleur historique vs baseline)
# =============================================================================


async def _evaluate_new_model(
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, float]:
    """Evalue le modele fraichement entraine sur le test set."""
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    from greentech.ai.models.classifier import TrainingConfig
    from greentech.ai.models.training import ChallengerClassifier

    classifier = ChallengerClassifier(
        config=TrainingConfig(nom_modele="meta-llama/Llama-3.2-3B", output_dir=LLAMA_TRAIN_DIR),
    )
    classifier.load(LLAMA_TRAIN_DIR)

    preds = []
    latencies = []
    for text in test_texts:
        pred = await classifier.predict(text)
        preds.append(pred.label.value)
        latencies.append(pred.temps_ms)

    metrics = {
        "f1": f1_score(test_labels, preds, average="binary", zero_division=0),
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, average="binary", zero_division=0),
        "recall": recall_score(test_labels, preds, average="binary", zero_division=0),
        "latence_moyenne_ms": float(np.mean(latencies)),
    }

    logger.info(f"  F1={metrics['f1']:.4f}  Accuracy={metrics['accuracy']:.4f}")
    logger.info(
        "\n"
        + classification_report(
            test_labels,
            preds,
            target_names=["Non Green IT", "Green IT"],
        )
    )

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


async def step_benchmark() -> dict | None:
    """Compare le nouveau modele au meilleur historique et au baseline.

    Returns:
        Resultats avec verdict (nouveau_est_meilleur: bool), ou None si erreur.
    """
    from greentech.ai.models.training import load_golden_dataset

    logger.info("=" * 70)
    logger.info("  BENCHMARK : NOUVEAU vs MEILLEUR HISTORIQUE vs BASELINE")
    logger.info("=" * 70)

    if not LLAMA_TRAIN_DIR.exists() or not (LLAMA_TRAIN_DIR / "adapter_config.json").exists():
        logger.error(f"Modele entraine introuvable dans {LLAMA_TRAIN_DIR}")
        return None

    _, _, test_texts, test_labels = load_golden_dataset(oversample=False)
    logger.info(f"Test set : {len(test_texts)} articles")

    logger.info("\nNouveau modele (post-entrainement) :")
    new_metrics = await _evaluate_new_model(test_texts, test_labels)

    best_ref = _load_json(BEST_METRICS_FILE)
    baseline_ref = _load_json(BASELINE_METRICS_FILE)

    # Rapport
    logger.info("\n" + "=" * 80)
    logger.info("  RAPPORT COMPARATIF")
    logger.info("=" * 80)

    if baseline_ref:
        b = baseline_ref["metrics"]
        gain = new_metrics["f1"] - b["f1"]
        logger.info(f"\n  Baseline (sans fine-tuning) : F1={b['f1']:.4f}")
        logger.info(f"  Gain du fine-tuning         : {'+' if gain >= 0 else ''}{gain:.4f} F1")

    if best_ref:
        old = best_ref["metrics"]
        delta = new_metrics["f1"] - old["f1"]
        logger.info(f"\n  Meilleur precedent ({best_ref['version']}) : F1={old['f1']:.4f}")
        logger.info(f"  Delta                       : {'+' if delta >= 0 else ''}{delta:.4f} F1")
    else:
        delta = 1.0

    logger.info(f"\n  Nouveau modele              : F1={new_metrics['f1']:.4f}")

    nouveau_est_meilleur = delta >= 0

    logger.info("\n" + "-" * 80)
    if nouveau_est_meilleur:
        logger.info("  VERDICT : NOUVEAU MODELE RETENU")
    else:
        logger.warning("  VERDICT : ANCIEN MODELE CONSERVE (regression)")
    logger.info("=" * 80)

    report = {
        "date": datetime.now(UTC).isoformat(),
        "test_set_size": len(test_texts),
        "nouveau": new_metrics,
        "meilleur_precedent": best_ref["metrics"] if best_ref else None,
        "baseline": baseline_ref["metrics"] if baseline_ref else None,
        "verdict": "nouveau_retenu" if nouveau_est_meilleur else "ancien_conserve",
    }
    report_path = BASE_DIR / "data" / "benchmark_versions.json"
    report_path.write_text(json.dumps(report, indent=2))

    return {"nouveau": new_metrics, "nouveau_est_meilleur": nouveau_est_meilleur}


# =============================================================================
# PROMOTION CONDITIONNELLE
# =============================================================================


async def step_auto_promote() -> bool:
    """Benchmark + promotion UNIQUEMENT si le nouveau modele est meilleur."""
    results = await step_benchmark()
    if results is None:
        logger.error("Benchmark echoue, pas de promotion")
        return False

    if not results["nouveau_est_meilleur"]:
        logger.info("ANCIEN MODELE CONSERVE EN PRODUCTION (pas de regression)")
        return True

    new_metrics = results["nouveau"]
    version_tag = _generate_version_tag()

    step_archive_current_production()

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file() and f.name != "README.md":
            f.unlink()

    fichiers = []
    for f in LLAMA_TRAIN_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, PRODUCTION_DIR / f.name)
            fichiers.append(f.name)

    _save_best_metrics(new_metrics, version_tag)

    meta = {
        "promoted_at": datetime.now(UTC).isoformat(),
        "version": version_tag,
        "metrics": new_metrics,
        "fichiers": fichiers,
    }
    (PRODUCTION_DIR / "promotion_info.json").write_text(json.dumps(meta, indent=2))

    logger.info(f"NOUVEAU MODELE PROMU : {version_tag} (F1={new_metrics['f1']:.4f})")
    logger.info("Redemarrez l'API pour charger le nouveau modele.")
    return True


def step_force_promote() -> bool:
    """Promotion manuelle forcee, sans benchmark."""
    logger.info("=" * 70)
    logger.info("  PROMOTION FORCEE")
    logger.info("=" * 70)

    source = LLAMA_TRAIN_DIR
    if not source.exists() or not (source / "adapter_config.json").exists():
        logger.error(f"Modele introuvable dans {source}")
        return False

    step_archive_current_production()

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file() and f.name != "README.md":
            f.unlink()

    fichiers = []
    for f in source.iterdir():
        if f.is_file():
            shutil.copy2(f, PRODUCTION_DIR / f.name)
            fichiers.append(f.name)

    (PRODUCTION_DIR / "promotion_info.json").write_text(
        json.dumps(
            {
                "promoted_at": datetime.now(UTC).isoformat(),
                "forced": True,
                "fichiers": fichiers,
            },
            indent=2,
        )
    )

    logger.info(f"Modele promu de force : {len(fichiers)} fichiers")
    return True


# =============================================================================
# ORCHESTRATION
# =============================================================================


async def run_pipeline(steps: list[str]) -> None:
    """Execute les etapes du pipeline dans l'ordre demande."""
    start = time.perf_counter()
    logger.info("")
    logger.info("#" * 70)
    logger.info("#  PIPELINE DE RE-ENTRAINEMENT — GreenTech Intelligence")
    logger.info(f"#  Etapes : {', '.join(steps)}")
    logger.info(f"#  Date   : {datetime.now(UTC).isoformat()}")
    logger.info("#" * 70)
    logger.info("")

    for step_name in steps:
        if step_name.startswith("ingest-file:"):
            file_path = step_name.split(":", 1)[1]
            if not step_ingest_file(file_path):
                logger.error(f"Interrompu a ingest-file ({file_path})")
                return
            continue

        if step_name == "benchmark":
            await step_benchmark()
            continue
        if step_name == "auto-promote":
            await step_auto_promote()
            continue
        if step_name == "baseline":
            await step_baseline()
            continue

        sync_steps = {
            "collect": ("Collecte des donnees", step_collect),
            "annotate": ("Re-annotation du dataset", step_annotate),
            "train": ("Re-entrainement Llama", step_train),
            "promote": ("Promotion forcee", step_force_promote),
        }

        if step_name not in sync_steps:
            valid = list(sync_steps.keys()) + ["benchmark", "auto-promote", "baseline"]
            logger.error(f"Etape inconnue : {step_name}")
            logger.info(f"Valides : {', '.join(valid)}, ingest-file <path>")
            return

        label, func = sync_steps[step_name]
        logger.info(f"\n>>> {label}...")
        if not func():
            logger.error(f"Interrompu a : {step_name}")
            return

    elapsed = time.perf_counter() - start
    logger.info("")
    logger.info("#" * 70)
    logger.info(f"#  PIPELINE TERMINE en {int(elapsed // 60)}m {int(elapsed % 60)}s")
    logger.info("#" * 70)


def main() -> None:
    """Point d'entree CLI."""
    args = sys.argv[1:]

    processed: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "ingest-file" and i + 1 < len(args):
            processed.append(f"ingest-file:{args[i + 1]}")
            i += 2
        else:
            processed.append(args[i])
            i += 1

    if not processed:
        processed = ["collect", "annotate", "train", "auto-promote"]

    asyncio.run(run_pipeline(processed))


if __name__ == "__main__":
    main()
