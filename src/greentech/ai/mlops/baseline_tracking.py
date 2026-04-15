"""Tracking unifie des resultats de baseline (JSON + MLflow + Prometheus).

Factorise la logique de persistance des metriques baseline pour les trois
destinations utilisees par le projet :

1. **Fichier JSON** (``models/baseline_metrics.json``) — reference locale
   utilisee par le pipeline de benchmark pour comparer le gain du
   fine-tuning.
2. **MLflow** — run tagge ``type=baseline`` dans l'experience
   ``greentech-classification``, avec log des emissions carbone via
   CodeCarbon.
3. **Pushgateway Prometheus** — expose les metriques baseline aux
   dashboards Grafana (panel "Baseline" dans ``pipeline-training``).

Ce module est importe par ``scripts/benchmark_baseline.py`` et par
``scripts/retrain_pipeline.py`` (step baseline) pour garantir que les
deux points d'entree produisent exactement la meme tracabilite.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from greentech.ai.models.baseline import BaselineResult


def save_baseline_metrics_json(
    result: BaselineResult,
    output_file: Path,
) -> None:
    """Persiste les metriques baseline dans un fichier JSON.

    Le fichier sert de reference permanente pour le script de benchmark
    (``scripts/retrain_pipeline.py benchmark``) qui calcule le gain
    apporte par le fine-tuning vs cette baseline.

    Args:
        result: Resultat complet de l'evaluation baseline.
        output_file: Chemin du fichier JSON a ecrire.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": result.model_name,
        "date": datetime.now(UTC).isoformat(),
        "evaluation_scope": "full_dataset",
        "n_articles": result.n_articles,
        "duration_seconds": result.duration_seconds,
        "metrics": dict(result.metrics),
    }
    output_file.write_text(json.dumps(payload, indent=2))
    logger.info(
        f"Baseline persistee : {output_file} "
        f"(MCC={result.metrics['mcc']:.4f}, F1={result.metrics['f1']:.4f}, "
        f"Recall={result.metrics['recall']:.4f})"
    )


def log_baseline_to_mlflow(result: BaselineResult) -> None:
    """Logge la baseline dans un run MLflow dedie.

    Le run est tagge ``type=baseline`` pour etre filtrable dans l'UI
    MLflow et facilement compare aux runs d'entrainement LoRA
    correspondants. L'empreinte carbone est mesuree via CodeCarbon et
    attachee au run.

    Args:
        result: Resultat complet de l'evaluation baseline.

    Raises:
        Exception: Toute erreur de connexion MLflow ou d'upload vers
            MinIO. Les appelants doivent la capturer si une defaillance
            du serveur de tracking ne doit pas invalider l'execution.
    """
    import mlflow

    from greentech.ai.mlops.tracking import ExperimentConfig, tracked_experiment

    safe_name = result.model_name.replace("/", "_").replace(".", "_")
    exp_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run=f"baseline-{safe_name}",
        tags={
            "type": "baseline",
            "modele": result.model_name,
            "method": "zero-shot",
            "evaluation_scope": "full_dataset",
        },
        params={
            "model": result.model_name,
            "method": "zero-shot",
            "n_articles": result.n_articles,
            "duration_seconds": round(result.duration_seconds, 2),
        },
        mesurer_carbone=True,
    )

    with tracked_experiment(exp_config):
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in result.metrics.items()
                if isinstance(v, (int, float))
            }
        )


def push_baseline_to_prometheus(result: BaselineResult) -> None:
    """Pousse les metriques baseline vers le Pushgateway Prometheus.

    Les dashboards Grafana peuvent alors afficher la baseline comme
    reference immuable a cote des courbes de training K-fold. Les
    erreurs reseau sont silencieuses (Pushgateway indisponible = warning
    log) pour ne jamais faire echouer le pipeline.

    Args:
        result: Resultat complet de l'evaluation baseline.
    """
    from greentech.ai.mlops.prometheus_metrics import record_baseline_metrics

    try:
        record_baseline_metrics(
            model_name=result.model_name,
            metrics=dict(result.metrics),
            n_articles=result.n_articles,
            duration_seconds=result.duration_seconds,
        )
    except Exception as exc:
        logger.warning(f"Push Prometheus baseline echoue : {exc}")


def track_baseline(result: BaselineResult, json_path: Path) -> None:
    """Orchestre la persistance triple (JSON + MLflow + Prometheus).

    Chaque destination est independante : un echec sur MLflow ou
    Prometheus n'empeche pas la sauvegarde JSON locale, critique pour
    le benchmark futur. MLflow et Prometheus sont appeles avec gestion
    d'exception pour ne jamais invalider un run baseline si les services
    tracking sont indisponibles.

    Args:
        result: Resultat complet de l'evaluation baseline.
        json_path: Chemin du fichier JSON de reference locale.
    """
    save_baseline_metrics_json(result, json_path)

    try:
        log_baseline_to_mlflow(result)
    except Exception as exc:
        logger.warning(f"MLflow indisponible, baseline non trackee : {exc}")

    push_baseline_to_prometheus(result)
