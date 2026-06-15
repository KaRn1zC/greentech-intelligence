"""Rejeu des metriques d'entrainement et de baseline (MLflow -> Pushgateway).

Pendant de ``training_emissions_exporter`` pour les metriques de qualite. Les
agreges de cross-validation (``cv_mcc_mean`` / ``cv_mcc_std`` par run) et les
metriques des baselines (``mcc`` / ``f1`` / ``recall``...) sont logues dans
MLflow par ``training.py``, mais Prometheus ne lit pas MLflow. Apres une purge
du Pushgateway ou l'expiration de l'ancienne retention 15 jours, le dashboard
``pipeline-training`` se retrouvait donc vide alors que les donnees existent.

Ce module rejoue ces metriques vers le Pushgateway, sous les **memes jobs** que
le training live (``greentech-training`` pour la CV, ``greentech-baseline`` pour
les baselines), afin qu'un futur entrainement remplace proprement le groupe sans
creer de doublon.

Limite assumee : les metriques **par fold** (``greentech_training_fold_*``) et
la **memoire GPU** ne sont pas rejouees -- elles ne sont pleinement disponibles
que pendant un run live. Les panneaux correspondants se rempliront au prochain
entrainement reel.

Usage : ``uv run python scripts/export_training_metrics.py``
"""

from __future__ import annotations

import os
import time

from loguru import logger
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from greentech.ai.mlops.training_emissions_exporter import (
    _EXPERIMENT_NAME,
    _detect_model_type,
    _detect_phase,
    _normalize_pandas_row,
)

# Defaut local (worker/scripts hors Docker) ; override via env en full Docker.
_PUSHGATEWAY_URL = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", "localhost:9091")

# Registres dedies : on pousse la CV sous le job ``greentech-training`` et les
# baselines sous ``greentech-baseline``, sans embarquer le reste du registre
# global (emissions, folds en cours...). Les noms de metriques sont identiques
# a ceux de ``prometheus_metrics.py`` pour que les dashboards les retrouvent.
_CV_REGISTRY = CollectorRegistry()
_BASELINE_REGISTRY = CollectorRegistry()

# Agreges cross-validation rejoues par run. On en expose davantage que le live
# (qui ne pousse que le MCC) afin de remplir le dashboard avec la qualite reelle
# de chaque modele : F1, Recall et Precision Green IT sont les metriques qui
# parlent au jury (le MCC seul ne suffit pas a raconter le comportement du
# classifieur sur la classe minoritaire).
_CV_MEAN_GAUGES: dict[str, Gauge] = {
    "mcc": Gauge(
        "greentech_training_cv_mcc_mean",
        "Moyenne MCC K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
    "f1": Gauge(
        "greentech_training_cv_f1_mean",
        "Moyenne F1 Green IT K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
    "recall": Gauge(
        "greentech_training_cv_recall_mean",
        "Moyenne Recall Green IT K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
    "precision": Gauge(
        "greentech_training_cv_precision_mean",
        "Moyenne Precision Green IT K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
    "accuracy": Gauge(
        "greentech_training_cv_accuracy_mean",
        "Moyenne Accuracy K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
    "balanced_accuracy": Gauge(
        "greentech_training_cv_balanced_accuracy_mean",
        "Moyenne Balanced Accuracy K-fold (rejeu MLflow)",
        ["model_type", "run_name"],
        registry=_CV_REGISTRY,
    ),
}
_cv_mcc_std = Gauge(
    "greentech_training_cv_mcc_std",
    "Ecart-type du MCC K-fold (rejeu MLflow)",
    ["model_type", "run_name"],
    registry=_CV_REGISTRY,
)
_cv_last_push = Gauge(
    "greentech_training_last_push_timestamp_seconds",
    "Horodatage Unix du dernier rejeu CV",
    ["model_type", "run_name"],
    registry=_CV_REGISTRY,
)

_BASELINE_GAUGES: dict[str, Gauge] = {
    "mcc": Gauge(
        "greentech_baseline_mcc",
        "MCC baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
    "f1": Gauge(
        "greentech_baseline_f1",
        "F1 Green IT baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
    "recall": Gauge(
        "greentech_baseline_recall",
        "Recall Green IT baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
    "precision": Gauge(
        "greentech_baseline_precision",
        "Precision Green IT baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
    "accuracy": Gauge(
        "greentech_baseline_accuracy",
        "Accuracy baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
    "balanced_accuracy": Gauge(
        "greentech_baseline_balanced_accuracy",
        "Balanced accuracy baseline (rejeu MLflow)",
        ["model"],
        registry=_BASELINE_REGISTRY,
    ),
}
_bl_last_push = Gauge(
    "greentech_baseline_last_push_timestamp_seconds",
    "Horodatage Unix du dernier rejeu baseline",
    ["model"],
    registry=_BASELINE_REGISTRY,
)


def _collect_runs(experiment_name: str = _EXPERIMENT_NAME) -> list[dict]:
    """Recupere tous les runs de l'experience MLflow, normalises en dicts."""
    import mlflow

    from greentech.ai.mlops.tracking import configure_mlflow

    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experience MLflow introuvable : {experiment_name}")
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=1,  # ACTIVE_ONLY
        max_results=10_000,
        output_format="pandas",
    )
    if runs.empty:
        logger.info(f"Aucun run MLflow trouve dans {experiment_name}")
        return []
    return [_normalize_pandas_row(row) for _, row in runs.iterrows()]


def export_training_metrics(experiment_name: str = _EXPERIMENT_NAME) -> dict[str, int]:
    """Republie les agreges CV et les baselines depuis MLflow vers le Pushgateway.

    Idempotent : chaque push remplace integralement le groupe de son job, donc
    on peut relancer la fonction (script CLI ou demarrage de l'API) sans cumuler.

    Returns:
        ``{"cv": n_runs_cv, "baseline": n_baselines}`` pour le logging appelant.
    """
    runs = _collect_runs(experiment_name)
    now = time.time()
    n_cv = 0
    n_baseline = 0

    for run in runs:
        metrics = run["metrics"]
        tags = run["tags"]
        params = run["params"]
        run_name = tags.get("mlflow.runName") or run.get("run_name") or "(sans nom)"
        model_type = _detect_model_type(
            tags.get("model_type"), params.get("model_type"), params.get("model"), run_name
        )

        if "cv_mcc_mean" in metrics:
            for key, gauge in _CV_MEAN_GAUGES.items():
                mlflow_key = f"cv_{key}_mean"
                if mlflow_key in metrics:
                    gauge.labels(model_type=model_type, run_name=run_name).set(metrics[mlflow_key])
            if "cv_mcc_std" in metrics:
                _cv_mcc_std.labels(model_type=model_type, run_name=run_name).set(
                    metrics["cv_mcc_std"]
                )
            _cv_last_push.labels(model_type=model_type, run_name=run_name).set(now)
            n_cv += 1
            continue

        if _detect_phase(run_name, tags) == "baseline" and "mcc" in metrics:
            model = (
                params.get("model")
                or params.get("base_model")
                or params.get("model_name")
                or run_name
            )
            for key, gauge in _BASELINE_GAUGES.items():
                if key in metrics:
                    gauge.labels(model=model).set(metrics[key])
            _bl_last_push.labels(model=model).set(now)
            n_baseline += 1

    try:
        push_to_gateway(_PUSHGATEWAY_URL, job="greentech-training", registry=_CV_REGISTRY)
        push_to_gateway(_PUSHGATEWAY_URL, job="greentech-baseline", registry=_BASELINE_REGISTRY)
    except Exception as exc:
        logger.warning(f"Push des metriques training echoue ({_PUSHGATEWAY_URL}) : {exc}")

    logger.info(
        f"Rejeu metriques training : {n_cv} run(s) CV + {n_baseline} baseline(s) republie(s)."
    )
    return {"cv": n_cv, "baseline": n_baseline}


if __name__ == "__main__":
    export_training_metrics()
