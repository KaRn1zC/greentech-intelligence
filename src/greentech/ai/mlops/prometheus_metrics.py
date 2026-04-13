"""Metriques Prometheus pour les jobs de re-entrainement ML.

Les scripts d'entrainement tournent hors conteneur (sur l'hote qui dispose
du GPU AMD ROCm), donc Prometheus ne peut pas les scraper directement.
Le pattern utilise est celui du Pushgateway : chaque etape pousse son
snapshot vers `pushgateway:9091`, Prometheus scrape cette passerelle et
les dashboards Grafana consomment les metriques en temps reel.

L'ensemble des metriques est regroupe dans un `CollectorRegistry` local
afin de pouvoir le pousser d'un bloc, sans polluer le registre global.
"""

from __future__ import annotations

import os
import time

from loguru import logger
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# URL du Pushgateway : override-able via env pour les jobs hors Docker
# (le script est lance depuis Windows, pushgateway est expose sur localhost:9091).
_PUSHGATEWAY_URL = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", "localhost:9091")

# Registre local : les metriques poussees ne polluent pas le registre global
# du process et peuvent etre remises a zero entre deux jobs independants.
_REGISTRY = CollectorRegistry()

# Jauges synthetisant l'etat d'un run d'entrainement. On privilegie des
# Gauge (plutot que Counter) pour que Grafana affiche la valeur absolue
# courante du fold sans agregation exotique.
training_fold_current = Gauge(
    "greentech_training_fold_current",
    "Numero du fold en cours (0 avant demarrage, K a la fin)",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_fold_total = Gauge(
    "greentech_training_fold_total",
    "Nombre total de folds prevus dans la cross-validation",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_fold_mcc = Gauge(
    "greentech_training_fold_mcc",
    "Matthews Correlation Coefficient du fold (plus c'est haut, mieux c'est)",
    ["model_type", "run_name", "fold"],
    registry=_REGISTRY,
)

training_fold_f1 = Gauge(
    "greentech_training_fold_f1",
    "F1-score sur la classe Green IT pour ce fold",
    ["model_type", "run_name", "fold"],
    registry=_REGISTRY,
)

training_fold_recall = Gauge(
    "greentech_training_fold_recall",
    "Recall sur la classe Green IT pour ce fold",
    ["model_type", "run_name", "fold"],
    registry=_REGISTRY,
)

training_fold_precision = Gauge(
    "greentech_training_fold_precision",
    "Precision sur la classe Green IT pour ce fold",
    ["model_type", "run_name", "fold"],
    registry=_REGISTRY,
)

training_fold_duration_seconds = Gauge(
    "greentech_training_fold_duration_seconds",
    "Duree de l'entrainement + evaluation pour ce fold",
    ["model_type", "run_name", "fold"],
    registry=_REGISTRY,
)

training_cv_mcc_mean = Gauge(
    "greentech_training_cv_mcc_mean",
    "Moyenne du MCC sur les folds (metrique principale de stabilite)",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_cv_mcc_std = Gauge(
    "greentech_training_cv_mcc_std",
    "Ecart-type du MCC sur les folds (garde-fou de stabilite, plafond 0.15)",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_gpu_memory_allocated_bytes = Gauge(
    "greentech_training_gpu_memory_allocated_bytes",
    "Memoire GPU allouee par PyTorch (compatible ROCm et CUDA)",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_gpu_memory_reserved_bytes = Gauge(
    "greentech_training_gpu_memory_reserved_bytes",
    "Memoire GPU reservee par PyTorch (cache interne)",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)

training_last_push_timestamp = Gauge(
    "greentech_training_last_push_timestamp_seconds",
    "Horodatage Unix du dernier push - utile pour detecter un job en panne",
    ["model_type", "run_name"],
    registry=_REGISTRY,
)


def push_metrics(job_name: str = "greentech-training") -> None:
    """Pousse le snapshot courant du registre vers le Pushgateway.

    Les erreurs reseau sont silencieuses : un Pushgateway indisponible ne
    doit jamais faire echouer l'entrainement. On logue simplement un
    avertissement pour traquer les deconnexions dans Grafana.

    Args:
        job_name: Identifiant du job dans le Pushgateway (utilise comme
            label Prometheus pour separer plusieurs pipelines paralleles).
    """
    try:
        push_to_gateway(_PUSHGATEWAY_URL, job=job_name, registry=_REGISTRY)
    except Exception as exc:
        logger.warning(f"Push Prometheus echoue ({_PUSHGATEWAY_URL}) : {exc}")


def update_gpu_memory(model_type: str, run_name: str) -> None:
    """Snapshot de la memoire GPU active via torch.cuda (compatible ROCm).

    Silencieux si PyTorch tourne en CPU-only : on ne veut pas faire du bruit
    sur les workstations sans GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        training_gpu_memory_allocated_bytes.labels(
            model_type=model_type, run_name=run_name
        ).set(allocated)
        training_gpu_memory_reserved_bytes.labels(
            model_type=model_type, run_name=run_name
        ).set(reserved)
    except Exception as exc:
        logger.debug(f"Lecture memoire GPU impossible : {exc}")


def record_fold_metrics(
    *,
    model_type: str,
    run_name: str,
    fold: int,
    total_folds: int,
    metrics: dict[str, float],
    duration_seconds: float,
) -> None:
    """Enregistre les metriques d'un fold et pousse vers Pushgateway.

    Args:
        model_type: "challenger-llama" ou "challenger-qwen".
        run_name: Nom du run MLflow associe (pour joindre les deux systemes).
        fold: Numero du fold (1-indexe).
        total_folds: K dans la cross-validation.
        metrics: Dictionnaire contenant au moins mcc/f1/recall/precision.
        duration_seconds: Duree training + evaluation du fold.
    """
    labels = {"model_type": model_type, "run_name": run_name}
    fold_label = str(fold)

    training_fold_current.labels(**labels).set(fold)
    training_fold_total.labels(**labels).set(total_folds)
    training_fold_duration_seconds.labels(**labels, fold=fold_label).set(duration_seconds)
    training_last_push_timestamp.labels(**labels).set(time.time())

    for name, gauge in (
        ("mcc", training_fold_mcc),
        ("f1", training_fold_f1),
        ("recall", training_fold_recall),
        ("precision", training_fold_precision),
    ):
        if name in metrics and metrics[name] is not None:
            gauge.labels(**labels, fold=fold_label).set(float(metrics[name]))

    update_gpu_memory(model_type, run_name)
    push_metrics()


def record_cv_aggregated(
    *,
    model_type: str,
    run_name: str,
    mcc_mean: float,
    mcc_std: float,
) -> None:
    """Enregistre les metriques agregees du K-fold (fin de run).

    Le couple (mcc_mean, mcc_std) sert directement au garde-fou de
    promotion : std <= 0.15 pour considerer le modele stable.
    """
    labels = {"model_type": model_type, "run_name": run_name}
    training_cv_mcc_mean.labels(**labels).set(float(mcc_mean))
    training_cv_mcc_std.labels(**labels).set(float(mcc_std))
    training_last_push_timestamp.labels(**labels).set(time.time())
    push_metrics()
