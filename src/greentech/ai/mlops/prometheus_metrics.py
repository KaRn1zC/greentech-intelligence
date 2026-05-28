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

# Metriques de la baseline (modele pre-entraine sans fine-tuning).
# Label "model" permet de distinguer plusieurs runs sur des architectures
# differentes (Qwen3-4B, Qwen2.5-3B, Llama 3.2 3B, etc.) dans Grafana.
baseline_mcc = Gauge(
    "greentech_baseline_mcc",
    "MCC de la baseline (modele pre-entraine sans fine-tuning)",
    ["model"],
    registry=_REGISTRY,
)

baseline_f1 = Gauge(
    "greentech_baseline_f1",
    "F1-score Green IT de la baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_recall = Gauge(
    "greentech_baseline_recall",
    "Recall Green IT de la baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_precision = Gauge(
    "greentech_baseline_precision",
    "Precision Green IT de la baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_accuracy = Gauge(
    "greentech_baseline_accuracy",
    "Accuracy globale de la baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_balanced_accuracy = Gauge(
    "greentech_baseline_balanced_accuracy",
    "Balanced accuracy de la baseline (moyenne sensibilite/specificite)",
    ["model"],
    registry=_REGISTRY,
)

baseline_duration_seconds = Gauge(
    "greentech_baseline_duration_seconds",
    "Duree totale de l'evaluation baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_n_articles = Gauge(
    "greentech_baseline_n_articles",
    "Nombre d'articles evalues par la baseline",
    ["model"],
    registry=_REGISTRY,
)

baseline_last_push_timestamp = Gauge(
    "greentech_baseline_last_push_timestamp_seconds",
    "Horodatage Unix du dernier push baseline - utile pour detecter une stase",
    ["model"],
    registry=_REGISTRY,
)

# === Empreinte carbone des runs d'entrainement (rejouee depuis MLflow) ===
#
# Ces jauges materialisent les emissions CO2eq deja loguees dans MLflow par
# ``tracked_experiment()`` pendant les entrainements. Un exporter scanne
# l'experience ``greentech-classification`` au demarrage de l'API (et a la
# demande via un script CLI) et publie ces jauges vers le Pushgateway, afin
# que le dashboard ``pipeline-training`` puisse afficher l'empreinte carbone
# cumulee des entrainements sans avoir a re-lancer un seul run.

training_emissions_grams = Gauge(
    "greentech_training_emissions_grams",
    "Emissions CO2eq d'un run d'entrainement (en grammes), rejouees depuis MLflow",
    ["model_type", "run_name", "phase"],
    registry=_REGISTRY,
)

training_emissions_grams_total = Gauge(
    "greentech_training_emissions_grams_total",
    "Somme des emissions CO2eq par type de modele (grammes)",
    ["model_type"],
    registry=_REGISTRY,
)

training_emissions_grams_grand_total = Gauge(
    "greentech_training_emissions_grams_grand_total",
    "Somme totale des emissions CO2eq de tous les entrainements (grammes)",
    registry=_REGISTRY,
)

training_emissions_runs_count = Gauge(
    "greentech_training_emissions_runs_count",
    "Nombre de runs d'entrainement ayant une mesure carbone exploitable",
    ["model_type"],
    registry=_REGISTRY,
)

training_emissions_last_export_timestamp = Gauge(
    "greentech_training_emissions_last_export_timestamp_seconds",
    "Horodatage Unix du dernier rejeu MLflow vers le Pushgateway",
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
        training_gpu_memory_allocated_bytes.labels(model_type=model_type, run_name=run_name).set(
            allocated
        )
        training_gpu_memory_reserved_bytes.labels(model_type=model_type, run_name=run_name).set(
            reserved
        )
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
        model_type: "llama3.2", "qwen2.5" ou "qwen3".
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


def record_baseline_metrics(
    *,
    model_name: str,
    metrics: dict[str, float | int],
    n_articles: int,
    duration_seconds: float,
) -> None:
    """Enregistre les metriques baseline dans le registre et les pousse.

    Permet aux dashboards Grafana d'afficher la baseline comme reference
    immuable a cote des courbes de training K-fold. Le job Pushgateway
    est distinct (``greentech-baseline``) pour que les metriques survivent
    meme si un job de training pousse en parallele avec le meme label.

    Args:
        model_name: Identifiant HF du modele evalue (ex. ``Qwen/Qwen3-4B``).
            Sert de label Prometheus, donc doit etre stable entre runs.
        metrics: Dictionnaire complet renvoye par ``compute_classification_metrics``.
        n_articles: Nombre d'articles traites par la baseline.
        duration_seconds: Duree totale de l'evaluation.
    """
    labels = {"model": model_name}

    for key, gauge in (
        ("mcc", baseline_mcc),
        ("f1", baseline_f1),
        ("recall", baseline_recall),
        ("precision", baseline_precision),
        ("accuracy", baseline_accuracy),
        ("balanced_accuracy", baseline_balanced_accuracy),
    ):
        if key in metrics and metrics[key] is not None:
            gauge.labels(**labels).set(float(metrics[key]))

    baseline_duration_seconds.labels(**labels).set(float(duration_seconds))
    baseline_n_articles.labels(**labels).set(int(n_articles))
    baseline_last_push_timestamp.labels(**labels).set(time.time())

    push_metrics(job_name="greentech-baseline")


def reset_training_emissions_gauges() -> None:
    """Vide les jauges d'emissions training avant un re-export complet.

    Un rejeu MLflow recalcule tous les snapshots ; on remet a zero pour
    eviter qu'un run supprime cote MLflow continue d'apparaitre dans
    Grafana avec sa derniere valeur publiee.
    """
    for gauge in (
        training_emissions_grams,
        training_emissions_grams_total,
        training_emissions_runs_count,
    ):
        gauge.clear()
    training_emissions_grams_grand_total.set(0.0)


def record_training_emissions_snapshot(
    *,
    model_type: str,
    run_name: str,
    phase: str,
    emissions_g: float,
) -> None:
    """Enregistre l'empreinte carbone d'un run d'entrainement sans pousser.

    Les push successifs avec le meme job_name remplaceraient la totalite des
    series. On accumule donc d'abord les jauges en memoire et on appelle
    ``push_training_emissions()`` une fois apres tous les snapshots.

    Args:
        model_type: Famille du modele (``qwen3``, ``mdeberta``, ``llama3.2``...).
        run_name: Nom MLflow du run, conserve tel quel comme label.
        phase: Etape (``baseline``, ``cv``, ``final``, ``benchmark``...).
        emissions_g: Emissions CO2eq mesurees pour ce run, en grammes.
    """
    training_emissions_grams.labels(
        model_type=model_type, run_name=run_name, phase=phase
    ).set(float(emissions_g))


def update_training_emissions_aggregates(
    *,
    per_model_total_g: dict[str, float],
    per_model_count: dict[str, int],
) -> None:
    """Met a jour les agregats par modele et le total general.

    Args:
        per_model_total_g: Somme des emissions par ``model_type`` (grammes).
        per_model_count: Nombre de runs avec mesure exploitable par modele.
    """
    grand_total = 0.0
    for model_type, total in per_model_total_g.items():
        training_emissions_grams_total.labels(model_type=model_type).set(float(total))
        grand_total += float(total)
    for model_type, count in per_model_count.items():
        training_emissions_runs_count.labels(model_type=model_type).set(int(count))
    training_emissions_grams_grand_total.set(grand_total)
    training_emissions_last_export_timestamp.set(time.time())


def push_training_emissions() -> None:
    """Pousse les jauges d'emissions training vers le Pushgateway.

    Job Pushgateway dedie (``greentech-training-emissions``) pour que ces
    jauges survivent meme si un job d'entrainement actif ecrase d'autres
    labels du job ``greentech-training``.
    """
    push_metrics(job_name="greentech-training-emissions")
