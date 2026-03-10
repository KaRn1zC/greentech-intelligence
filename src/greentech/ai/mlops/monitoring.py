"""Metriques Prometheus pour le monitoring des modeles IA en production.

Definit et expose les indicateurs cles pour surveiller la sante
et les performances du systeme de classification Green IT :
- Drift des donnees (distribution des predictions)
- Temps de reponse des inferences
- Pourcentage de classification Green IT
- Emissions carbone cumulees

Les metriques sont collectees par Prometheus et visualisees dans Grafana.
L'integration effective avec les endpoints FastAPI se fera a l'etape 4.

Redige par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# === Metriques d'inference ===

INFERENCE_REQUESTS_TOTAL = Counter(
    "greentech_inference_requests_total",
    "Nombre total de requetes d'inference",
    ["modele", "resultat"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "greentech_inference_duration_seconds",
    "Duree des inferences en secondes",
    ["modele"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_CONFIDENCE_SCORE = Histogram(
    "greentech_inference_confidence",
    "Distribution des scores de confiance",
    ["modele", "label"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

# === Metriques de classification ===

ARTICLES_CLASSIFIED_TOTAL = Counter(
    "greentech_articles_classified_total",
    "Nombre total d'articles classifies",
    ["label"],
)

GREEN_IT_RATIO = Gauge(
    "greentech_green_it_ratio",
    "Ratio actuel d'articles classifies Green IT (0.0-1.0)",
)

ARTICLES_PENDING = Gauge(
    "greentech_articles_pending_classification",
    "Nombre d'articles en attente de classification",
)

# === Metriques de resume ===

SUMMARY_REQUESTS_TOTAL = Counter(
    "greentech_summary_requests_total",
    "Nombre total de requetes de resume",
    ["statut"],
)

SUMMARY_DURATION_SECONDS = Histogram(
    "greentech_summary_duration_seconds",
    "Duree des resumes en secondes",
    buckets=(1.0, 2.0, 3.0, 5.0, 10.0, 20.0),
)

# === Metriques Green IT (empreinte carbone) ===

CARBON_EMISSIONS_GRAMS = Counter(
    "greentech_carbon_emissions_grams_total",
    "Emissions carbone cumulees en grammes de CO2eq",
    ["operation"],
)

ENERGY_CONSUMED_KWH = Counter(
    "greentech_energy_consumed_kwh_total",
    "Energie consommee cumulee en kWh",
    ["operation"],
)

# === Metriques de drift ===

DATA_DRIFT_SCORE = Gauge(
    "greentech_data_drift_score",
    "Score de drift des donnees (0 = stable, 1 = derive complete)",
)

PREDICTION_DISTRIBUTION = Gauge(
    "greentech_prediction_distribution",
    "Distribution des predictions par label",
    ["label"],
)

# === Informations du modele ===

MODEL_INFO = Info(
    "greentech_model",
    "Informations sur le modele en production",
)


def record_inference(
    modele: str,
    label: str,
    confiance: float,
    duree_secondes: float,
) -> None:
    """Enregistre les metriques d'une inference.

    Appele apres chaque prediction pour alimenter les compteurs
    et histogrammes Prometheus.

    Args:
        modele: Nom du modele utilise.
        label: Label predit ("green_it" ou "non_green_it").
        confiance: Score de confiance (0.0-1.0).
        duree_secondes: Duree de l'inference en secondes.
    """
    INFERENCE_REQUESTS_TOTAL.labels(modele=modele, resultat=label).inc()
    INFERENCE_DURATION_SECONDS.labels(modele=modele).observe(duree_secondes)
    INFERENCE_CONFIDENCE_SCORE.labels(modele=modele, label=label).observe(confiance)
    ARTICLES_CLASSIFIED_TOTAL.labels(label=label).inc()


def record_summary(
    succes: bool,
    duree_secondes: float,
) -> None:
    """Enregistre les metriques d'un resume.

    Args:
        succes: True si le resume a ete genere avec succes.
        duree_secondes: Duree de la generation en secondes.
    """
    statut = "succes" if succes else "echec"
    SUMMARY_REQUESTS_TOTAL.labels(statut=statut).inc()
    if succes:
        SUMMARY_DURATION_SECONDS.observe(duree_secondes)


def record_carbon(
    operation: str,
    emissions_g: float,
    energie_kwh: float,
) -> None:
    """Enregistre les metriques d'empreinte carbone.

    Args:
        operation: Nom de l'operation (inference, training, summary).
        emissions_g: Emissions en grammes de CO2eq.
        energie_kwh: Energie en kWh.
    """
    CARBON_EMISSIONS_GRAMS.labels(operation=operation).inc(emissions_g)
    ENERGY_CONSUMED_KWH.labels(operation=operation).inc(energie_kwh)


def update_model_info(
    nom: str,
    version: str,
    type_modele: str,
) -> None:
    """Met a jour les informations du modele en production.

    Args:
        nom: Nom du modele (ex: "microsoft/deberta-v3-base").
        version: Version ou identifiant du run MLflow.
        type_modele: Type (champion, challenger).
    """
    MODEL_INFO.info(
        {
            "nom": nom,
            "version": version,
            "type": type_modele,
        }
    )


def update_green_it_ratio(green: int, total: int) -> None:
    """Met a jour le ratio Green IT.

    Args:
        green: Nombre d'articles classifies Green IT.
        total: Nombre total d'articles classifies.
    """
    if total > 0:
        GREEN_IT_RATIO.set(green / total)
    PREDICTION_DISTRIBUTION.labels(label="green_it").set(green)
    PREDICTION_DISTRIBUTION.labels(label="non_green_it").set(total - green)
