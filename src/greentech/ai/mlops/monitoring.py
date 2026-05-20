"""Metriques Prometheus pour le monitoring des modeles IA en production.

Definit et expose les indicateurs cles pour surveiller la sante
et les performances du systeme de classification Green IT :
- Drift des donnees (distribution des predictions)
- Temps de reponse des inferences
- Pourcentage de classification Green IT
- Emissions carbone cumulees

Les metriques sont collectees par Prometheus et visualisees dans Grafana.
L'integration est realisee dans ``api/tasks.py`` (``record_inference`` /
``record_summary`` apres chaque analyse, ``refresh_business_metrics_from_db``
en fin de pipeline) et ``api/main.py`` (seed des jauges metier au demarrage).

"""

from __future__ import annotations

import os

from loguru import logger
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    push_to_gateway,
)

# URL du Pushgateway. Defaut adapte au conteneur (nom de service Docker),
# override-able via env. Le worker Celery n'expose pas d'endpoint /metrics
# scrutable ; il pousse donc ses metriques d'inference vers cette passerelle,
# que Prometheus scrape (honor_labels=true), exactement comme les metriques
# d'entrainement (cf. prometheus_metrics.py).
_PUSHGATEWAY_URL = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL", "pushgateway:9091")

# Registre dedie aux metriques ENREGISTREES PAR LE WORKER Celery (process
# distinct de l'API). Elles ne doivent PAS etre sur le registre global, sinon
# l'API les exposerait a 0 sur son /metrics et creerait des series fantomes
# en doublon. Le worker pousse ce registre vers le Pushgateway apres chaque
# analyse via push_worker_metrics().
WORKER_REGISTRY = CollectorRegistry()

# === Metriques d'inference (worker -> Pushgateway) ===

INFERENCE_REQUESTS_TOTAL = Counter(
    "greentech_inference_requests_total",
    "Nombre total de requetes d'inference",
    ["modele", "resultat"],
    registry=WORKER_REGISTRY,
)

INFERENCE_DURATION_SECONDS = Histogram(
    "greentech_inference_duration_seconds",
    "Duree des inferences en secondes",
    ["modele"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=WORKER_REGISTRY,
)

INFERENCE_CONFIDENCE_SCORE = Histogram(
    "greentech_inference_confidence",
    "Distribution des scores de confiance",
    ["modele", "label"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=WORKER_REGISTRY,
)

# === Metriques de classification (worker -> Pushgateway) ===

ARTICLES_CLASSIFIED_TOTAL = Counter(
    "greentech_articles_classified_total",
    "Nombre total d'articles classifies",
    ["label"],
    registry=WORKER_REGISTRY,
)

# === Metriques de resume (worker -> Pushgateway) ===

SUMMARY_REQUESTS_TOTAL = Counter(
    "greentech_summary_requests_total",
    "Nombre total de requetes de resume",
    ["statut"],
    registry=WORKER_REGISTRY,
)

SUMMARY_DURATION_SECONDS = Histogram(
    "greentech_summary_duration_seconds",
    "Duree des resumes en secondes",
    buckets=(1.0, 2.0, 3.0, 5.0, 10.0, 20.0),
    registry=WORKER_REGISTRY,
)

# === Metriques Green IT / empreinte carbone (worker -> Pushgateway) ===

CARBON_EMISSIONS_GRAMS = Counter(
    "greentech_carbon_emissions_grams_total",
    "Emissions carbone cumulees en grammes de CO2eq",
    ["operation"],
    registry=WORKER_REGISTRY,
)

ENERGY_CONSUMED_KWH = Counter(
    "greentech_energy_consumed_kwh_total",
    "Energie consommee cumulee en kWh",
    ["operation"],
    registry=WORKER_REGISTRY,
)

# === Jauges metier (process API -> /metrics, seedees depuis PostgreSQL) ===
# Ces gauges sont sur le registre GLOBAL : elles sont seedees dans le lifespan
# de l'API (qui expose /metrics) et reflettent la source de verite PostgreSQL.

GREEN_IT_RATIO = Gauge(
    "greentech_green_it_ratio",
    "Ratio actuel d'articles classifies Green IT (0.0-1.0)",
)

ARTICLES_PENDING = Gauge(
    "greentech_articles_pending_classification",
    "Nombre d'articles en attente de classification",
)

DATA_DRIFT_SCORE = Gauge(
    "greentech_data_drift_score",
    "Score de drift des donnees (0 = stable, 1 = derive complete)",
)

PREDICTION_DISTRIBUTION = Gauge(
    "greentech_prediction_distribution",
    "Distribution des predictions par label",
    ["label"],
)

MODEL_INFO = Info(
    "greentech_model",
    "Informations sur le modele en production",
)


def push_worker_metrics() -> None:
    """Pousse les metriques d'inference du worker vers le Pushgateway.

    Le worker Celery n'est pas un serveur HTTP : il n'expose aucun endpoint
    ``/metrics`` que Prometheus pourrait scraper. On pousse donc le registre
    dedie ``WORKER_REGISTRY`` vers le Pushgateway (job ``greentech-inference``),
    que Prometheus scrape. Sans ce push, les compteurs et histogrammes
    d'inference resteraient confines a la memoire du process worker et
    n'apparaitraient jamais dans les dashboards.

    Idempotent et non bloquant : un echec de push (Pushgateway indisponible)
    est logge en warning sans interrompre l'analyse en cours.
    """
    try:
        push_to_gateway(
            _PUSHGATEWAY_URL,
            job="greentech-inference",
            registry=WORKER_REGISTRY,
        )
    except Exception as exc:
        logger.warning(f"Push des metriques worker vers Pushgateway echoue : {exc}")


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
        type_modele: Type (production, baseline, benchmark).
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


def seed_model_info_from_production() -> None:
    """Renseigne ``greentech_model_info`` depuis les metadonnees du modele promu.

    Lit ``models/production/promotion_info.json`` pour exposer le nom du modele
    de base, sa version de promotion et son type. Appelee au demarrage de l'API
    afin que le panneau « Modele en production » du dashboard metier affiche le
    vrai modele servi plutot qu'une valeur par defaut.
    """
    import json
    from pathlib import Path

    nom, version = "Qwen/Qwen3-4B", "inconnue"
    info_path = Path("models/production/promotion_info.json")
    if info_path.exists():
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            version = str(data.get("version", version))
            nom = str(data.get("training_protocol", {}).get("base_model", nom))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Lecture de promotion_info.json impossible : {exc}")
    update_model_info(nom=nom, version=version, type_modele="production")
    logger.debug(f"Model info expose : {nom} ({version})")


async def refresh_business_metrics_from_db() -> None:
    """Recalcule les jauges metier (ratio Green IT, distribution) depuis PostgreSQL.

    Le dashboard « Metier GreenTech » lit ces jauges dans Prometheus. Comme la
    source de verite des classifications est la table ``articles`` (alimentee a
    la fois par les analyses temps-reel et par les traitements batch), on
    rafraichit ``greentech_green_it_ratio`` et ``greentech_prediction_distribution``
    directement depuis la base. Appelee au demarrage de l'API puis apres chaque
    analyse, pour garder les jauges a jour sans dependre du seul trafic live.
    """
    from sqlalchemy import func, select

    from greentech.data.storage.database import async_session_factory
    from greentech.data.storage.models import Article

    try:
        async with async_session_factory() as session:
            total = await session.scalar(select(func.count()).select_from(Article)) or 0
            green = (
                await session.scalar(
                    select(func.count()).select_from(Article).where(Article.est_green_it.is_(True))
                )
                or 0
            )
    except Exception as exc:
        logger.warning(f"Rafraichissement des metriques metier impossible : {exc}")
        return
    update_green_it_ratio(int(green), int(total))
    logger.debug(f"Metriques metier rafraichies : {green}/{total} Green IT")
