"""Module de tracking MLflow pour les expériences d'entraînement.

Configure et gère le suivi des expériences ML avec MLflow, incluant
la mesure d'empreinte carbone via CodeCarbon. Centralise les fonctions
de logging pour les paramètres, métriques et artefacts.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from codecarbon import EmissionsTracker
from loguru import logger

from greentech.config import get_settings

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class ExperimentConfig:
    """Configuration d'une expérience MLflow.

    Attributes:
        nom_experience: Nom de l'expérience MLflow.
        nom_run: Nom du run individuel (optionnel).
        tags: Tags supplémentaires à attacher au run.
        params: Hyperparamètres à logger.
        mesurer_carbone: Active la mesure CodeCarbon.
    """

    nom_experience: str = "greentech-classification"
    nom_run: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    mesurer_carbone: bool = True


def configure_mlflow() -> None:
    """Configure MLflow avec les paramètres du projet.

    Initialise l'URI de tracking et l'expérience par défaut.
    Doit être appelée une fois au démarrage de l'application.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Désactiver le logging automatique pour garder le contrôle
    mlflow.autolog(disable=True)

    logger.info(
        f"MLflow configuré : tracking_uri={settings.mlflow_tracking_uri}, "
        f"experiment={settings.mlflow_experiment_name}"
    )


@contextmanager
def tracked_experiment(
    config: ExperimentConfig,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager pour une expérience MLflow trackée avec mesure carbone.

    Configure automatiquement MLflow, lance un run, mesure les émissions
    carbone via CodeCarbon, et log tous les résultats à la fin.

    Args:
        config: Configuration de l'expérience (nom, params, tags).

    Yields:
        Le run MLflow actif pour logger des métriques supplémentaires.

    Example:
        ```python
        config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="deberta-v3-base-lr3e5",
            params={"learning_rate": 3e-5, "epochs": 5},
        )
        with tracked_experiment(config) as run:
            # Entraînement du modèle
            mlflow.log_metrics({"accuracy": 0.95, "f1": 0.93})
        ```
    """
    configure_mlflow()
    mlflow.set_experiment(config.nom_experience)

    tracker = None
    if config.mesurer_carbone:
        tracker = EmissionsTracker(
            project_name=config.nom_experience,
            log_level="warning",
            save_to_file=True,
            output_dir=str(Path("logs")),
        )

    tags = {
        "projet": "greentech-intelligence",
        "auteur": "KaRn1zC",
        **config.tags,
    }

    with mlflow.start_run(run_name=config.nom_run, tags=tags) as run:
        logger.info(f"MLflow Run démarré : {run.info.run_id} ({config.nom_run})")

        # Logger les paramètres
        if config.params:
            mlflow.log_params(config.params)
            logger.debug(f"Paramètres loggés : {list(config.params.keys())}")

        # Démarrer le tracking carbone
        if tracker:
            tracker.start()
            logger.info("CodeCarbon : mesure des émissions activée")

        try:
            yield run
        finally:
            # Arrêter le tracking carbone et logger les résultats
            if tracker:
                emissions = tracker.stop()
                if emissions is not None:
                    mlflow.log_metrics(
                        {
                            "emissions_carbone_kg": emissions,
                            "emissions_carbone_g": emissions * 1000,
                        }
                    )
                    logger.info(
                        f"CodeCarbon : {emissions * 1000:.4f} g CO2eq émis (run: {config.nom_run})"
                    )

            logger.info(f"MLflow Run terminé : {run.info.run_id}")


def log_model_metrics(
    metrics: dict[str, float],
    *,
    step: int | None = None,
) -> None:
    """Log des métriques dans le run MLflow actif.

    Wrapper autour de mlflow.log_metrics avec logging Loguru.

    Args:
        metrics: Dictionnaire de métriques {nom: valeur}.
        step: Étape d'entraînement (epoch, batch, etc.).
    """
    mlflow.log_metrics(metrics, step=step)
    logger.debug(f"Métriques loggées (step={step}) : {metrics}")


def log_model_artifact(
    chemin_local: str | Path,
    *,
    dossier_artifact: str = "model",
) -> None:
    """Sauvegarde un artefact (modèle, config, etc.) dans MLflow.

    Args:
        chemin_local: Chemin local du fichier ou dossier à logger.
        dossier_artifact: Sous-dossier dans les artefacts MLflow.
    """
    mlflow.log_artifact(str(chemin_local), artifact_path=dossier_artifact)
    logger.info(f"Artefact loggé : {chemin_local} → {dossier_artifact}/")


def get_best_run(
    nom_experience: str | None = None,
    metrique: str = "accuracy",
    *,
    ascending: bool = False,
) -> dict[str, Any] | None:
    """Récupère le meilleur run d'une expérience selon une métrique.

    Args:
        nom_experience: Nom de l'expérience (défaut: celle du projet).
        metrique: Métrique de comparaison.
        ascending: Si True, le meilleur est le plus petit (ex: loss).

    Returns:
        Dictionnaire avec les infos du meilleur run, ou None si aucun run.
    """
    settings = get_settings()
    nom = nom_experience or settings.mlflow_experiment_name

    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(nom)
    if experiment is None:
        logger.warning(f"Expérience '{nom}' introuvable")
        return None

    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metrique} {order}"],
        max_results=1,
    )

    if runs.empty:
        logger.info(f"Aucun run trouvé pour l'expérience '{nom}'")
        return None

    best = runs.iloc[0]
    logger.info(
        f"Meilleur run : {best['run_id']} ({metrique}={best.get(f'metrics.{metrique}', 'N/A')})"
    )

    return best.to_dict()
