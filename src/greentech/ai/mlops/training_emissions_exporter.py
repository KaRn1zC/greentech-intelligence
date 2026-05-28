"""Rejeu des emissions carbone d'entrainement (MLflow -> Pushgateway).

Les runs d'entrainement, baseline et benchmark sont deja traces avec
CodeCarbon via ``tracked_experiment()`` (cf. ``tracking.py``) : la metrique
``emissions_carbone_g`` est attachee a chaque run MLflow. Mais Prometheus
ne sait pas lire MLflow directement et les dashboards Grafana ne
disposaient donc d'aucun panneau d'empreinte carbone alimente, alors meme
que les donnees existent depuis le premier run.

Ce module joue le role de pont :

1. Il scanne l'experience ``greentech-classification`` dans MLflow.
2. Il extrait pour chaque run la metrique ``emissions_carbone_g`` et
   reconstitue le type de modele (``qwen3``, ``mdeberta``...) ainsi que la
   phase (``baseline``, ``cv``, ``final``, ``benchmark``) via les tags MLflow.
3. Il publie les snapshots dans des jauges Prometheus dediees
   (``greentech_training_emissions_grams*``) puis pousse vers le Pushgateway.

Le rejeu est idempotent : la totalite des jauges est purgee avant chaque
export, ce qui permet de relancer ce module a la demande (script CLI) ou
automatiquement au demarrage de l'API sans cumuler d'anciennes valeurs.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from loguru import logger

from greentech.ai.mlops.prometheus_metrics import (
    push_training_emissions,
    record_training_emissions_snapshot,
    reset_training_emissions_gauges,
    update_training_emissions_aggregates,
)

# Nom de l'experience MLflow qui contient l'integralite des runs Green IT.
# Aligne avec ``Settings.mlflow_experiment_name`` (defaut du projet).
_EXPERIMENT_NAME = "greentech-classification"

# Cle MLflow contenant l'empreinte carbone du run (loguee par CodeCarbon).
# La version ``_g`` (grammes) est attendue ; ``_kg`` est utilise en repli si
# d'anciens runs n'ont logue qu'en kilogrammes.
_EMISSIONS_METRIC_G = "emissions_carbone_g"
_EMISSIONS_METRIC_KG = "emissions_carbone_kg"

# Heuristique de detection du type de modele a partir des tags / params /
# nom de run, dans l'ordre de priorite. La premiere correspondance gagne.
_MODEL_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("qwen3", re.compile(r"qwen[\W_-]?3", re.IGNORECASE)),
    ("qwen2.5", re.compile(r"qwen[\W_-]?2\.5|qwen[\W_-]?2_5", re.IGNORECASE)),
    ("mdeberta", re.compile(r"m[\W_-]?deberta", re.IGNORECASE)),
    ("deberta", re.compile(r"deberta", re.IGNORECASE)),
    ("llama3.2", re.compile(r"llama[\W_-]?3", re.IGNORECASE)),
)

# Mapping des phases reconnues a partir du nom de run / tags. L'inconnu
# retombe sur ``other`` pour conserver une comptabilisation exhaustive.
_PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("baseline", re.compile(r"baseline", re.IGNORECASE)),
    ("benchmark", re.compile(r"benchmark", re.IGNORECASE)),
    ("augmentation", re.compile(r"augment", re.IGNORECASE)),
    ("cv", re.compile(r"cv|k\d|fold|unified|challenger", re.IGNORECASE)),
    ("final", re.compile(r"champion|final|production|promote", re.IGNORECASE)),
)


@dataclass(frozen=True)
class TrainingEmissionRecord:
    """Snapshot d'empreinte carbone normalise pour un run MLflow.

    Attributes:
        model_type: Famille de modele (``qwen3``, ``mdeberta``, ``llama3.2``...).
        run_name: Nom du run MLflow, conserve tel quel pour servir de label.
        phase: Etape pipeline (``baseline``, ``cv``, ``final``...).
        emissions_g: Empreinte carbone en grammes de CO2eq.
    """

    model_type: str
    run_name: str
    phase: str
    emissions_g: float


def _detect_model_type(*candidates: str | None) -> str:
    """Detecte le type de modele en testant les patterns sur plusieurs sources.

    Args:
        *candidates: Chaines a inspecter (tags, params, nom de run...).

    Returns:
        Le slug du modele detecte, ou ``unknown`` si aucun pattern ne matche.
    """
    haystack = " ".join(filter(None, candidates))
    for slug, pattern in _MODEL_TYPE_PATTERNS:
        if pattern.search(haystack):
            return slug
    return "unknown"


def _detect_phase(run_name: str, tags: dict[str, str]) -> str:
    """Devine la phase du pipeline a partir du nom et des tags du run."""
    explicit = tags.get("phase") or tags.get("type")
    haystack = " ".join(filter(None, (explicit, run_name)))
    for slug, pattern in _PHASE_PATTERNS:
        if pattern.search(haystack):
            return slug
    return "other"


def _build_record(run_dict: dict) -> TrainingEmissionRecord | None:
    """Convertit un run pandas MLflow en ``TrainingEmissionRecord``.

    Renvoie ``None`` si le run n'a pas de metrique d'emission exploitable :
    les anciens runs sans CodeCarbon ne doivent pas polluer le total.
    """
    metrics = run_dict.get("metrics", {}) or {}
    if isinstance(metrics, dict):
        emissions_g = metrics.get(_EMISSIONS_METRIC_G)
        if emissions_g is None and _EMISSIONS_METRIC_KG in metrics:
            emissions_g = float(metrics[_EMISSIONS_METRIC_KG]) * 1000.0
    else:
        emissions_g = None
    if emissions_g is None:
        return None
    try:
        emissions_g = float(emissions_g)
    except (TypeError, ValueError):
        return None
    if emissions_g <= 0.0:
        return None

    tags = run_dict.get("tags", {}) or {}
    params = run_dict.get("params", {}) or {}
    run_name = (
        tags.get("mlflow.runName")
        or run_dict.get("run_name")
        or run_dict.get("info", {}).get("run_name", "")
        or ""
    )

    model_type = _detect_model_type(
        tags.get("model_type"),
        params.get("model_type"),
        params.get("model"),
        tags.get("modele"),
        run_name,
    )
    phase = _detect_phase(run_name, tags)

    return TrainingEmissionRecord(
        model_type=model_type,
        run_name=run_name or "(sans nom)",
        phase=phase,
        emissions_g=emissions_g,
    )


def _normalize_pandas_row(row) -> dict:
    """Aplati une ligne ``mlflow.search_runs`` en dict ``{metrics, tags, params}``.

    ``mlflow.search_runs`` renvoie un DataFrame avec des colonnes prefixees
    (``metrics.``, ``tags.``, ``params.``). On les regroupe dans un dict
    compatible avec ``_build_record`` pour decoupler la logique du format.
    """
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}
    params: dict[str, str] = {}
    for col, value in row.items():
        if value is None:
            continue
        try:
            import pandas as pd  # local pour eviter l'import si jamais utilise

            if pd.isna(value):
                continue
        except (ImportError, TypeError):
            pass
        if col.startswith("metrics."):
            try:
                metrics[col[len("metrics.") :]] = float(value)
            except (TypeError, ValueError):
                continue
        elif col.startswith("tags."):
            tags[col[len("tags.") :]] = str(value)
        elif col.startswith("params."):
            params[col[len("params.") :]] = str(value)
    return {
        "metrics": metrics,
        "tags": tags,
        "params": params,
        "run_name": tags.get("mlflow.runName", ""),
    }


def collect_training_emissions(
    experiment_name: str = _EXPERIMENT_NAME,
) -> list[TrainingEmissionRecord]:
    """Recupere tous les snapshots carbone exploitables depuis MLflow.

    Args:
        experiment_name: Nom de l'experience MLflow a scanner. Defaut aligne
            sur ``Settings.mlflow_experiment_name``.

    Returns:
        Liste des records normalises. Vide si MLflow est inaccessible ou si
        l'experience n'existe pas (l'appelant logue un avertissement).
    """
    import mlflow

    from greentech.ai.mlops.tracking import configure_mlflow

    configure_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experience MLflow introuvable : {experiment_name}")
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        run_view_type=1,  # ACTIVE_ONLY
        max_results=10_000,
        output_format="pandas",
    )
    if runs.empty:
        logger.info(f"Aucun run MLflow trouve dans {experiment_name}")
        return []

    records: list[TrainingEmissionRecord] = []
    for _, row in runs.iterrows():
        record = _build_record(_normalize_pandas_row(row))
        if record is not None:
            records.append(record)
    logger.debug(f"{len(records)} run(s) avec mesure carbone exploitable")
    return records


def export_training_emissions(
    experiment_name: str = _EXPERIMENT_NAME,
) -> dict[str, float]:
    """Scanne MLflow et publie les emissions training vers le Pushgateway.

    Idempotent : les jauges sont purgees avant chaque export, ce qui permet
    de rappeler cette fonction (lifespan API ou script CLI) sans cumuler.

    Args:
        experiment_name: Nom de l'experience MLflow a scanner.

    Returns:
        Dictionnaire ``{model_type: total_emissions_g}`` pour faciliter le
        logging cote appelant. Vide si aucun run exploitable.
    """
    records = collect_training_emissions(experiment_name=experiment_name)
    reset_training_emissions_gauges()
    if not records:
        update_training_emissions_aggregates(per_model_total_g={}, per_model_count={})
        push_training_emissions()
        return {}

    per_model_total: dict[str, float] = defaultdict(float)
    per_model_count: dict[str, int] = defaultdict(int)
    for record in records:
        record_training_emissions_snapshot(
            model_type=record.model_type,
            run_name=record.run_name,
            phase=record.phase,
            emissions_g=record.emissions_g,
        )
        per_model_total[record.model_type] += record.emissions_g
        per_model_count[record.model_type] += 1

    update_training_emissions_aggregates(
        per_model_total_g=dict(per_model_total),
        per_model_count=dict(per_model_count),
    )
    push_training_emissions()
    total_g = sum(per_model_total.values())
    logger.info(
        f"Empreinte carbone training rejouee : {len(records)} run(s), "
        f"{total_g:.2f} g CO2eq cumules ({dict(per_model_total)})"
    )
    return dict(per_model_total)


if __name__ == "__main__":
    export_training_emissions()
