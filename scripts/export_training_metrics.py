"""Rejoue les metriques d'entrainement MLflow vers le Pushgateway Prometheus.

Script CLI autonome qui execute le module
``greentech.ai.mlops.training_metrics_exporter``. Republie les agreges de
cross-validation (``cv_mcc_mean`` / ``cv_mcc_std``) et les metriques des
baselines (``mcc`` / ``f1`` / ``recall``...) deja traces dans MLflow, afin que
le dashboard ``pipeline-training`` se remplisse sans avoir a relancer un
entrainement (utile apres une purge du Pushgateway ou l'expiration de l'ancienne
retention Prometheus de 15 jours).

Le rejeu est idempotent : chaque push remplace le groupe de son job, on peut
donc le relancer sans risque de double-comptage.

Usage
-----

    uv run python scripts/export_training_metrics.py

"""

from __future__ import annotations

from loguru import logger

from greentech.ai.mlops.training_metrics_exporter import export_training_metrics


def main() -> int:
    """Lance le rejeu et retourne un code de sortie shell exploitable."""
    counts = export_training_metrics()
    if not counts["cv"] and not counts["baseline"]:
        logger.warning(
            "Aucun run CV ni baseline exploitable trouve dans MLflow : Pushgateway inchange."
        )
        return 1
    logger.info(
        f"Export termine : {counts['cv']} run(s) de cross-validation et "
        f"{counts['baseline']} baseline(s) republie(s) vers le Pushgateway."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
