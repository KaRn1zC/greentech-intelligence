"""Rejoue l'empreinte carbone MLflow vers le Pushgateway Prometheus.

Script CLI autonome qui execute le module
``greentech.ai.mlops.training_emissions_exporter``. Sert deux scenarios
operationnels :

1. **Bootstrap** : republier la totalite des emissions training apres un
   nettoyage du Pushgateway ou un re-deploiement de la stack monitoring.
2. **Apres un re-entrainement** : recalculer les agregats par modele sans
   attendre le prochain redemarrage de l'API.

Le rejeu est idempotent (les jauges sont purgees avant l'export) : on peut
le relancer autant de fois que necessaire sans risque de double-comptage.

Usage
-----

    uv run python scripts/export_training_emissions.py

"""

from __future__ import annotations

from loguru import logger

from greentech.ai.mlops.training_emissions_exporter import export_training_emissions


def main() -> int:
    """Lance le rejeu et retourne un code de sortie shell exploitable."""
    totals = export_training_emissions()
    if not totals:
        logger.warning(
            "Aucun run avec mesure carbone trouve : Pushgateway purge mais "
            "aucun snapshot publie."
        )
        return 1
    total_g = sum(totals.values())
    logger.info(
        f"Export termine : {len(totals)} type(s) de modele, "
        f"{total_g:.2f} g CO2eq cumules."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
