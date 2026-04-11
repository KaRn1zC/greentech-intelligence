"""Mesure d'empreinte carbone pour les workloads IA.

Wrapper autour de CodeCarbon pour mesurer les emissions CO2 des
operations d'inference et d'entrainement. Les metriques sont exposees
au format Prometheus pour le monitoring en production.

"""

from __future__ import annotations

from dataclasses import dataclass

from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from loguru import logger

from greentech.config import BASE_DIR

LOGS_DIR = BASE_DIR / "logs"


@dataclass(frozen=True)
class CarbonReport:
    """Rapport d'emissions carbone pour une operation.

    Attributes:
        operation: Nom de l'operation mesuree.
        emissions_kg: Emissions totales en kg de CO2eq.
        duree_secondes: Duree de l'operation en secondes.
        energie_kwh: Energie consommee en kWh.
    """

    operation: str
    emissions_kg: float
    duree_secondes: float
    energie_kwh: float

    @property
    def emissions_g(self) -> float:
        """Emissions en grammes de CO2eq."""
        return self.emissions_kg * 1000


def create_tracker(
    nom_projet: str = "greentech-intelligence",
    *,
    offline: bool = False,
    country_iso_code: str = "FRA",
) -> EmissionsTracker | OfflineEmissionsTracker:
    """Cree un tracker CodeCarbon configure pour le projet.

    En mode online, CodeCarbon utilise l'API pour recuperer le facteur
    d'emission du reseau electrique local. En mode offline, on utilise
    le facteur moyen de la France (faible grace au nucleaire).

    Args:
        nom_projet: Nom du projet pour les rapports.
        offline: Si True, utilise les facteurs d'emission locaux.
        country_iso_code: Code ISO du pays pour le mode offline.

    Returns:
        Tracker configure pret a demarrer.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if offline:
        tracker = OfflineEmissionsTracker(
            project_name=nom_projet,
            country_iso_code=country_iso_code,
            log_level="warning",
            save_to_file=True,
            output_dir=str(LOGS_DIR),
        )
    else:
        tracker = EmissionsTracker(
            project_name=nom_projet,
            log_level="warning",
            save_to_file=True,
            output_dir=str(LOGS_DIR),
        )

    logger.debug(f"CodeCarbon tracker cree : projet={nom_projet}, offline={offline}")
    return tracker


def measure_operation(
    tracker: EmissionsTracker | OfflineEmissionsTracker,
    operation: str,
) -> CarbonReport:
    """Arrete le tracker et retourne un rapport structure.

    Doit etre appele apres tracker.start() et l'execution de l'operation.
    Lit le fichier emissions.csv genere par CodeCarbon pour extraire
    les metriques de duree et d'energie sans acceder aux attributs prives.

    Args:
        tracker: Tracker CodeCarbon en cours d'execution.
        operation: Nom de l'operation pour le rapport.

    Returns:
        Rapport d'emissions carbone.
    """
    emissions_kg = tracker.stop()
    if emissions_kg is None:
        emissions_kg = 0.0

    # Extraire duree et energie depuis le CSV genere par CodeCarbon
    # (evite l'acces aux attributs prives _last_measured_time / _total_energy)
    duree = 0.0
    energie = 0.0
    emissions_csv = LOGS_DIR / "emissions.csv"
    if emissions_csv.exists():
        try:
            import csv

            with emissions_csv.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    duree = float(last.get("duration", 0.0))
                    energie = float(last.get("energy_consumed", 0.0))
        except (KeyError, ValueError, OSError) as e:
            logger.warning(f"Impossible de lire emissions.csv : {e}")

    report = CarbonReport(
        operation=operation,
        emissions_kg=emissions_kg,
        duree_secondes=duree,
        energie_kwh=energie,
    )

    logger.info(
        f"Empreinte carbone [{operation}] : "
        f"{report.emissions_g:.4f}g CO2eq, "
        f"{report.energie_kwh:.6f} kWh"
    )

    return report
