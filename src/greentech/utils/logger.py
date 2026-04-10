"""Configuration centralisee de Loguru avec integration Loki.

Fournit un setup unique pour le logging structure (JSON) avec envoi
automatique vers Grafana Loki pour la centralisation des logs.
Compatible avec le middleware HTTP de l'API FastAPI.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import contextlib
import json
import sys
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings

# Constante pour le format Loki
_LOKI_PUSH_PATH = "/loki/api/v1/push"


def _loki_sink(message: Any) -> None:
    """Envoie un log structure vers Grafana Loki via l'API push.

    Formatte le message en JSON et l'envoie a Loki de maniere synchrone.
    En cas d'echec de connexion, le log est silencieusement ignore
    pour eviter une boucle de logs infinie.

    Args:
        message: Objet message Loguru contenant le record complet.
    """
    record = message.record
    level = record["level"].name.lower()

    # Labels Loki (indexables, utilises pour le filtrage)
    labels = {
        "job": "greentech",
        "level": level,
        "module": record["module"],
    }

    # Construire le payload JSON structure
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": level,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Ajouter les extras (contexte additionnel)
    if record["extra"]:
        log_entry["extra"] = {k: str(v) for k, v in record["extra"].items()}

    # Ajouter l'exception si presente
    if record["exception"]:
        log_entry["exception"] = str(record["exception"])

    # Format Loki push API
    payload = {
        "streams": [
            {
                "stream": labels,
                "values": [
                    [str(int(datetime.now(UTC).timestamp() * 1e9)), json.dumps(log_entry)]
                ],
            }
        ]
    }

    settings = get_settings()
    loki_url = f"{settings.loki_url}{_LOKI_PUSH_PATH}"

    with contextlib.suppress(httpx.ConnectError, httpx.TimeoutException):
        httpx.post(loki_url, json=payload, timeout=2.0)


def setup_logging(
    *,
    level: str = "INFO",
    enable_loki: bool = True,
    json_logs: bool = False,
) -> None:
    """Configure Loguru avec console, fichier rotatif et envoi Loki.

    Supprime les handlers existants et reconfigure avec :
    - Sortie console (stderr) coloree
    - Fichier rotatif dans logs/ (10 Mo, retention 7 jours)
    - Envoi vers Loki si active et accessible

    Args:
        level: Niveau minimum de log (DEBUG, INFO, WARNING, ERROR).
        enable_loki: Active l'envoi vers Grafana Loki.
        json_logs: Si True, format JSON pour les logs console.
    """
    # Reset complet des handlers
    logger.remove()

    # Format console standard
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    if json_logs:
        console_format = "{message}"
        logger.add(
            sys.stderr,
            level=level,
            format=console_format,
            serialize=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=console_format,
            colorize=True,
        )

    # Fichier rotatif
    logger.add(
        "logs/greentech_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
    )

    # Loki sink
    if enable_loki:
        settings = get_settings()
        try:
            response = httpx.get(f"{settings.loki_url}/ready", timeout=2.0)
            if response.status_code == 200:
                logger.add(
                    _loki_sink,
                    level=level,
                    serialize=False,
                )
                logger.info(f"Loki connecte : {settings.loki_url}")
            else:
                logger.warning(f"Loki non pret (status {response.status_code}), logs locaux uniquement")
        except (httpx.ConnectError, httpx.TimeoutException):
            logger.warning("Loki inaccessible, logs locaux uniquement")

    logger.info("Logging configure avec succes")
