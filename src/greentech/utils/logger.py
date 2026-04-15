"""Configuration centralisee de Loguru avec integration Loki.

Fournit un setup unique pour le logging structure (JSON) avec envoi
automatique vers Grafana Loki pour la centralisation des logs.
Compatible avec le middleware HTTP de l'API FastAPI.

"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings

# Constante pour le format Loki
_LOKI_PUSH_PATH = "/loki/api/v1/push"

# Sondage de l'endpoint /ready de Loki. Loki 2.x met jusqu'a 40 s pour
# terminer son init (chargement WAL, compactor, schema). On retente plusieurs
# fois pour eviter les faux-positifs 503 quand on lance le pipeline juste
# apres `docker compose up -d`.
_LOKI_READY_MAX_ATTEMPTS = 5
_LOKI_READY_BACKOFF_SECONDS = (1.0, 3.0, 5.0, 10.0, 15.0)

# Loggers externes tres verbeux : on abaisse leur niveau par defaut pour qu'ils
# ne polluent pas la sortie. L'utilisateur peut les reactiver au niveau INFO en
# configurant explicitement leur level via logging.getLogger(...).setLevel().
_NOISY_LOGGERS = (
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "sqlalchemy.dialects",
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "botocore",
    "boto3",
    "s3transfer",
)


class _InterceptHandler(logging.Handler):
    """Redirige les logs Python standard (logging) vers Loguru.

    Permet d'unifier le format des logs provenant de librairies tierces
    (SQLAlchemy, httpx, etc.) avec ceux emis via Loguru, pour qu'ils
    apparaissent dans le meme fichier au meme format.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


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
                "values": [[str(int(datetime.now(UTC).timestamp() * 1e9)), json.dumps(log_entry)]],
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
    - Interception des logs Python standard (SQLAlchemy, httpx, etc.) vers Loguru
    - Quietening des loggers tiers tres verbeux (SQLAlchemy echo, boto, ...)

    Args:
        level: Niveau minimum de log (DEBUG, INFO, WARNING, ERROR).
        enable_loki: Active l'envoi vers Grafana Loki.
        json_logs: Si True, format JSON pour les logs console.
    """
    # Reset complet des handlers
    logger.remove()

    # Intercepter tous les logs Python standard (sqlalchemy, httpx, boto, ...)
    # et les rediriger vers Loguru pour un format de log unifie.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Abaisser le niveau des loggers tres bavards pour eviter la pollution.
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

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

    # Fichier rotatif (encoding UTF-8 explicite : Windows utilise cp1252 par
    # defaut, ce qui fait planter l'ecriture des logs contenant du texte
    # d'articles avec caracteres Unicode -- emojis, accents non latin-1, etc.)
    logger.add(
        "logs/greentech_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
        encoding="utf-8",
    )

    # Loki sink : sondage avec backoff pour encaisser le demarrage differe du
    # service (jusqu'a 40 s apres `docker compose up -d`).
    if enable_loki:
        settings = get_settings()
        ready = _wait_for_loki_ready(settings.loki_url)
        if ready:
            logger.add(
                _loki_sink,
                level=level,
                serialize=False,
            )
            logger.info(f"Loki connecte : {settings.loki_url}")
        else:
            logger.warning(
                f"Loki non pret apres {_LOKI_READY_MAX_ATTEMPTS} tentatives, logs locaux uniquement"
            )

    logger.info("Logging configure avec succes")


def _wait_for_loki_ready(loki_url: str) -> bool:
    """Sondage de l'endpoint ``/ready`` de Loki avec backoff progressif.

    Loki renvoie HTTP 503 pendant sa phase d'initialisation (chargement WAL,
    boot du compactor). Sans retry, un pipeline lance juste apres
    ``docker compose up -d`` tombe en 503 alors que le service serait
    disponible quelques secondes plus tard. Les timeouts cumules plafonnent
    a ~34 s (1 + 3 + 5 + 10 + 15), ce qui couvre largement le temps moyen
    de boot observe.

    Args:
        loki_url: URL de base de l'API Loki (ex. ``http://localhost:3100``).

    Returns:
        True si Loki a repondu 200 avant l'epuisement des tentatives.
    """
    ready_url = f"{loki_url}/ready"
    for attempt, delay in enumerate(_LOKI_READY_BACKOFF_SECONDS, start=1):
        try:
            response = httpx.get(ready_url, timeout=2.0)
            if response.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        if attempt < _LOKI_READY_MAX_ATTEMPTS:
            time.sleep(delay)
    return False
