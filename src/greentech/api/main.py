"""Application principale FastAPI - GreenTech Intelligence.

Point d'entree de l'API REST exposant les donnees, les fonctionnalites d'IA
et les metriques de monitoring. Configure le cycle de vie de l'application,
les middlewares CORS, le logging Loguru et les routes.

Demarrage : uv run uvicorn src.greentech.api.main:app --reload --port 8000

"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger
from prometheus_client import generate_latest

from greentech.api.routes.analyze import router as analyze_router
from greentech.api.routes.articles import router as articles_router
from greentech.api.routes.auth import router as auth_router
from greentech.api.routes.stats import router as stats_router
from greentech.config import get_settings

_settings = get_settings()


# === Configuration Loguru ===


def _setup_logging() -> None:
    """Configure Loguru comme logger centralise.

    Remplace les logs systeme (uvicorn, sqlalchemy) par Loguru
    pour un format unifie et structure.
    """
    # Supprimer le handler par defaut
    logger.remove()

    # Handler console avec format lisible
    logger.add(
        sys.stderr,
        level=_settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Handler fichier avec rotation (logs persistants)
    logger.add(
        "logs/api_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Intercepter les logs uvicorn
    import logging

    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [InterceptHandler()]
        logging.getLogger(name).propagate = False


# === Cycle de vie de l'application ===


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Gere le demarrage et l'arret de l'application.

    Au demarrage : configure le logging, verifie la connexion DB.
    A l'arret : ferme proprement les connexions.
    """
    _setup_logging()
    logger.info(f"Demarrage de {_settings.app_name} v0.1.0 ({_settings.app_env})")

    # Verifier la connexion a la base de donnees
    from greentech.data.storage.database import check_connection, engine
    from greentech.data.storage.models import Base

    db_ok = await check_connection()
    if not db_ok:
        logger.error("PostgreSQL inaccessible — l'API demarre en mode degrade")
    elif _settings.app_env == "production":
        # En production (Render, etc.), creer automatiquement les tables si
        # elles n'existent pas. Idempotent : ne touche pas aux tables
        # existantes. Permet un demarrage propre sur une base vierge sans
        # devoir executer manuellement init.sql.
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Tables SQLAlchemy verifiees/creees")
        except Exception as exc:
            logger.warning(f"Echec create_all (non bloquant) : {exc}")

    # Initialiser les metriques metier exposees a Prometheus (dashboard « Metier
    # GreenTech ») : modele en production + ratio Green IT / distribution lus
    # depuis PostgreSQL. Sans ce seed, les jauges restent a leur valeur par
    # defaut tant qu'aucune analyse temps-reel n'a tourne.
    if db_ok:
        from greentech.ai.mlops.monitoring import (
            refresh_business_metrics_from_db,
            seed_model_info_from_production,
        )

        seed_model_info_from_production()
        try:
            await refresh_business_metrics_from_db()
        except Exception as exc:
            logger.warning(f"Seed des metriques metier au demarrage echoue : {exc}")

    logger.info("API prete a recevoir des requetes")

    yield

    # Nettoyage a l'arret
    from greentech.data.storage.database import engine

    await engine.dispose()
    logger.info("API arretee proprement")


# === Creation de l'application FastAPI ===

app = FastAPI(
    title="GreenTech Intelligence API",
    description=(
        "API REST pour l'analyse et la classification automatique "
        "d'articles technologiques selon leur pertinence Green IT."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# === Middleware CORS ===

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Instrumentation Prometheus (metriques HTTP automatiques) ===
#
# ``prometheus-fastapi-instrumentator`` ajoute automatiquement les metriques
# standard a chaque requete HTTP :
#
# * ``http_requests_total{handler, method, status}`` : compteur de requetes
# * ``http_request_duration_seconds_*`` : histogramme de latence p50/p95/p99
# * ``http_request_size_bytes`` / ``http_response_size_bytes`` : taille payload
#
# Les metriques sont enregistrees dans le default registry de
# ``prometheus_client``, donc elles sont automatiquement exposees par
# l'endpoint ``/metrics`` defini plus bas (pas besoin de ``.expose()``).
#
# Les endpoints ``/health`` et ``/metrics`` sont exclus pour eviter du bruit
# dans les dashboards (les checks Prometheus eux-memes generent des requetes).
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402

Instrumentator(
    excluded_handlers=["/health", "/metrics"],
    should_group_status_codes=False,
).instrument(app)


# === Middleware de logging des requetes ===


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logue chaque requete HTTP avec sa duree de traitement."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration_ms:.0f}ms)"
    )

    return response


# === Gestionnaire d'erreurs global ===


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
    """Capture les erreurs non gerees pour eviter les crashs brutaux.

    Logue l'erreur complete et retourne une reponse 500 generique
    sans exposer les details internes (OWASP A09 : Security Logging).
    """
    logger.exception(f"Erreur non geree sur {request.method} {request.url.path}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du serveur"},
    )


# === Enregistrement des routes ===

app.include_router(auth_router)
app.include_router(articles_router)
app.include_router(analyze_router)
app.include_router(stats_router)


# === Endpoints utilitaires ===


@app.get(
    "/health",
    tags=["Systeme"],
    summary="Health check",
)
async def health_check() -> dict:
    """Verifie l'etat de sante de l'API et de ses dependances."""
    from greentech.api.schemas.stats import HealthResponse
    from greentech.data.storage.database import check_connection

    db_ok = await check_connection()

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database=db_ok,
        version="0.1.0",
    ).model_dump()


@app.get(
    "/metrics",
    tags=["Systeme"],
    summary="Metriques Prometheus",
    response_class=PlainTextResponse,
)
async def prometheus_metrics() -> PlainTextResponse:
    """Expose les metriques au format Prometheus pour le scraping.

    Inclut les metriques d'inference, de classification, de resume
    et d'empreinte carbone definies dans le module monitoring.
    """
    # Importer les metriques pour s'assurer qu'elles sont enregistrees
    import greentech.ai.mlops.monitoring  # noqa: F401

    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
