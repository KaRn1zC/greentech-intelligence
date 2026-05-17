"""Application Celery pour la file d'attente des analyses IA.

Cree une instance ``Celery`` partagee entre l'API FastAPI (cote producteur :
enqueue d'une tache via ``analyze.delay(...)``) et le worker (cote
consommateur : execute la tache, persiste le resultat dans Redis).

Architecture
------------

::

    ┌─────────────┐    POST /analyze    ┌──────────┐
    │  FastAPI    │ ──── enqueue ─────► │  Redis   │
    │  /analyze   │                     │  broker  │
    └──────┬──────┘                     │  (DB 0)  │
           │                            └────┬─────┘
           │ GET /analyze/{job_id}           │ pop
           │                                 ▼
           │                            ┌──────────┐
           │                            │  Celery  │
           │                            │  worker  │
           │                            │  (GPU)   │
           │                            └────┬─────┘
           │                                 │
           │                                 ▼
           │     read status            ┌──────────┐
           └──────────────────────────► │  Redis   │
                                        │  backend │
                                        │  (DB 1)  │
                                        └──────────┘

Le worker est lance separement de l'API :

.. code-block:: bash

    # Mode dev (Windows ou Linux), 1 tache GPU a la fois :
    uv run celery -A greentech.api.celery_app worker --pool=solo --loglevel=info

    # Mode prod Linux (prefork avec 1 worker, eviter sur GPU partage) :
    uv run celery -A greentech.api.celery_app worker --concurrency=1 --loglevel=info

Le pool ``solo`` est utilise par defaut car le classifieur Qwen3-4B occupe
~7.7 GB VRAM : un seul appel d'inference a la fois evite l'OOM. Pour scaler
horizontalement, lancer N workers sur N GPUs differents.
"""

from __future__ import annotations

from celery import Celery

from greentech.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "greentech",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=["greentech.api.tasks"],
)

celery_app.conf.update(
    # Serialisation JSON : portable, debugable, pas de code arbitraire execute.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Conservation des resultats : 24h par defaut. Au-dela on libere Redis.
    result_expires=_settings.celery_result_expires,
    # Limite l'execution d'une tache : 10 min max (pipeline complet ~30s
    # normalement, marge pour fallback local cold start).
    task_time_limit=_settings.celery_task_time_limit,
    task_soft_time_limit=_settings.celery_task_time_limit - 30,
    # Acknowledge la tache APRES execution : si le worker plante en cours
    # d'execution, la tache est re-mise dans la queue pour un autre worker.
    # Tradeoff : risque de double-execution en cas de timeout reseau, mais
    # le classifieur est idempotent par article_id (ON CONFLICT DO NOTHING
    # sur la table articles), donc acceptable.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Track started state : le statut passe de PENDING a STARTED au moment ou
    # le worker pick la tache. Permet a l'UI de distinguer "en attente" vs
    # "en cours" (sinon les deux apparaissent comme PENDING).
    task_track_started=True,
)
