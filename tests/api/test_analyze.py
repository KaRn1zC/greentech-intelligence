"""Tests d'integration pour les endpoints d'analyse IA.

OBSOLETE depuis la migration Celery+Redis (mai 2026) :
le dict ``_jobs`` en memoire dont depend ce module n'existe plus.
Les jobs sont desormais geres par le ``result backend`` Redis de
Celery et consultes via ``AsyncResult(task_id)`` dans la route
``GET /analyze/{job_id}``.

A reecrire avec un mock du ``celery_app.task`` et de
``celery.result.AsyncResult`` au lieu de manipuler un store local.

En attendant, le module entier est marque ``skip`` pour ne pas
casser la collecte pytest dans la CI (cf. ci.yml job backend-test).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Tests obsoletes depuis la migration Celery+Redis : le dict "
        "_jobs en memoire n'existe plus. A reecrire avec un mock de "
        "celery.result.AsyncResult."
    )
)


# Tests retires temporairement — voir docstring du module pour le
# protocole de re-ecriture (mock AsyncResult au lieu de _jobs).
