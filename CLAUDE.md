# CLAUDE.md - GreenTech Intelligence

## Projet

Plateforme web d'analyse et classification automatique d'articles technologiques selon leur pertinence "Green IT". Projet de diplome validant 5 blocs (E1-E5).

**Documents de reference (A CONSULTER IMPERATIVEMENT)** :

1. **`docs/PLAN_ETAPES.md`** : TOUJOURS se referer a ce fichier pour connaitre l'ordre precis de developpement. Avant de developper quoi que ce soit, consulter ce plan pour savoir exactement quoi faire et quelle est l'etape suivante.

2. **`docs/CHECKLIST_SUIVI.md`** : APRES CHAQUE developpement d'une nouvelle etape, TOUJOURS consulter cette checklist pour verifier si des cases doivent etre cochees. Si c'est le cas, les cocher immediatement pour maintenir le suivi a jour.

---

## Tech Stack

### Systeme
- **OS** : Windows 11 Pro | **Terminal** : PowerShell
- **IDE** : VSCode (extensions : Python, Ruff, Docker, MyST-Parser, Playwright)
- **PC Fixe** : Ryzen 9 7950X, 32 Gb DDR5, AMD Radeon RX 7900 XTX 24 Go + ROCm 7.1/7.2
- **PC Portable** : Ryzen AI 9 HX 370 (NPU inclus), 32 Gb DDR5, chipset graphique integre
- **Python** : 3.12 via UV (Astral)

### Data (E1)
httpx (async) | scrapy + playwright + scrapy-playwright | pyspark | sqlalchemy 2.0+ (async) + asyncpg | PostgreSQL 15 (Docker) | MinIO (S3)

### IA (E2 & E3)
PyTorch (ROCm/DirectML) | scikit-learn | transformers + huggingface_hub | peft (LoRA) + accelerate | deepchecks | mlflow | dvc + dvc-s3 | codecarbon

### Backend (E4)
FastAPI | uvicorn[standard] | fastapi-users[sqlalchemy] | pydantic 2.x + pydantic-settings | python-multipart | loguru | prometheus-client

### Frontend (E4)
Vite | React 18 + TypeScript | Tailwind CSS | shadcn/ui | lucide-react | @axe-core/playwright

### DevOps (E5)
GitHub Actions | Docker + docker-compose | Prometheus + Grafana + Loki | Render

### Documentation
Sphinx + myst-parser + furo

### Gestion de Projet
Scrum | GitHub Projects (Kanban) | Discord | Penpot (wireframes) | Looping (MCD/MLD Merise) | Inoreader + Perplexity Pro (veille)

---

## Structure Projet

```
greentech-intelligence/
├── src/greentech/               # Package Python principal
│   ├── config.py                # Settings Pydantic
│   ├── data/                    # === BLOC E1 ===
│   │   ├── collectors/          # api_collector.py, scraper.py, file_ingester.py, base.py
│   │   ├── processors/          # spark_cleaner.py, aggregator.py
│   │   └── storage/             # database.py, minio_client.py, models.py
│   ├── ai/                      # === BLOCS E2 & E3 ===
│   │   ├── services/            # summarizer.py (HuggingFace API)
│   │   ├── models/              # classifier.py, inference.py, training.py
│   │   └── mlops/               # tracking.py, validation.py, carbon.py
│   └── api/                     # === BLOC E4 Backend ===
│       ├── main.py              # App FastAPI
│       ├── dependencies.py
│       ├── routes/              # articles.py, analyze.py, stats.py, auth.py
│       ├── schemas/             # article.py, analysis.py, user.py
│       └── security/            # auth.py (JWT/OAuth2)
├── frontend/                    # === BLOC E4 Frontend ===
│   └── src/
│       ├── components/          # ui/, layout/, features/
│       ├── pages/               # Login, Dashboard, ArticleDetail
│       ├── hooks/
│       └── lib/                 # api.ts, auth.ts
├── tests/                       # unit/, integration/, ai/
├── docs/                        # Sphinx + specs + CHECKLIST + PLAN
├── config/                      # prometheus/, grafana/, loki/
├── scripts/sql/                 # init.sql
├── .github/workflows/           # ci.yml, cd.yml
├── Dockerfile.api
└── Dockerfile.frontend
```

---

## Commandes

```bash
# Environnement
uv sync                                    # Installer dependances
uv add <pkg> / uv add --dev <pkg>         # Ajouter dependance
uv run <cmd>                               # Executer dans venv

# Backend
uv run uvicorn src.greentech.api.main:app --reload --port 8000

# Qualite
uv run ruff check src/ tests/ --fix && uv run ruff format src/ tests/

# Tests
uv run pytest tests/ -v --cov=src/greentech
uv run pytest tests/ai -v                 # Deepchecks

# Frontend
cd frontend && npm install && npm run dev
cd frontend && npm run test:a11y           # Axe-core

# Docker
docker-compose up -d                       # Services base
docker-compose --profile full up -d        # Stack complete

# Data & ML
uv run python -m greentech.data.collectors.api_collector
uv run python -m greentech.data.processors.spark_cleaner
uv run mlflow ui --port 5000
uv run python -m greentech.ai.models.training

# Documentation
cd docs && uv run sphinx-build -b html . _build/html
```

---

## Conventions de Code

### Python
- **Ruff** : line-length=100
- **Type hints** : Obligatoires partout
- **Imports** : Absolus (`from greentech.module import X`)
- **Docstrings** : Google style, en **FRANCAIS**, ton professionnel et naturel (style humain, pas IA)
- **Logging** : loguru exclusivement (jamais `print()`)
- **Async** : Obligatoire pour toute I/O (DB, HTTP, fichiers)
- **Nommage** : `PascalCase` (classes), `snake_case` (fonctions/variables/fichiers), `UPPER_SNAKE_CASE` (constantes)

### Pattern de code standard

```python
"""Description du module."""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.data.storage.models import Article

if TYPE_CHECKING:
    from datetime import datetime


async def get_articles(
    db: AsyncSession,
    *,
    limit: int = 10,
    is_green_it: bool | None = None,
) -> list[Article]:
    """Recupere les articles depuis la base de donnees.

    Args:
        db: Session de base de donnees asynchrone.
        limit: Nombre maximum d'articles a retourner.
        is_green_it: Filtre par classification Green IT.

    Returns:
        Liste des articles correspondants aux criteres.

    Raises:
        DatabaseError: Si la requete echoue.
    """
    logger.debug(f"Recuperation articles : limit={limit}, is_green_it={is_green_it}")

    query = select(Article).limit(limit)
    if is_green_it is not None:
        query = query.where(Article.is_green_it == is_green_it)

    result = await db.execute(query)
    articles = result.scalars().all()

    logger.info(f"{len(articles)} articles recuperes")
    return list(articles)
```

### Commits
Format : `type(scope): message`
Types : `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `ci`

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/register` | Creer un compte | Non |
| POST | `/auth/login` | Connexion (JWT) | Non |
| POST | `/auth/logout` | Deconnexion | Oui |
| GET | `/auth/me` | Profil utilisateur | Oui |
| GET | `/articles` | Liste paginee (filtres: page, limit, is_green_it, source_id, date_from, date_to) | Non |
| GET | `/articles/{id}` | Detail article | Non |
| GET | `/articles/search` | Recherche | Non |
| POST | `/analyze` | Analyser URL ou texte | Oui |
| GET | `/analyze/{job_id}` | Statut analyse | Oui |
| GET | `/stats` | Stats globales | Non |
| GET | `/stats/daily` | Stats par jour | Non |
| GET | `/stats/sources` | Stats par source | Non |
| GET | `/health` | Health check | Non |
| GET | `/metrics` | Prometheus metrics | Non |

---

## Base de Donnees

**Tables** : `search_config`, `sources`, `articles`, `users`, `analysis_logs`, `daily_stats`
**Schema SQL complet** : `scripts/sql/init.sql`
**MinIO Buckets** : `raw-data` (brut), `clean-data` (nettoye), `models` (ML), `mlflow` (artifacts)

---

## Variables d'Environnement

```bash
APP_ENV=development
DEBUG=true
SECRET_KEY=<random-string>
DATABASE_URL=postgresql+asyncpg://greentech:password@localhost:5432/greentech_db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
HUGGINGFACE_TOKEN=hf_xxxxx
MLFLOW_TRACKING_URI=http://localhost:5000
JWT_SECRET_KEY=<random-string>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30
```

---

## Regles Critiques

1. **Async everywhere** : Toute I/O doit etre async
2. **RGPD** : Anonymiser donnees personnelles (auteurs, emails) avant stockage
3. **Scraping ethique** : Respecter robots.txt, delai 2s min, User-Agent identifie
4. **Green IT** : Mesurer empreinte carbone avec CodeCarbon
5. **Accessibilite** : Tous les composants UI doivent passer Axe-core (WCAG)
6. **Documentation** : Chaque module doit avoir sa doc Sphinx
7. **Tests** : Couverture minimale 80% sur nouveau code
8. **Git** : Ne jamais commiter sur main directement, toujours via PR
9. **Secrets** : Jamais de credentials dans le code, utiliser .env
10. **PyTorch** : `"cuda"` avec ROCm (PC Fixe) ou `torch_directml.device()` (PC Portable)

---

## Workflow de Developpement

### A chaque fin de session
1. **Cocher les taches terminees** dans `docs/PLAN_ETAPES.md`
2. **Cocher les competences validees** dans `docs/CHECKLIST_SUIVI.md`
3. **Mettre a jour le Kanban GitHub** : Voir instructions dans `docs/GESTION_KANBAN_GITHUB.md`

### A la fin de chaque ETAPE complete
**OBLIGATOIRE** : Avant de passer a l'etape suivante, executer cette checklist :

**📋 GUIDE RAPIDE** : `docs/CHECKLIST_FIN_ETAPE.md` (9 actions obligatoires)

1. ✅ Verifier `docs/GESTION_KANBAN_GITHUB.md` - Section de l'etape terminee
2. ✅ Cocher toutes les cases de l'issue GitHub correspondante
3. ✅ Modifier le statut de l'issue : `✅ TERMINÉ - Date`
4. ✅ Retirer label `en-cours` + Ajouter label `termine`
5. ✅ Deplacer l'issue vers colonne "Done"
6. ✅ Fermer l'issue sur GitHub
7. ✅ Cocher toutes les cases de l'etape dans `docs/PLAN_ETAPES.md`
8. ✅ Cocher les competences validees dans `docs/CHECKLIST_SUIVI.md`
9. ✅ Preparer l'etape suivante (deplacer issue vers "Ready")

**Fichiers de reference** :
- `docs/CHECKLIST_FIN_ETAPE.md` : Checklist rapide avec correspondance competences
- `docs/GESTION_KANBAN_GITHUB.md` : Detail complet pour chaque etape

---

## Attribution et Signature

**IMPORTANT** : Dans TOUS les fichiers (code, documentation, commentaires) :
- ✅ **Auteur** : KaRn1zC
- ❌ **NE JAMAIS** mentionner "Claude Code", "Claude", "IA", "AI", "Assistant", ou reference IA
- ❌ **NE JAMAIS** utiliser "genere automatiquement", "cree par IA", "Co-Authored-By: Claude"
- ✅ **Signature standard** : `Redige par KaRn1zC - YYYY-MM-DD`
