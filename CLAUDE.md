# CLAUDE.md - GreenTech Intelligence

## Project Overview

**GreenTech Intelligence** est une plateforme web complète d'analyse et de classification automatique d'articles technologiques selon leur pertinence "Green IT" (informatique durable, éco-responsable).

**Objectif** : Projet de mémoire validant 5 blocs de compétences (E1 à E5) du diplôme.

**Fonctionnalités principales** :
- Collecte automatisée multi-sources (API, Scraping, Fichiers)
- Nettoyage Big Data avec Apache Spark
- Classification IA custom (DeBERTa fine-tuné)
- Résumé automatique via Hugging Face Serverless API
- API REST sécurisée (FastAPI + OAuth2/JWT)
- Interface React accessible (WCAG)
- Pipeline CI/CD complet avec monitoring

---

## Tech Stack (Obligatoire)

### Système & Environnement
- **OS** : Windows 11 Pro
- **Terminal** : PowerShell (version récente)
- **IDE** : VSCode avec extensions Python, Ruff, Docker, MyST-Parser, Playwright
- **GPU** : AMD 7900 XTX + ROCm/HIP SDK 6.x+ (PC Fixe)
- **Python** : 3.12 via UV (gestionnaire Astral)

### Data (Bloc E1)
- **HTTP Client** : httpx (async)
- **Scraping** : scrapy + playwright + scrapy-playwright
- **Big Data** : pyspark (Apache Spark)
- **ORM** : sqlalchemy 2.0+ (async)
- **Driver PostgreSQL** : asyncpg
- **SGBD** : PostgreSQL 15 (Docker)
- **Stockage Objet** : MinIO (compatible S3)

### Intelligence Artificielle (Blocs E2 & E3)
- **Framework DL** : PyTorch (ROCm ou DirectML)
- **ML Classique** : scikit-learn
- **Transformers** : transformers + huggingface_hub
- **Fine-tuning** : peft (LoRA), accelerate
- **Tests IA** : deepchecks
- **MLOps** : mlflow (tracking), dvc + dvc-s3 (versioning données)
- **Green IT** : codecarbon (empreinte carbone)

### Backend API (Bloc E4)
- **Framework** : FastAPI
- **Serveur** : uvicorn[standard]
- **Auth** : fastapi-users[sqlalchemy] (OAuth2/JWT)
- **Validation** : pydantic 2.x, pydantic-settings
- **Upload** : python-multipart
- **Logging** : loguru
- **Métriques** : prometheus-client

### Frontend (Bloc E4)
- **Build** : Vite
- **Framework** : React 18 + TypeScript
- **CSS** : Tailwind CSS
- **Composants** : shadcn/ui
- **Icônes** : lucide-react
- **Tests A11y** : @axe-core/playwright

### DevOps (Bloc E5)
- **CI/CD** : GitHub Actions
- **Conteneurs** : Docker, docker-compose
- **Monitoring** : Prometheus + Grafana + Loki
- **Hébergement** : Render

### Documentation
- **Générateur** : Sphinx + myst-parser + furo

---

## Project Structure

```
greentech-intelligence/
├── CLAUDE.md                    # Ce fichier
├── README.md                    # Documentation projet
├── pyproject.toml               # Config Python/UV
├── .env                         # Variables (non versionné)
├── .env.example                 # Template variables
├── .gitignore
├── docker-compose.yml           # Infrastructure locale
│
├── src/greentech/               # Package Python principal
│   ├── __init__.py
│   ├── config.py                # Settings Pydantic
│   │
│   ├── data/                    # === BLOC E1 ===
│   │   ├── __init__.py
│   │   ├── collectors/          # Extraction données
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Classe abstraite collector
│   │   │   ├── api_collector.py # Module 1: API (httpx)
│   │   │   ├── scraper.py       # Module 2: Scrapy + Playwright
│   │   │   └── file_ingester.py # Module 3: Fichiers CSV/JSON
│   │   ├── processors/          # Traitement données
│   │   │   ├── __init__.py
│   │   │   ├── spark_cleaner.py # Nettoyage PySpark
│   │   │   └── aggregator.py    # Agrégation sources
│   │   └── storage/             # Accès stockage
│   │       ├── __init__.py
│   │       ├── database.py      # SQLAlchemy async
│   │       ├── minio_client.py  # Client MinIO/S3
│   │       └── models.py        # Modèles ORM
│   │
│   ├── ai/                      # === BLOCS E2 & E3 ===
│   │   ├── __init__.py
│   │   ├── services/            # Services IA SaaS (E2)
│   │   │   ├── __init__.py
│   │   │   └── summarizer.py    # HuggingFace Inference API
│   │   ├── models/              # Modèles custom (E3)
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py    # Fine-tuning DeBERTa/Llama
│   │   │   ├── inference.py     # Chargement et prédiction
│   │   │   └── training.py      # Scripts entraînement
│   │   └── mlops/               # MLOps (E3)
│   │       ├── __init__.py
│   │       ├── tracking.py      # MLflow integration
│   │       ├── validation.py    # Deepchecks tests
│   │       └── carbon.py        # CodeCarbon wrapper
│   │
│   ├── api/                     # === BLOC E4 Backend ===
│   │   ├── __init__.py
│   │   ├── main.py              # App FastAPI principale
│   │   ├── dependencies.py      # Injection dépendances
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── articles.py      # CRUD articles
│   │   │   ├── analyze.py       # Endpoint analyse IA
│   │   │   ├── stats.py         # Statistiques
│   │   │   └── auth.py          # Authentification
│   │   ├── schemas/             # Modèles Pydantic
│   │   │   ├── __init__.py
│   │   │   ├── article.py
│   │   │   ├── analysis.py
│   │   │   └── user.py
│   │   └── security/
│   │       ├── __init__.py
│   │       └── auth.py          # JWT/OAuth2 config
│   │
│   └── cli.py                   # CLI (optionnel)
│
├── frontend/                    # === BLOC E4 Frontend ===
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── components/          # Composants UI
│   │   │   ├── ui/              # shadcn/ui
│   │   │   ├── layout/          # Header, Footer, Layout
│   │   │   └── features/        # Composants métier
│   │   ├── pages/               # Pages application
│   │   │   ├── Login.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   └── ArticleDetail.tsx
│   │   ├── hooks/               # Custom hooks
│   │   ├── lib/                 # Utilitaires
│   │   │   ├── api.ts           # Client API
│   │   │   └── auth.ts          # Gestion auth
│   │   └── styles/
│   │       └── globals.css
│   └── tests/
│       └── a11y/                # Tests accessibilité
│
├── tests/                       # Tests Python
│   ├── __init__.py
│   ├── conftest.py              # Fixtures pytest
│   ├── unit/
│   │   ├── test_collectors.py
│   │   ├── test_processors.py
│   │   └── test_api.py
│   ├── integration/
│   │   ├── test_database.py
│   │   └── test_endpoints.py
│   └── ai/
│       └── test_model.py        # Tests Deepchecks
│
├── docs/                        # Documentation Sphinx
│   ├── conf.py
│   ├── index.md
│   ├── specs/                   # Spécifications techniques
│   │   ├── data_extraction.md   # Specs collecte
│   │   ├── api_design.md        # Specs API
│   │   └── benchmark_ia.md      # Benchmark services IA
│   ├── api/                     # Doc API
│   ├── mlops/                   # Doc MLOps
│   └── user/                    # Manuel utilisateur
│
├── config/                      # Configs Docker services
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   └── provisioning/
│   └── loki/
│       └── loki-config.yml
│
├── scripts/                     # Scripts utilitaires
│   ├── sql/
│   │   └── init.sql             # Init PostgreSQL
│   ├── init_db.py
│   └── seed_data.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml               # Pipeline CI
│       └── cd.yml               # Pipeline CD
│
├── Dockerfile.api               # Image API
└── Dockerfile.frontend          # Image Frontend
```

---

## Commands Reference

### Environment Management
```bash
uv sync                              # Installer dépendances
uv sync --extra rocm                 # Avec PyTorch ROCm
uv sync --extra directml             # Avec PyTorch DirectML
uv add <package>                     # Ajouter dépendance
uv add --dev <package>               # Ajouter dépendance dev
uv run <command>                     # Exécuter dans venv
.\.venv\Scripts\Activate.ps1         # Activer venv (Windows)
```

### Development
```bash
# API Backend
uv run uvicorn src.greentech.api.main:app --reload --port 8000

# Linting & Formatting
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/

# Tests
uv run pytest tests/ -v
uv run pytest tests/unit -v --cov=src/greentech
uv run pytest tests/integration -v
uv run pytest tests/ai -v  # Tests Deepchecks

# Frontend
cd frontend && npm install
cd frontend && npm run dev
cd frontend && npm run build
cd frontend && npm run test:a11y
```

### Docker Infrastructure
```bash
docker-compose up -d                           # Services base
docker-compose --profile full up -d            # Stack complète
docker-compose logs -f <service>               # Logs service
docker-compose down -v                         # Arrêter + supprimer volumes
```

### Data & ML
```bash
# Collecte données
uv run python -m greentech.data.collectors.api_collector
uv run python -m greentech.data.collectors.scraper
uv run python -m greentech.data.collectors.file_ingester

# Traitement Spark
uv run python -m greentech.data.processors.spark_cleaner

# MLflow
uv run mlflow ui --port 5000

# DVC
dvc init
dvc remote add -d minio s3://clean-data
dvc add data/dataset.csv
dvc push

# Entraînement modèle
uv run python -m greentech.ai.models.training
```

### Documentation
```bash
cd docs && uv run sphinx-build -b html . _build/html
```

---

## Coding Conventions

### Python Style
- **Formatter** : Ruff (line-length=100)
- **Type hints** : Obligatoires sur toutes les fonctions
- **Imports** : Absolus (`from greentech.module import X`)
- **Docstrings** : Google style
- **Logging** : loguru exclusivement (jamais print())
- **Async** : Privilégier async/await pour I/O (DB, HTTP)

### Naming Conventions
```python
# Classes : PascalCase
class ArticleCollector:
    pass

# Fonctions/variables : snake_case
def get_articles_by_date(start_date: datetime) -> list[Article]:
    pass

# Constantes : UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# Fichiers : snake_case.py
# api_collector.py, spark_cleaner.py
```

### Code Pattern Standard
```python
"""Module description."""
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
    """Fetch articles from database.

    Args:
        db: Async database session.
        limit: Maximum number of articles to return.
        is_green_it: Filter by Green IT classification.

    Returns:
        List of Article objects.

    Raises:
        DatabaseError: If query fails.
    """
    logger.debug(f"Fetching articles with limit={limit}, is_green_it={is_green_it}")
    
    query = select(Article).limit(limit)
    if is_green_it is not None:
        query = query.where(Article.is_green_it == is_green_it)
    
    result = await db.execute(query)
    articles = result.scalars().all()
    
    logger.info(f"Retrieved {len(articles)} articles")
    return list(articles)
```

### Commit Convention
Format : `type(scope): message`

Types : `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `ci`

Exemples :
```
feat(data): add API collector for NewsAPI
fix(api): handle 404 on article not found
docs(readme): update installation instructions
test(ai): add Deepchecks validation suite
refactor(processors): optimize Spark cleaning pipeline
ci(github): add accessibility tests to pipeline
```

---

## Database Schema

### Tables Principales

```sql
-- Configuration dynamique des recherches (Source SQL)
CREATE TABLE search_config (
    id SERIAL PRIMARY KEY,
    keyword VARCHAR(100) NOT NULL,
    source_url TEXT,
    source_type VARCHAR(20) CHECK (source_type IN ('api', 'scraping', 'file')),
    priority INTEGER DEFAULT 1,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sources de données
CREATE TABLE sources (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'scraping', 'file')),
    base_url TEXT,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    last_fetched_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Articles (table principale)
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    
    -- Contenu
    title VARCHAR(500) NOT NULL,
    url TEXT UNIQUE NOT NULL,
    content TEXT,
    summary TEXT,                    -- Résumé généré par IA SaaS
    
    -- Métadonnées
    author VARCHAR(200),
    published_at TIMESTAMP WITH TIME ZONE,
    language VARCHAR(10) DEFAULT 'en',
    
    -- Résultats classification IA
    is_green_it BOOLEAN,             -- Classification binaire
    confidence_score FLOAT,          -- Score confiance [0-1]
    classification_model VARCHAR(100),
    
    -- Traçabilité
    raw_data_path TEXT,              -- Chemin MinIO raw-data
    analyzed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Utilisateurs (FastAPI Users)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(320) UNIQUE NOT NULL,
    hashed_password VARCHAR(1024) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Logs d'analyse IA (MLOps monitoring)
CREATE TABLE analysis_logs (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    inference_time_ms INTEGER,
    carbon_emissions_kg FLOAT,       -- CodeCarbon
    prediction BOOLEAN,
    confidence FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Statistiques quotidiennes
CREATE TABLE daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_articles INTEGER DEFAULT 0,
    green_it_articles INTEGER DEFAULT 0,
    non_green_it_articles INTEGER DEFAULT 0,
    avg_confidence_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### MinIO Buckets
- `raw-data` : Données brutes (HTML, JSON non traités)
- `clean-data` : Données nettoyées (Parquet/JSON)
- `models` : Modèles ML versionnés
- `mlflow` : Artifacts MLflow

---

## API Endpoints Specification

### Authentication
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/register` | Créer un compte | Non |
| POST | `/auth/login` | Connexion (retourne JWT) | Non |
| POST | `/auth/logout` | Déconnexion | Oui |
| GET | `/auth/me` | Profil utilisateur | Oui |

### Articles
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/articles` | Liste paginée | Non |
| GET | `/articles/{id}` | Détail article | Non |
| GET | `/articles/search` | Recherche | Non |

Query params pour `/articles` :
- `page` (int) : Numéro page
- `limit` (int) : Articles par page (max 100)
- `is_green_it` (bool) : Filtrer par classification
- `source_id` (int) : Filtrer par source
- `date_from` (date) : Date début
- `date_to` (date) : Date fin

### Analysis
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/analyze` | Analyser URL ou texte | Oui |
| GET | `/analyze/{job_id}` | Statut analyse | Oui |

Request body `/analyze` :
```json
{
  "url": "https://example.com/article",
  "text": null
}
```
ou
```json
{
  "url": null,
  "text": "Contenu de l'article à analyser..."
}
```

### Statistics
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/stats` | Stats globales | Non |
| GET | `/stats/daily` | Stats par jour | Non |
| GET | `/stats/sources` | Stats par source | Non |

### Monitoring
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/health` | Health check | Non |
| GET | `/metrics` | Prometheus metrics | Non |

---

## Validation Criteria by Block

### 🟦 BLOC E1 : Data (Gérer des données)

#### A1. Programmer la collecte de données

**C1. Automatiser l'extraction**
- [ ] Documenter contraintes techniques sources (API limits, robots.txt)
- [ ] Rédiger specs techniques extraction (`docs/specs/data_extraction.md`)
- [ ] Module API : requêtes HTTP avec httpx (headers, timeouts, retry)
- [ ] Module Scraping : Spider Scrapy + Playwright (sites JS)
- [ ] Module Fichiers : lecture CSV/JSON
- [ ] Connexion SGBD PostgreSQL via SQLAlchemy async
- [ ] Connexion Big Data : lecture depuis MinIO via Spark
- [ ] Parsing/filtrage données de chaque source
- [ ] Script structuré (init, traitement, erreurs, sauvegarde)
- [ ] Code versionné sur GitHub

**C2. Requêtes SQL**
- [ ] Requêtes extraction depuis PostgreSQL
- [ ] Requêtes extraction depuis système Big Data (Spark SQL)
- [ ] Documentation des requêtes avec justification

**C3. Agrégation et nettoyage**
- [ ] Specs techniques agrégation
- [ ] Agrégation 3 sources en DataFrame unique
- [ ] Détection entrées corrompues/incomplètes
- [ ] Suppression entrées invalides
- [ ] Homogénéisation formats (dates ISO 8601, encodages)
- [ ] Script versionné avec documentation complète

#### A2. Mise à disposition des données

**C4. Base de données + RGPD**
- [ ] Specs techniques stockage
- [ ] Modélisation Merise : MCD sur Looping
- [ ] Génération MLD et script SQL
- [ ] SGBD PostgreSQL opérationnel (Docker)
- [ ] Documentation installation SGBD
- [ ] Registre traitements RGPD
- [ ] Procédures tri données personnelles (anonymisation)
- [ ] Script import fonctionnel et documenté

**C5. API REST**
- [ ] Specs techniques API (`docs/specs/api_design.md`)
- [ ] Connexion données depuis serveur API
- [ ] Validation requêtes client (Pydantic)
- [ ] Requêtes BDD déclenchées par API
- [ ] Réponses JSON formatées
- [ ] Règles autorisation/accès endpoints
- [ ] Sécurité OWASP Top 10
- [ ] Documentation OpenAPI/Swagger complète

---

### 🟪 BLOC E2 : Services IA (Veille & SaaS)

#### A3. Choix et intégration service IA

**C6. Veille technique**
- [ ] Thématiques définies : Green IT, Sustainable AI, Model Efficiency
- [ ] Planning veille hebdomadaire
- [ ] Outil agrégation : Inoreader (flux RSS)
- [ ] Outil recherche : Perplexity Pro (synthèses auto)
- [ ] Sources fiables identifiées et qualifiées
- [ ] Configuration outils
- [ ] Synthèses mensuelles rédigées (Markdown)
- [ ] Communication synthèses (format accessible)

**C7. Benchmark services IA**
- [ ] Reformulation besoin : résumé automatique articles
- [ ] Contraintes identifiées (coût, technique, ops)
- [ ] Benchmark documenté (`docs/specs/benchmark_ia.md`)
- [ ] Critères : Coût, Intégration, Impact Carbone, Qualité
- [ ] Services comparés : OpenAI, Mistral, Hugging Face
- [ ] Raisons exclusion explicitées
- [ ] Adéquation fonctionnelle détaillée
- [ ] Démarche éco-responsable analysée
- [ ] Conclusion : Hugging Face Serverless Inference API

**C8. Paramétrer service IA**
- [ ] Compte Hugging Face configuré
- [ ] SDK huggingface_hub installé
- [ ] Gestion accès (API token)
- [ ] Module `summarizer.py` opérationnel
- [ ] Service répond aux besoins fonctionnels
- [ ] Documentation technique complète et accessible

---

### 🟧 BLOC E3 : MLOps (Modèle IA custom)

#### A4. Intégration modèle IA

**C9. API exposant modèle**
- [ ] Analyse specs fonctionnelles/techniques
- [ ] Architecture API conçue (endpoint `/analyze`)
- [ ] Environnement dev configuré (PyTorch ROCm/DirectML)
- [ ] Validation/transformation paramètres requête
- [ ] Exécution modèle depuis requête
- [ ] Réponse formatée au client
- [ ] Auth/autorisation sur endpoint
- [ ] Sécurité OWASP
- [ ] Tests intégration endpoints
- [ ] Code versionné, doc OpenAPI

**C10. Intégration API dans application**
- [ ] Frontend connecté à l'API IA
- [ ] Auth programmée côté client
- [ ] Communication endpoints fonctionnelle
- [ ] Adaptations interface (loading, résultats)
- [ ] Accessibilité interfaces testée
- [ ] Tests intégration côté app
- [ ] Code versionné

#### A5. Déploiement MLOps

**C11. Monitoring modèle**
- [ ] Métriques définies : drift, latence, % Green IT, santé système
- [ ] Déclencheurs réentraînement définis
- [ ] Prometheus configuré pour scraper métriques
- [ ] Dashboard Grafana opérationnel
- [ ] Alertes configurées (latence > 2s)
- [ ] Accessibilité dashboard
- [ ] Code versionné, documentation complète

**C12. Tests automatisés modèle**
- [ ] Périmètre tests défini (format data, entraînement, éval)
- [ ] Deepchecks configuré
- [ ] Suite tests implémentée :
  - Data Leakage
  - Biais détection
  - Robustesse au bruit
- [ ] Rapport validation généré
- [ ] Code + données (DVC) versionnés
- [ ] Documentation tests

**C13. CI/CD pour IA**
- [ ] Pipeline défini (étapes, déclencheurs)
- [ ] Variables env configurées
- [ ] Étape test données intégrée
- [ ] Étape test/entraînement/validation modèle
- [ ] Génération rapports (confusion matrix, accuracy)
- [ ] Étape livraison (PR auto ou deploy)
- [ ] Fichiers CI/CD versionnés et documentés

---

### 🟩 BLOC E4 : Application (Dev Full-Stack)

#### A6. Conception

**C14. Analyse besoin**
- [ ] MCD/MLD sur Looping (formalisme Merise)
- [ ] Wireframes parcours utilisateur (Penpot)
- [ ] User Stories avec contexte, scénarios, critères validation
- [ ] Objectifs accessibilité dans critères (WCAG 2.1 AA)

**C15. Cadre technique**
- [ ] Specs techniques : architecture, dépendances, environnement
- [ ] Choix éco-responsables documentés (Green IT)
- [ ] Diagramme flux données
- [ ] POC réalisée et accessible
- [ ] Conclusion POC validant poursuite projet

#### A7. Développement

**C16. Coordination Agile**
- [ ] Cycles Scrum respectés
- [ ] Kanban GitHub Projects à jour
- [ ] Rituels documentés (sprint planning, review, retro)
- [ ] Backlog accessible à tous

**C17. Composants et interfaces**
- [ ] Environnement dev conforme specs
- [ ] Interfaces respectent maquettes Penpot
- [ ] Comportements (validation, navigation) conformes
- [ ] Composants métier fonctionnels :
  - Formulaire analyse (URL/Texte)
  - Liste articles (table/cards)
  - Détail article avec résultat IA
  - Dashboard statistiques
- [ ] Gestion droits/accès implémentée
- [ ] Éco-conception respectée
- [ ] OWASP Top 10 implémenté
- [ ] Tests unitaires/intégration
- [ ] Code versionné, documentation technique

#### A8. Tests et Contrôle

**C18. CI automatisée**
- [ ] GitHub Actions configuré
- [ ] Étapes build intégrées
- [ ] Tests exécutés automatiquement :
  - Ruff (lint)
  - Pytest (backend)
  - Deepchecks (IA)
  - Axe-core (accessibilité)
- [ ] Config versionnée, documentation

**C19. CD automatisée**
- [ ] Pipeline CD configuré
- [ ] Packaging : Docker build, minification frontend
- [ ] Déploiement Render automatique sur push main
- [ ] Documentation complète

---

### 🟥 BLOC E5 : Maintenance & Monitoring

#### A9. Maintien en condition opérationnelle

**C20. Surveillance application**
- [ ] Métriques, seuils, alertes documentés
- [ ] Justification choix outils (Prometheus/Grafana/Loki)
- [ ] Collecteurs métriques installés
- [ ] Règles journalisation (Loguru) intégrées
- [ ] Alertes fonctionnelles
- [ ] Dashboard Grafana :
  - Performance système (CPU, RAM, latence)
  - Métier (articles analysés, ratio Green IT)
- [ ] Documentation installation monitoring

**C21. Résolution incidents**
- [ ] Procédure identification cause problème
- [ ] Reproduction en env dev documentée
- [ ] Procédure débogage (lecture logs Grafana)
- [ ] Solution documentée étape par étape
- [ ] Fix versionné (Merge Request)
- [ ] Simulation incidents (chaos engineering léger) :
  - Coupure BDD
  - Indisponibilité API Hugging Face
- [ ] Messages erreur utilisateur compréhensibles

---

## Development Workflow

### 1. Créer une branche feature
```bash
git checkout -b feat/E1-api-collector
```

### 2. Développer avec TDD
```bash
# Écrire test d'abord
uv run pytest tests/unit/test_collectors.py -v -k "test_api_collector"

# Implémenter jusqu'à test vert
# Refactorer si nécessaire
```

### 3. Vérifier qualité
```bash
uv run ruff check --fix && uv run ruff format
uv run pytest tests/ -v --cov=src/greentech
```

### 4. Commit et push
```bash
git add .
git commit -m "feat(data): implement API collector with httpx"
git push origin feat/E1-api-collector
```

### 5. Pull Request
- Créer PR vers `develop` ou `main`
- CI doit passer
- Review si applicable

---

## Module Implementation Guide

### Bloc E1 - Module 0 : Config SQL
```python
# src/greentech/data/collectors/config_loader.py
"""Load search configuration from PostgreSQL."""

async def get_config_from_db(db: AsyncSession) -> list[SearchConfig]:
    """Extract search keywords and URLs from search_config table."""
    result = await db.execute(
        select(SearchConfigModel).where(SearchConfigModel.active == True)
    )
    return result.scalars().all()
```

### Bloc E1 - Module 1 : API Collector
```python
# src/greentech/data/collectors/api_collector.py
"""Collect articles from REST APIs (NewsAPI, etc.)."""

class APICollector(BaseCollector):
    async def collect(self, keywords: list[str]) -> list[RawArticle]:
        """Fetch articles from configured APIs."""
        # httpx async client
        # Handle 4xx/5xx errors
        # Save raw JSON to MinIO raw-data bucket
```

### Bloc E1 - Module 2 : Scraper
```python
# src/greentech/data/collectors/scraper.py
"""Scrapy spider with Playwright for JS rendering."""

class TechBlogSpider(scrapy.Spider):
    # Use scrapy-playwright for dynamic content
    # Respect robots.txt and rate limits
    # Extract full HTML content
    # Save to MinIO raw-data bucket
```

### Bloc E2 - Summarizer
```python
# src/greentech/ai/services/summarizer.py
"""HuggingFace Serverless Inference API wrapper."""

class Summarizer:
    def __init__(self, model: str = "facebook/bart-large-cnn"):
        self.client = InferenceClient(token=settings.huggingface_token)
    
    async def summarize(self, text: str) -> str:
        """Generate summary using HF Inference API."""
```

### Bloc E3 - Classifier
```python
# src/greentech/ai/models/classifier.py
"""DeBERTa fine-tuning for Green IT classification."""

class GreenITClassifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        pass
    
    def train(self, dataset_path: str, output_dir: str):
        """Fine-tune model with MLflow tracking and CodeCarbon."""
    
    def predict(self, text: str) -> tuple[bool, float]:
        """Return (is_green_it, confidence_score)."""
```

### Bloc E4 - API Main
```python
# src/greentech/api/main.py
"""FastAPI application entry point."""

app = FastAPI(
    title="GreenTech Intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for React frontend
# Prometheus metrics endpoint
# Health check endpoint
# Include routers
```

---

## Environment Variables

```bash
# Application
APP_ENV=development
DEBUG=true
SECRET_KEY=<random-string>

# Database
DATABASE_URL=postgresql+asyncpg://greentech:password@localhost:5432/greentech_db

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Hugging Face
HUGGINGFACE_TOKEN=hf_xxxxx

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# JWT
JWT_SECRET_KEY=<random-string>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30
```

---

## Important Notes

1. **Async everywhere** : Toutes les opérations I/O doivent être async (DB, HTTP, fichiers)

2. **RGPD** : Anonymiser les données personnelles (noms auteurs, emails) avant stockage

3. **Scraping éthique** : Respecter robots.txt, délais entre requêtes (2s min), User-Agent identifié

4. **Green IT** : Mesurer empreinte carbone avec CodeCarbon pendant entraînement

5. **Accessibilité** : Tous les composants UI doivent passer Axe-core sans erreur critique

6. **Documentation** : Chaque module doit avoir sa doc Sphinx mise à jour

7. **Tests** : Couverture minimale 80% sur nouveau code

8. **Git** : Ne jamais commiter sur main directement, toujours via PR

9. **Secrets** : Jamais de credentials dans le code, utiliser .env

10. **PyTorch device** : Utiliser `torch_directml.device()` sur Windows ou `"cuda"` avec ROCm
