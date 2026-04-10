# 🌍 GreenTech Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Plateforme d'analyse Green IT avec Intelligence Artificielle**

Application web complète pour la collecte, l'analyse et la classification automatique d'articles technologiques selon leur pertinence "Green IT" (informatique durable, éco-responsable).

---

## 📋 Table des Matières

- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Documentation](#-documentation)
- [Contribution](#-contribution)

---

## ✨ Fonctionnalités

### Bloc E1 : Data Factory
- 📥 Collecte multi-sources (API REST, Web Scraping, Fichiers)
- 🧹 Nettoyage et normalisation automatique (Apache Spark)
- 💾 Stockage Big Data (MinIO) + SQL (PostgreSQL)
- 🔒 Conformité RGPD

### Bloc E2 : Services IA (SaaS)
- 🔍 Veille technologique automatisée
- 📝 Résumé automatique via Hugging Face API
- 📊 Benchmark et sélection de services IA

### Bloc E3 : Modèle IA Custom (MLOps)
- 🤖 Classification Green IT — 3 modèles en compétition :
  - Champion : DeBERTa-v3-base (fine-tuning classique)
  - Challenger 1 : Qwen2.5-3B (fine-tuning LoRA)
  - Challenger 2 : Llama 3.2 3B (fine-tuning LoRA)
- 📈 Tracking expériences (MLflow)
- 🌱 Mesure empreinte carbone (CodeCarbon)
- ✅ Tests automatisés du modèle (Deepchecks)

### Bloc E4 : Application Full-Stack
- ⚡ API REST sécurisée (FastAPI + OAuth2/JWT)
- 🎨 Interface utilisateur moderne (React + Shadcn/UI)
- ♿ Accessibilité WCAG (Axe-core)

### Bloc E5 : DevOps & Monitoring
- 🔄 CI/CD (GitHub Actions)
- 🐳 Conteneurisation (Docker)
- 📊 Monitoring (Prometheus + Grafana + Loki)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│                    Shadcn/UI + TailwindCSS                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────▼───────────────────────────────────┐
│                     Backend API (FastAPI)                        │
│              OAuth2/JWT │ Pydantic │ SQLAlchemy                  │
└────────┬────────────────┼────────────────┬──────────────────────┘
         │                │                │
    ┌────▼────┐     ┌─────▼─────┐    ┌────▼────┐
    │PostgreSQL│     │  MinIO    │    │ ML Model│
    │  (SQL)   │     │(Big Data) │    │(PyTorch)│
    └──────────┘     └───────────┘    └─────────┘
```

---

## 📦 Prérequis

- **OS** : Windows 11 Pro
- **Python** : 3.12+
- **Node.js** : 20+ LTS
- **Docker** : Desktop 4.x
- **GPU** : AMD 7900 XTX + ROCm/HIP SDK (pour l'entraînement)
- **Git** : 2.40+

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/greentech-intelligence.git
cd greentech-intelligence
```

### 2. Installer les dépendances Python

```bash
# Installer UV (gestionnaire de paquets)
irm https://astral.sh/uv/install.ps1 | iex

# Créer l'environnement et installer les dépendances
uv sync

# Activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1
```

### 3. Configurer l'environnement

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env avec vos configurations
```

### 4. Lancer l'infrastructure Docker

```bash
# Services de base (PostgreSQL, MinIO, Monitoring)
docker-compose up -d

# Stack complète avec API et Frontend
docker-compose --profile full up -d
```

### 5. Initialiser la base de données

```bash
uv run python scripts/init_db.py
```

---

## 💻 Utilisation

### Développement

```bash
# Lancer l'API en mode développement
uv run uvicorn src.greentech.api.main:app --reload --port 8000

# Lancer le frontend
cd frontend && npm run dev
```

### Accès aux services

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | - |
| Frontend | http://localhost:5173 | - |
| MLflow | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |
| Grafana | http://localhost:3000 | admin / admin123 |
| Prometheus | http://localhost:9090 | - |

---

## 📚 Documentation

La documentation complète est disponible dans `/docs` :

- [Spécifications Techniques](docs/specs/)
- [Documentation API](docs/api/)
- [Guide MLOps](docs/mlops/)
- [Manuel Utilisateur](docs/user/)

Générer la documentation Sphinx :

```bash
cd docs && make html
```

---

## 🧪 Tests

```bash
# Tests unitaires
uv run pytest tests/unit -v

# Tests d'intégration
uv run pytest tests/integration -v

# Couverture de code
uv run pytest --cov=src/greentech --cov-report=html

# Tests d'accessibilité (Frontend)
cd frontend && npm run test:a11y
```

---

## 🤝 Contribution

Ce projet est développé dans le cadre d'un mémoire. Les contributions externes ne sont pas acceptées pour le moment.

---

## 📄 Licence

MIT License - voir [LICENSE](LICENSE)

---

## 👤 Auteur

**Arnaud "KaRn1zC" BOY**
- Projet de mémoire - [Titre Professionnel de niveau 6 - Développeur en Intelligence Artificielle et Data Analyst]
- [2025-2026]
