# GreenTech Intelligence

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Plateforme web d'analyse et classification automatique d'articles technologiques selon leur pertinence **Green IT** (informatique durable, eco-responsable).

Le systeme collecte des articles depuis plusieurs sources (API, scraping, fichiers), les nettoie via Apache Spark, puis applique une **classification hybride en deux etages** :

1. **Pre-filtre mots-cles permissif** (etage 1) : scoring multi-criteres qui distingue les articles manifestement Non Green IT (rejet direct) des **candidats** qui meritent une verification plus fine.
2. **LLM judge** (etage 2) : les candidats passent par un LLM instructif (`Qwen/Qwen3-4B-Instruct-2507`) qui tranche en zero-shot avec un prompt specialise. L'appel se fait via l'API Hugging Face Serverless, avec un **fallback automatique sur GPU AMD local** (`Qwen/Qwen2.5-3B-Instruct` via ROCm) si le quota mensuel HF est epuise (HTTP 402).

Un **classifieur de production** supplementaire (`Qwen/Qwen3-4B + LoRA`, Apache-2.0, multilingue natif FR/EN/DE/ES/ZH) fine-tune sur le golden dataset produit par cette classification hybride est utilise pour l'inference temps reel via l'endpoint `/analyze`. Il remplace depuis avril 2026 l'ancien modele `meta-llama/Llama-3.2-3B` (gated) et permet de classifier directement des articles dans plusieurs langues sans etape de traduction prealable.

Pour les articles confirmes Green IT, deux resumes sont generes via le meme LLM instructif Qwen : un resume general et un resume centre sur les aspects ecologiques.

---

## Table des matieres

1. [Prerequis](#1-prerequis)
2. [Cloner le projet](#2-cloner-le-projet)
3. [Configurer les secrets (.env)](#3-configurer-les-secrets-env)
4. [Installer les dependances](#4-installer-les-dependances)
5. [Lancer l'infrastructure Docker](#5-lancer-linfrastructure-docker)
6. [Recuperer le modele IA](#6-recuperer-le-modele-ia)
7. [Lancer l'application](#7-lancer-lapplication)
8. [Utiliser l'application](#8-utiliser-lapplication)
9. [Re-entrainement du modele](#9-re-entrainement-du-modele)
10. [Tests](#10-tests)
11. [Architecture technique](#11-architecture-technique)
12. [Deploiement production (Render)](#12-deploiement-production-render)

---

## 1. Prerequis

Logiciels a installer sur votre machine avant de commencer :

| Logiciel | Version minimum | Installation |
|----------|----------------|-------------|
| **Git** | 2.40+ | https://git-scm.com/downloads |
| **Python** | 3.12 | https://www.python.org/downloads/ |
| **Node.js** | 20 LTS | https://nodejs.org/ |
| **Docker Desktop** | 4.x | https://www.docker.com/products/docker-desktop/ |
| **UV** (gestionnaire Python) | latest | `irm https://astral.sh/uv/install.ps1 \| iex` (Windows) ou `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Linux/Mac) |

**Optionnel (pour entrainer les modeles)** :
- GPU AMD + ROCm/HIP SDK 7.1+ **ou** GPU NVIDIA + CUDA 12+
- Sans GPU, l'inference fonctionne en CPU (plus lent, ~5-15s par article)

---

## 2. Cloner le projet

```bash
git clone https://github.com/KaRn1zC/greentech-intelligence.git
cd greentech-intelligence
```

---

## 3. Configurer les secrets (.env)

Le fichier `.env` contient les secrets et parametres locaux. Il n'est pas versionne dans Git.

```bash
# Copier le template
cp .env.example .env
```

Ouvrez `.env` dans votre editeur et **remplacez les valeurs suivantes** :

### Secrets obligatoires a modifier

| Variable | Description | Ou l'obtenir |
|----------|-------------|-------------|
| `SECRET_KEY` | Cle de chiffrement de l'application | Generez une chaine aleatoire : `python -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_SECRET_KEY` | Cle de signature des tokens JWT | Generez une chaine aleatoire (meme commande que ci-dessus) |
| `HUGGINGFACE_TOKEN` | Token API Hugging Face pour le LLM judge et les resumes (ainsi que le telechargement du modele local en fallback) | Creez un compte sur https://huggingface.co, puis allez dans Settings > Access Tokens > New token (scope `read`) |

### Secrets optionnels

| Variable | Description | Quand la modifier |
|----------|-------------|------------------|
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL admin | Uniquement si vous changez la config Docker |
| `POSTGRES_APP_PASSWORD` | Mot de passe PostgreSQL applicatif | Uniquement si vous changez la config Docker |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | Identifiants MinIO | Uniquement si vous changez la config Docker |
| `API_NEWS_KEY` | Cle API NewsData.io pour la collecte | Uniquement si vous voulez relancer la collecte de donnees (https://newsdata.io) |
| `SCRAPING_USER_AGENT` | User-Agent pour le scraping | Remplacez `YOUR_USERNAME` par votre pseudo GitHub |

### Valeurs par defaut (ne pas modifier sauf besoin)

Les variables suivantes ont des valeurs par defaut fonctionnelles pour le developpement local. Ne les modifiez que si vous savez ce que vous faites :

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
MINIO_ENDPOINT=localhost:9000
MLFLOW_TRACKING_URI=http://localhost:5000
API_PORT=8000
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

---

## 4. Installer les dependances

### Backend (Python)

```bash
# Installer toutes les dependances (dev inclus)
uv sync

# Verifier que l'installation fonctionne
uv run python -c "from greentech.config import get_settings; print(get_settings().app_name)"
```

Vous devriez voir : `GreenTech Intelligence`

### Frontend (Node.js)

```bash
cd frontend
npm install
cd ..
```

---

## 5. Lancer l'infrastructure Docker

Docker fournit PostgreSQL, MinIO, MLflow, Prometheus, Loki et Grafana.

### Etape 1 : Lancer les services de base

```bash
# Demarrer PostgreSQL, MinIO, MLflow, Prometheus, Loki, Grafana
docker compose up -d
```

Attendez environ 30 secondes que tous les services demarrent. Verifiez :

```bash
docker compose ps
```

Tous les conteneurs doivent etre en status `Up` ou `healthy`.

### Etape 2 : Verifier que la base est initialisee

Le script `scripts/sql/init.sql` est execute automatiquement au premier demarrage de PostgreSQL via le volume Docker. Les tables, index et donnees de reference sont crees.

Pour verifier :

```bash
docker exec greentech-postgres psql -U greentech -d greentech_db -c "\dt"
```

Vous devriez voir les tables : `search_config`, `sources`, `articles`, `users`, `analysis_logs`, `daily_stats`.

### Etape 3 (optionnel) : Lancer la stack complete avec API + Frontend Docker

```bash
docker compose --profile full up -d
```

Cela ajoute les conteneurs `greentech-api` (port 8000) et `greentech-frontend` (port 80).

> **Note** : En developpement, il est plus pratique de lancer l'API et le frontend hors Docker (voir section 7).

### Etape 4 : Arreter l'infrastructure

```bash
docker compose --profile "*" down
```

Le wildcard `--profile "*"` active tous les profils declares dans `docker-compose.yml` et garantit que les conteneurs `greentech-api` et `greentech-frontend` (profil `full`) sont bien supprimes avec les services de base. Sans ce flag, `docker compose down` ignore les services du profil `full` et laisse le reseau `greentech-network` en etat `still in use`.

---

## 6. Recuperer le modele IA

Le modele entraine (Qwen3-4B + LoRA, ~30 Mo) est versionne via DVC et stocke dans MinIO. Les fichiers du modele dans `models/production/` sont inclus dans le depot Git pour simplifier l'utilisation.

### Option A : Utiliser les fichiers deja presents (recommande)

Les fichiers suivants sont deja dans `models/production/` :

```
models/production/
  adapter_config.json       # Configuration LoRA (r=16, alpha=32, target: attention + MLP)
  adapter_model.safetensors # Poids du modele fine-tune (~30 Mo)
  tokenizer.json            # Tokenizer du modele de base
  tokenizer_config.json     # Configuration du tokenizer
  README.md                 # Model Card (metriques, hyperparametres)
```

Au premier appel d'inference, le systeme :
1. Telecharge automatiquement le modele de base `Qwen/Qwen3-4B` depuis Hugging Face (~8 Go en BF16, cache local)
2. Charge les poids LoRA depuis `models/production/adapter_model.safetensors`
3. Met le modele en memoire pour les requetes suivantes

> **Important** : `Qwen/Qwen3-4B` est sous licence **Apache-2.0**, librement
> accessible sans demande d'acces. Il suffit d'un `HUGGINGFACE_TOKEN` valide
> dans `.env` (scope `read`) pour beneficier d'un debit de telechargement
> correct. Aucune acceptation de licence n'est requise.

### Option B : Recuperer via DVC (si les fichiers ne sont pas presents)

Si le dossier `models/production/` est vide (par exemple apres un clone partiel) :

```bash
# S'assurer que MinIO tourne
docker compose up -d minio

# Configurer les credentials DVC pour MinIO
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123

# Telecharger les artefacts du modele
dvc pull
```

Cela telecharge les 5 fichiers depuis le bucket MinIO `s3://models/dvc`.

### Parametres du modele de production

| Parametre | Valeur |
|-----------|--------|
| **Modele de base** | `Qwen/Qwen3-4B` (~4 milliards de parametres, dense transformer, multilingue natif) |
| **Licence** | Apache-2.0 (librement accessible, pas de gated access) |
| **Methode** | LoRA (Low-Rank Adaptation) via PEFT |
| **Rang LoRA (r)** | 16 |
| **Alpha LoRA** | 32 |
| **Dropout LoRA** | 0.05 |
| **Modules cibles** | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention uniquement, MLP exclu pour garder la VRAM sous 24 Go avec gradient checkpointing) |
| **Parametres entrainables** | ~20 M / 4 000 M (~0.5%) |
| **Taille des poids LoRA** | ~80 Mo (`adapter_model.safetensors`) |
| **Epochs** | 3 |
| **Learning rate** | 2e-4 |
| **Batch effectif** | 8 (batch 2 x gradient accumulation 4) |
| **Precision** | bf16 |
| **Max tokens** | 1024 |

### Benchmark historique inter-architectures

Les 3 architectures comparees lors du benchmark initial (septembre 2025 a fevrier 2026) :

| Metrique | DeBERTa-v3-base | Qwen2.5-3B + LoRA | Llama 3.2 3B + LoRA |
|----------|----------------|-------------------|----------------------|
| **F1** | 0.444 | 0.400 | 0.667 |
| **Accuracy** | 99.57% | 99.74% | 99.83% |
| **Precision** | 0.40 | 1.00 | 1.00 |
| **Recall** | 0.50 | 0.25 | 0.50 |
| **CO2** | 97.8 g | 108.8 g | 112.0 g |

Llama 3.2 3B + LoRA a ete le modele de production jusqu'en avril 2026.
Il est remplace par **Qwen3-4B + LoRA** pour trois raisons :
- **Multilinguisme natif** (FR/EN/DE/ES/ZH) : traitement direct d'articles
  non anglophones, sans etape de traduction.
- **Licence Apache-2.0** : pas de demande d'acces gated Meta.
- **Homogeneite du stack** : meme famille de tokenizer/chat template que le
  LLM judge et les summarizers deja en place (`Qwen/Qwen3-4B-Instruct-2507`).

Un benchmark **B4** est en cours pour comparer Qwen3-4B avec **mDeBERTa-v3-base**
(encoder multilingue 278M params) selon le **protocole unifie B3** (stratification
croisee `(langue x label)`, 3 seeds par fold, class_weight, back-translation
EN<->FR, calibration). Les metriques precises du modele actuellement en production
sont mises a jour dans `models/production/README.md` apres chaque promotion via
`scripts/retrain_pipeline.py auto-promote`.

---

## 7. Lancer l'application

### Mode developpement (recommande)

Ouvrez **deux terminaux** :

**Terminal 1 — API Backend** :

```bash
uv run uvicorn src.greentech.api.main:app --reload --port 8000
```

L'API est accessible sur http://localhost:8000

**Terminal 2 — Frontend** :

```bash
cd frontend
npm run dev
```

Le frontend est accessible sur http://localhost:5173

### Mode Docker (stack complete)

Si vous preferez tout lancer via Docker :

```bash
docker compose --profile full up -d --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:80 |
| API | http://localhost:8000 |

---

## 8. Utiliser l'application

### 8.1 Interfaces disponibles

| Interface | URL | Description |
|-----------|-----|-------------|
| **Frontend React** | http://localhost:5173 (dev) ou http://localhost:80 (Docker) | Interface utilisateur principale |
| **API Swagger** | http://localhost:8000/docs | Documentation interactive de l'API (tester les endpoints) |
| **API ReDoc** | http://localhost:8000/redoc | Documentation API en lecture seule |
| **MLflow** | http://localhost:5000 | Tracking des experiences ML (metriques, artefacts) |
| **Grafana** | http://localhost:3000 | Dashboards de monitoring (login : `admin` / `admin123`) |
| **Prometheus** | http://localhost:9090 | Metriques brutes de l'application |
| **MinIO Console** | http://localhost:9001 | Gestion du stockage objet (login : `minioadmin` / `minioadmin123`) |

### 8.2 Parcours utilisateur (Frontend)

#### Creer un compte

1. Ouvrez http://localhost:5173
2. Cliquez sur **"S'inscrire"** en bas du formulaire
3. Renseignez votre email et un mot de passe (8 caracteres minimum)
4. Cliquez sur **"S'inscrire"**
5. Vous etes automatiquement connecte et redirige vers le Dashboard

#### Analyser un article

Trois modes d'entree sont disponibles sur le **Dashboard** :

1. **URL** : collez un lien `http://` ou `https://` — le contenu de la page est extrait automatiquement
2. **Texte** : collez directement un extrait d'article (50 caracteres minimum)
3. **Fichier** : cliquez sur l'icone d'upload a cote du champ de saisie et choisissez
   un fichier local (formats supportes : `.txt`, `.md`, `.pdf`, `.docx`, `.html`, max 10 Mo)

Cliquez ensuite sur le bouton d'envoi (ou appuyez sur Entree). L'analyse se deroule
en arriere-plan (~1-30s selon que le modele est deja charge ou non) :

- Le LLM instructif **`Qwen/Qwen3-4B-Instruct-2507`** genere un **resume de classification** (style abstract scientifique, 150-220 mots) qui sert de feature d'entree au classifieur.
  - L'appel passe d'abord par l'API Hugging Face Serverless Inference (gratuit, fair-use).
  - En cas de quota mensuel epuise (HTTP 402), le dispatcher bascule automatiquement sur **`Qwen/Qwen2.5-3B-Instruct`** execute en local sur le GPU AMD RX 7900 XTX via ROCm 7.2, sans interruption du service.
- Le modele IA fine-tune (**Qwen3-4B + LoRA**) classifie l'article comme **Green IT** ou **Non Green IT** a partir du resume.
- Si l'article est classe Green IT, un second appel au meme LLM avec un prompt specifique genere un **resume des aspects ecologiques**, egalement en francais.

Le resultat s'affiche avec :
- Le statut Green IT (badge vert) ou Non Green IT (badge rouge)
- Le score de confiance du modele
- Le resume general de l'article
- Si Green IT : un bloc vert distinct avec les aspects ecologiques identifies
- Un lien vers la page de detail

#### Consulter les statistiques

Le Dashboard affiche un **camembert** avec la repartition :
- Articles Green IT (vert)
- Articles Non Green IT (rouge)
- Articles en attente d'analyse (gris)

#### Page de detail d'un article

Cliquez sur n'importe quel article dans la liste pour voir :
- Le titre, l'auteur, la date, la source
- Le badge de classification (Green IT / Non Green IT)
- La barre de score de confiance (pourcentage)
- Le modele utilise pour la classification
- Le resume IA
- Le contenu de l'article

### 8.3 Utiliser l'API directement (Swagger)

1. Ouvrez http://localhost:8000/docs
2. **Creer un compte** : POST `/auth/register` avec `{ "email": "...", "password": "..." }`
3. **Se connecter** : POST `/auth/login` — recuperez le `access_token`
4. **Autoriser** : Cliquez sur le bouton "Authorize" en haut a droite, collez le token
5. **Analyser par URL ou texte** : POST `/analyze` avec `{ "url": "https://..." }` ou `{ "texte": "..." }`
6. **Analyser par fichier** : POST `/analyze/file` en `multipart/form-data` avec le champ `fichier`
   (formats acceptes : .txt, .md, .pdf, .docx, .html ; max 10 Mo)
7. **Suivre l'analyse** : GET `/analyze/{job_id}` (le job_id est retourne par les etapes 5 ou 6)
8. **Lister les articles** : GET `/articles?page=1&limit=10&is_green_it=true`
9. **Statistiques** : GET `/stats`

### 8.4 Monitoring (Grafana)

1. Ouvrez http://localhost:3000 (login : `admin` / `admin123`)
2. Deux dashboards sont pre-configures :
   - **Performance Systeme** : latence HTTP, taux d'erreurs, etat des services
   - **Metier GreenTech** : nombre d'articles analyses, ratio Green IT, temps d'inference
3. Pour consulter les logs : allez dans **Explore** > selectionnez **Loki** > tapez `{container="greentech-api"}`

### 8.5 MLflow (suivi des experiences)

1. Ouvrez http://localhost:5000
2. L'experience `greentech-classification` contient les runs d'entrainement :
   - **Metriques** : F1, accuracy, precision, recall, loss
   - **Parametres** : hyperparametres de chaque run
   - **Artefacts** : modeles, tokenizers, rapports
   - **CodeCarbon** : empreinte CO2 de chaque entrainement

---

## 9. Re-entrainement du modele

### Comment ca fonctionne

Le corpus contient ~5 730 articles provenant de 4 sources actives, stockes dans PostgreSQL :
- **arXiv** (~5 000 articles) — ingere depuis un dump Kaggle (dataset Cornell University)
- **The Guardian** (~500 articles) — collecte via API REST/JSON (section environment + technology)
- **Dev.to** (~120 articles) — collecte via API REST/JSON (tags greenit, sustainability, etc.)
- **TechCrunch Climate** (~100 articles) — collecte via scraping hybride RSS + Playwright

> **Historique** : la source API initiale etait NewsData.io, desactivee en avril 2026 car son free tier tronquait le contenu au placeholder "ONLY AVAILABLE IN PAID PLANS". Remplacee par The Guardian + Dev.to qui fournissent le contenu integral.

Le pipeline de re-entrainement sert a **ajouter de nouveaux articles** depuis les sources en ligne, puis a re-entrainer le modele sur le corpus elargi (anciens + nouveaux). Les articles deja en base ne sont jamais perdus ni re-ingeres (deduplication par URL). Une etape `clean` supprime les articles inexploitables (contenu < 50 chars, placeholder NewsData) avant la generation des resumes.

Le modele est toujours re-entraine **depuis le modele de base** Qwen3-4B (pas depuis la version precedemment fine-tunee). Les poids LoRA sont recalcules entierement a chaque entrainement sur le dataset complet. La feature d'entrainement est `titre + resume_classification` (resume LLM style abstract scientifique) et non le contenu brut tronque.

La promotion en production est **conditionnelle** : le nouveau modele ne remplace l'ancien que s'il satisfait 4 criteres composites (MCC >= seuil, Recall Green IT >= 0.5, F1 non-regression, stabilite CV). L'application utilise donc toujours la meilleure version jamais entrainee.

### Pipeline complet (une seule commande)

```bash
uv run python scripts/retrain_pipeline.py
```

Cette commande execute **9 etapes dans l'ordre** (pipeline complet : collecte, nettoyage, resume, classification hybride, Golden Dataset et K-fold robuste) :

| Etape | Ce qui se passe |
|-------|----------------|
| **1. collect** | Interroge les 3 sources actives : **The Guardian** (API REST/JSON, 15 mots-cles Green IT, 5000 req/jour), **Dev.to** (tags `greenit`, `sustainability`, `climatechange`..., pas de cle), **TechCrunch Climate** (scraping hybride RSS pagine + Playwright headless, 100 articles max par session avec blacklist promos et fallback selecteurs + trafilatura). Nettoie via Spark et ingere dans PostgreSQL. Les doublons sont ignores (`ON CONFLICT (url) DO NOTHING`). |
| **2. clean** | **Supprime les articles inexploitables** de la base : contenu NULL, contenu < 50 caracteres (abstracts arXiv corrompus/tronques), et articles avec le placeholder NewsData `"ONLY AVAILABLE IN PAID PLANS"`. Idempotent et rapide (~1 s). Evite que ces articles polluent le dataset et fassent gonfler le compteur d'echecs du resume. |
| **3. summarize-classif** | Genere un **resume de classification** (style abstract scientifique, 150-220 mots, prompt centralise dans `classification_summarizer.py`) pour **chaque article** ayant `resume IS NULL`. Ce resume est la feature d'entree canonique du classifieur Qwen3-4B + LoRA et sert aussi d'affichage UI. Utilise le dispatcher HF Serverless avec fallback automatique sur Qwen2.5-3B local (GPU AMD ROCm) si le quota HF est epuise. |
| **4. annotate** (etage 1) | **Pre-filtre mots-cles permissif** : scoring multi-criteres sur les articles non encore classifies (`modele_classification IS NULL`). Chaque article est classe `NON_GREEN` (rejet direct, `est_green_it=false`) ou `CANDIDATE` (`est_green_it=NULL`, en attente du LLM judge). Marque `modele_classification='keyword_filter'`. |
| **5. classify** (etage 2) | **LLM judge** : envoie les articles `CANDIDATE` a `Qwen/Qwen3-4B-Instruct-2507` via l'API HF Serverless (fallback automatique sur Qwen2.5-3B local si quota HF epuise). Le LLM tranche avec un prompt zero-shot sur le **contenu brut de l'article** (pas le resume) pour maximiser la qualite du ground truth. Ecrit le verdict final (`est_green_it=true/false`, `score_confiance`, `modele_classification='keyword_filter+qwen_llm_judge'`). |
| **6. summarize-green** | Genere un **resume ecologique** specifique uniquement pour les articles confirmes Green IT (`est_green_it=true` + `resume_ecologique IS NULL`). Ce resume specialise est distinct du resume de classification et alimente la section "aspects ecologiques" dans la page detail de l'UI. |
| **7. export-golden** | Regenere `data/golden_dataset.csv` a partir de l'etat final de la DB. Exporte la colonne `articles.resume` comme `resume_classification` (feature d'entrainement). Les articles sans resume ou sans label sont exclus. |
| **8. train-cv** | Re-entraine **Qwen3-4B + LoRA** en **K-fold stratifie (K=5)** sur la feature `titre + resume_classification`, puis entraine un modele final sur l'integralite des donnees. Les metriques par fold et agregees sont trackees dans MLflow + `models/cv_report.json`. L'empreinte CO2 est mesuree par CodeCarbon. |
| **9. auto-promote** | Benchmark le nouveau modele vs le meilleur historique vs la baseline. Evalue 4 criteres composites : MCC >= seuil, Recall Green IT >= 0.5, F1 non-regression, stabilite CV. Si tous OK : archive l'ancien, copie le nouveau en production, enregistre ses metriques. Sinon : rien ne change, l'ancien reste en production. |

> **Note technique** : les etapes 2 a 6 (summarize-classif, annotate, classify, summarize-green, export-golden) s'executent en **appel Python direct** (async/await dans le meme process) plutot qu'en subprocess. Cette architecture evite la saturation du buffer PIPE stdout sous Windows qui causait des freezes systematiques avec les logs SQLAlchemy verbose. L'etape `collect` reste en subprocess car Spark necessite un process Python dedie pour sa JVM.

### Ajouter des fichiers manuellement

Si vous avez un fichier de donnees supplementaire (export JSON Lines, nouveau dump arXiv, etc.), deposez-le dans le dossier `data/` puis lancez :

```bash
# Ingere le fichier, resume, classifie (etages 1+2), resume eco, exporte le golden, re-entraine, promote
uv run python scripts/retrain_pipeline.py ingest-file data/mon_fichier.json summarize-classif annotate classify summarize-green export-golden train-cv auto-promote
```

Le fichier doit etre au format JSON Lines (une entree JSON par ligne, avec au minimum les champs `title` et `abstract` ou `content`). L'ingestion est idempotente : relancer sur le meme fichier ne cree pas de doublons.

### Etapes individuelles

Chaque etape peut etre lancee separement :

```bash
# Collecte seule (Guardian + Dev.to + TechCrunch + Spark cleaner + SQL ingester)
uv run python scripts/retrain_pipeline.py collect

# Nettoyage des articles inexploitables (contenu NULL, < 50 chars, placeholder NewsData)
uv run python scripts/retrain_pipeline.py clean

# Resume de classification pour tous les articles sans resume
uv run python scripts/retrain_pipeline.py summarize-classif

# Pre-filtre mots-cles seul (etage 1 : NON_GREEN ou CANDIDATE)
uv run python scripts/retrain_pipeline.py annotate

# LLM judge seul (etage 2 : classifie les CANDIDATE via Qwen + fallback local)
uv run python scripts/retrain_pipeline.py classify

# Resume ecologique pour les Green IT confirmes
uv run python scripts/retrain_pipeline.py summarize-green

# Alias : enchaine summarize-classif puis summarize-green
uv run python scripts/retrain_pipeline.py summarize

# Regenere golden_dataset.csv depuis l'etat final de la DB
uv run python scripts/retrain_pipeline.py export-golden

# Re-entrainement rapide (split 80/20 stratifie)
uv run python scripts/retrain_pipeline.py train

# Re-entrainement robuste (protocole unifie B3 : K=5 folds x 3 seeds, stratif (langue x label),
# class_weight, calibration post-fold, ~4-6h pour Qwen3 sur RX 7900 XTX)
uv run python scripts/retrain_pipeline.py train-cv

# Cibler un modele specifique
uv run python scripts/retrain_pipeline.py train-cv --model=mdeberta

# Augmentation par back-translation EN<->FR (opus-mt) sur les positifs (~5 min)
uv run python scripts/retrain_pipeline.py augment

# Benchmark seul (nouveau vs meilleur historique vs baseline)
uv run python scripts/retrain_pipeline.py benchmark

# Benchmark + promotion conditionnelle
uv run python scripts/retrain_pipeline.py auto-promote

# Promotion forcee (sans benchmark, a eviter sauf premier entrainement)
uv run python scripts/retrain_pipeline.py promote

# Baseline du modele brut (evaluation sans fine-tuning)
uv run python scripts/retrain_pipeline.py baseline
uv run python scripts/retrain_pipeline.py baseline --model=mdeberta  # Ciblee mDeBERTa
```

### Commandes B4 (benchmark equitable Qwen3 vs mDeBERTa)

```bash
# Baseline brute des 2 modeles cibles (~10 min cumulees)
uv run python scripts/retrain_pipeline.py baseline-both

# K-fold protocole unifie B3 sur Qwen3-4B PUIS mDeBERTa-v3-base (~6-8h cumulees)
uv run python scripts/retrain_pipeline.py train-cv-both

# Benchmark final B4.4 : compare les 2 modeles entraines, produit
# docs/BENCHMARK_FINAL_2026-04.md avec MCC/F1/latence p50-p95-p99/VRAM/CO2
uv run python scripts/retrain_pipeline.py benchmark-models
```

### Combiner des etapes

```bash
# Alimenter le dataset sans re-entrainer
uv run python scripts/retrain_pipeline.py collect clean summarize-classif annotate classify summarize-green export-golden

# Reprendre les articles en attente de LLM judge (apres reset quota HF)
uv run python scripts/retrain_pipeline.py classify

# Re-entrainer + benchmarker sans promouvoir
uv run python scripts/retrain_pipeline.py train benchmark

# Ingerer un fichier + pipeline complet
uv run python scripts/retrain_pipeline.py ingest-file data/export.json clean summarize-classif annotate classify summarize-green export-golden train-cv auto-promote
```

### Systeme de selection du meilleur modele

Le pipeline maintient 4 fichiers de reference :

| Fichier | Contenu | Quand il est mis a jour |
|---------|---------|------------------------|
| `models/best_metrics.json` | Metriques completes (MCC, F1, accuracy, balanced_accuracy, precision, recall, specificite, matrice de confusion, distribution des predictions) de la **meilleure version jamais entrainee**. | Uniquement quand un nouveau modele satisfait tous les criteres de promotion |
| `models/baseline_metrics.json` | Metriques du modele brut **sans fine-tuning**, evalue sur l'integralite du dataset (5730+ articles). Sert de reference permanente pour mesurer le gain du fine-tuning. | Une seule fois, via `baseline` (ou automatiquement si legacy format detecte) |
| `models/cv_report.json` | Rapport du dernier K-fold : metriques par fold (MCC, F1, recall, etc.), agregees (moyenne + ecart-type + min/max), et globales (sur concatenation des predictions). | A chaque execution de `train-cv` |
| `data/benchmark_versions.json` | Rapport du dernier benchmark (nouveau vs meilleur vs baseline + verdict + detail des 4 criteres). | A chaque benchmark |

**Logique de promotion composite** (4 criteres cumulatifs) :

| # | Critere | Constante (modifiable) | Raison |
|---|---------|------------------------|--------|
| 1 | `MCC_nouveau >= MCC_ancien - epsilon` | `MCC_EPSILON = 0.01` | **Metrique principale** : MCC est robuste au desequilibre (~0.4% de Green IT). |
| 2 | `Recall_Green_IT >= 0.5` | `MIN_RECALL_GREEN_IT = 0.5` | **Garde-fou metier** : empeche la promotion d'un modele qui "triche" en predisant tout en Non Green IT. |
| 3 | `F1_nouveau >= F1_ancien * 0.95` | `F1_REGRESSION_TOLERANCE = 0.95` | **Non-regression F1** : tolerance de 5% pour absorber le bruit statistique. |
| 4 | `std(MCC) entre folds <= 0.15` | `MAX_MCC_STD = 0.15` | **Stabilite CV** : applique uniquement si un rapport K-fold est utilise. |

Si **tous** les criteres passent → promotion. Si **un seul** echoue → l'ancien est conserve.

Pour le **premier modele** (pas de `best_metrics.json`) : les criteres 1 et 3 sont assouplis
(MCC > 0 au lieu de >= ancien), le critere 2 reste applique.

### Versioning des modeles

Chaque promotion archive automatiquement l'ancien modele :

```
models/
  production/              # Modele actif (utilise par l'API)
  qwen3/                   # Derniere version entrainee (Qwen3-4B + LoRA, modele actif)
    folds/                 # Adapters LoRA des 5 folds x 3 seeds (protocole unifie B3)
      fold_1_seed_1/ ...
    temperature.json       # Calibration : T optimal moyen (post-fold)
    optimal_threshold.json # Seuil de decision optimal (argmax MCC sur val)
  mdeberta/                # mDeBERTa-v3-base entraine pour le benchmark B4 (encoder)
    folds/                 # 5 folds x 3 seeds + temperature.json + optimal_threshold.json
  qwen2.5/                 # Legacy : Qwen2.5-3B + LoRA (archive)
  llama3.2/                # Legacy : Llama 3.2 3B + LoRA (archive)
  deberta-legacy/          # Legacy : DeBERTa-v3-base EN-only (archive)
  best_metrics.json        # Metriques completes du meilleur modele promu
  baseline_metrics.json    # Metriques zero-shot Qwen3-4B (defaut)
  baseline_metrics_mdeberta-v3-base.json  # Metriques zero-shot mDeBERTa
  cv_report.json           # Rapport K-fold complet (par defaut, dernier run)
  cv_report_qwen3.json     # Rapport K-fold Qwen3 (apres train-cv-both)
  cv_report_mdeberta.json  # Rapport K-fold mDeBERTa (apres train-cv-both)
  benchmark_final_metrics.json  # Comparatif B4.4 (apres benchmark-models)
  versions/
    v20260411_143022/      # Archive avec metadata.json + poids
    v20260415_091500/      # Archive suivante
```

Apres promotion, **redemarrez l'API** pour que le nouveau modele soit charge :

```bash
# En mode dev
# Ctrl+C dans le terminal de l'API, puis relancer :
uv run uvicorn src.greentech.api.main:app --reload --port 8000

# En mode Docker
docker compose restart api
```

### Entrainer les modeles individuellement (benchmark inter-architectures)

Pour relancer un entrainement isole d'une architecture (sans le pipeline B3
complet `retrain_pipeline.py`) :

```bash
# Entraine tous les modeles sequentiellement (les 5 du registre historique)
uv run python -m greentech.ai.models.training

# Ou entraine un modele specifique (identifiants courts depuis avril 2026)
uv run python -m greentech.ai.models.training qwen3       # Modele de production actuel (Qwen3-4B + LoRA)
uv run python -m greentech.ai.models.training mdeberta    # mDeBERTa-v3-base (encoder concurrent du benchmark B4)
uv run python -m greentech.ai.models.training llama3.2    # Legacy
uv run python -m greentech.ai.models.training qwen2.5     # Legacy
uv run python -m greentech.ai.models.training deberta     # DeBERTa-v3-base EN-only (legacy)

# Benchmark comparatif sur le test set commun
uv run python -m greentech.ai.models.training benchmark

# Evaluation zero-shot (baseline) isolee avec run MLflow dedie
uv run python scripts/benchmark_baseline.py
```

Ce workflow est independant du pipeline de re-entrainement et sert a verifier si une autre architecture serait plus performante sur le dataset actuel.

---

## 10. Tests

### Tests backend (Python)

```bash
# Tous les tests avec couverture
uv run pytest tests/ -v --cov=src/greentech

# Tests API uniquement
uv run pytest tests/api/ -v

# Tests modele IA (Deepchecks)
uv run pytest tests/ai/ -v
```

### Tests frontend (TypeScript)

```bash
cd frontend

# Verification TypeScript
npm run type-check

# Linting ESLint
npm run lint

# Tests accessibilite (necessite Playwright installe)
npx playwright install chromium
npm run test:a11y
```

### Qualite du code Python

```bash
# Linting + formatage
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
```

### Documentation Sphinx

La documentation du projet est generee avec Sphinx, MyST-Parser (Markdown) et le theme Furo.

```bash
# Generer la documentation HTML
cd docs
uv run sphinx-build -b html . _build/html
```

Pour la consulter, ouvrez directement le fichier dans votre navigateur :

```bash
# Windows
start docs/_build/html/index.html

# Linux / Mac
open docs/_build/html/index.html
```

Ou lancez un serveur local :

```bash
cd docs/_build/html
python -m http.server 8080
```

Puis ouvrez http://localhost:8080.

---

## 11. Architecture technique

```
greentech-intelligence/
├── src/greentech/               # Package Python principal
│   ├── config.py                # Configuration Pydantic Settings
│   ├── data/                    # Collecte, nettoyage, stockage (Bloc E1)
│   │   ├── collectors/          # API, scraping, fichiers
│   │   ├── processors/          # Nettoyage Spark
│   │   └── storage/             # PostgreSQL, MinIO, modeles ORM
│   ├── ai/                      # Intelligence artificielle (Blocs E2 & E3)
│   │   ├── services/            # summarizer.py (HuggingFace API)
│   │   ├── models/              # classifier.py, inference.py, training.py
│   │   └── mlops/               # tracking.py, validation.py, carbon.py
│   └── api/                     # API REST (Bloc E4)
│       ├── main.py              # App FastAPI (14 endpoints)
│       ├── routes/              # articles, analyze, auth, stats
│       ├── schemas/             # Pydantic (article, analysis, user, stats)
│       └── security/            # JWT auth (bcrypt + python-jose)
├── frontend/                    # Application React (Bloc E4)
│   ├── src/
│   │   ├── components/          # ui/ (shadcn), layout/ (Header, Footer)
│   │   ├── pages/               # Login, Dashboard, ArticleDetail
│   │   ├── hooks/               # useAuth
│   │   ├── lib/                 # api.ts, auth.ts
│   │   └── types/               # Miroir des schemas Pydantic
│   ├── tests/                   # Tests Playwright + Axe-core
│   ├── Dockerfile               # Production (nginx multi-stage)
│   └── nginx.conf               # Config NGINX (SPA, proxy, securite)
├── models/production/           # Modele IA retenu (LoRA weights)
├── tests/                       # Tests Python (api/, ai/)
├── config/                      # Prometheus, Grafana, Loki
├── scripts/sql/                 # init.sql (schema PostgreSQL)
├── .github/workflows/           # CI (ci.yml) + CD (cd.yml)
├── docker-compose.yml           # Stack complete
├── Dockerfile.api               # API multi-stage (uv + python:3.12-slim)
├── render.yaml                  # Blueprint deploiement Render
└── docs/                        # Documentation projet
    ├── PLAN_ETAPES.md           # Feuille de route detaillee
    ├── CHECKLIST_SUIVI.md       # Suivi competences diplome
    ├── PLAYBOOK_MAINTENANCE.md  # Procedures de maintenance
    └── PROCEDURE_MAJ_MODELE.md  # Mise a jour du modele IA
```

### Stack technique

| Couche | Technologies |
|--------|-------------|
| **Data** | httpx, feedparser (RSS), Scrapy + Playwright + scrapy-playwright (scraping HTML), PySpark, SQLAlchemy 2.0 async, PostgreSQL 15, MinIO |
| **IA** | PyTorch (ROCm/CUDA), transformers, PEFT (LoRA), scikit-learn (StratifiedKFold, MCC), Deepchecks, MLflow, DVC, CodeCarbon |
| **Backend** | FastAPI, Uvicorn, Pydantic 2, Loguru (+ interception logging standard), prometheus-client, pypdf + python-docx (parsing uploads) |
| **Frontend** | React 19, TypeScript 6, Vite 8, Tailwind CSS v4, shadcn/ui, recharts, Playwright + Axe-core |
| **DevOps** | Docker, GitHub Actions, Prometheus, Grafana, Loki, Render |

---

## 12. Deploiement production (Render)

Le projet est configure pour un deploiement automatique sur Render via le fichier `render.yaml`.

### Configuration

1. Creez un compte sur https://render.com
2. Liez votre depot GitHub
3. Rendez-vous dans **Blueprints** > **New Blueprint Instance**
4. Selectionnez le depot `greentech-intelligence`
5. Render detecte automatiquement `render.yaml` et cree :
   - Un **Web Service** pour l'API (Dockerfile)
   - Un **Static Site** pour le frontend (build Vite)
   - Une **base PostgreSQL**

### Secrets a configurer dans Render

Dans le dashboard Render > Environment :

| Variable | Valeur |
|----------|--------|
| `HUGGINGFACE_TOKEN` | Votre token HuggingFace |
| `CORS_ORIGINS` | URL de votre frontend Render |

Les variables `SECRET_KEY`, `JWT_SECRET_KEY` et `DATABASE_URL` sont generees automatiquement par le Blueprint.

### Deploiement automatique

Chaque push sur la branche `main` declenche :
1. Le pipeline CI (tests, linting, build)
2. Si le CI passe, le pipeline CD deploie sur Render

---

## Licence

MIT License - voir [LICENSE](LICENSE)

---

## Auteur

**Arnaud "KaRn1zC" BOY**

Projet de memoire — Titre Professionnel de niveau 6 — Developpeur en Intelligence Artificielle et Data Analyst (2025-2026)
