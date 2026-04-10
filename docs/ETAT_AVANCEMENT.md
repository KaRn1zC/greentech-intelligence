# Etat d'avancement du projet - GreenTech Intelligence

> **Derniere mise a jour** : 2026-04-10
> **Redige par** : KaRn1zC

---

## Ou nous en sommes

### ETAPE 1 : Installation & Configuration - **TERMINEE**
Toutes les dependances et outils sont installes et configures.

### ETAPE 2 : Data Factory & Gestion de Donnees (Bloc E1) - **TERMINEE**
Pipeline complet operationnel : 3 sources de collecte, nettoyage PySpark, ingestion PostgreSQL.

### ETAPE 3 : Intelligence Artificielle (Blocs E2 & E3) - **EN COURS (~85%)**

#### Section 3.1 : Veille Technologique & Benchmark - **TERMINEE**
- Veille Inoreader + Perplexity Pro configuree
- Benchmark services IA realise (choix : HuggingFace Serverless API)
- Module summarizer.py developpe et fonctionnel

#### Section 3.2 : Preparation des Donnees & MLOps - **TERMINEE**
- **Golden Dataset** : CREE et annote (5808 articles, 22 Green IT / 5786 Non Green IT)
- **DVC** : Initialise, remote MinIO configure, dataset versionne et pousse (s3://models/dvc)

#### Section 3.3 : Entrainement & Competition des Modeles - **EN COURS**

Architecture a 3 modeles en competition (au lieu de 2 initialement) :

| Modele | Type | Params | Methode | Statut | F1 | Accuracy | CO2 |
|--------|------|--------|---------|--------|-----|----------|-----|
| **Champion** DeBERTa-v3-base | Encoder | 86M | Full fine-tuning | ENTRAINE | 0.44 | 99.6% | 97.8g |
| **Challenger 1** Qwen2.5-3B | Causal LM | 3085M | LoRA (r=16) | ENTRAINE | 0.40 | 99.74% | 108.8g |
| **Challenger 2** Llama 3.2 3B | Causal LM | 3213M | LoRA (r=16) | ENTRAINE | **0.667** | **99.83%** | 112.0g |

- Oversampling de la classe minoritaire (22 → 1152, ratio 20%)
- Benchmark final execute sur 1162 articles de test
- **VAINQUEUR : Llama 3.2 3B + LoRA** (meilleur F1=0.667, precision 1.0, recall 0.50)

#### Section 3.4 : Validation & Packaging - **PARTIELLEMENT FAIT**
- Tests Deepchecks ecrits
- **Reste** : packaging modele gagnant (safetensors), push DVC, redaction Model Card

#### Section 3.5 : Deploiement MLOps (Monitoring) - **TERMINE**
- Metriques de production definies (14 metriques Prometheus)
- Configuration Prometheus preparee (scrape API, MLflow, MinIO)
- Stack Prometheus + Loki + Grafana operationnelle via docker-compose
- 2 dashboards Grafana provisionnes (Performance Systeme + Metier GreenTech)
- MLflow Tracking Server Docker (PostgreSQL backend + MinIO S3 artifacts)
- CodeCarbon integre dans le tracking MLflow (mesure CO2 par run)

### ETAPE 4 : Backend & API (Blocs E1 & E4) - **AVANCE PARTIELLE**

> Travail anticipe lors de la session du 2026-03-13. L'etape 3 doit etre terminee avant de finaliser l'etape 4.

#### Ce qui a ete fait :
- Architecture API FastAPI complete (14 endpoints)
- Schemas Pydantic (article, user, analysis, stats)
- Authentification JWT (bcrypt + python-jose)
- Routes : auth (4), articles (3), stats (3), analyze (2), health (1), metrics (1)
- Suite de tests d'integration : 27 tests, tous passent
- Middleware logging, CORS, gestion d'erreurs globale

#### Ce qui reste :
- Finaliser apres completion de l'etape 3
- Integration effective du modele IA vainqueur dans l'endpoint /analyze
- Documentation OpenAPI/Swagger

---

## Travail effectue lors de cette session (2026-04-10)

### 1. Entrainement des modeles IA (3 modeles)

**Refactoring architecture training** :
- `training.py` refactorise pour supporter 3 modeles via CLI
- Factory `_build_classifier_and_config()` pour instancier champion, challenger-qwen ou challenger-llama
- `ChallengerClassifier` rendu generique (LoRA sur tout causal LM)
- Benchmark `benchmark_models()` compare dynamiquement tous les modeles disponibles

**Commandes CLI** :
```bash
uv run python -m greentech.ai.models.training                  # Les 3 modeles
uv run python -m greentech.ai.models.training champion-deberta # Champion seul
uv run python -m greentech.ai.models.training challenger-qwen  # Challenger Qwen seul
uv run python -m greentech.ai.models.training challenger-llama # Challenger Llama seul
uv run python -m greentech.ai.models.training benchmark        # Comparaison
```

**Champion DeBERTa-v3-base** (entraine le 2026-03-13) :
- F1=0.44, Accuracy=99.6%, Precision=0.40, Recall=0.50
- CodeCarbon: 97.8g CO2eq
- Correction fp16 → fp32 (transformers 5.1.0 charge en fp16 par defaut)

**Challenger 1 Qwen2.5-3B + LoRA** (entraine le 2026-04-10) :
- F1=0.40, Accuracy=99.74%, Precision=1.00, Recall=0.25
- CodeCarbon: 108.8g CO2eq, duree 38 min
- 3,690,496 / 3,089,633,280 parametres entrainables (0.12%)

**Challenger 2 Llama 3.2 3B + LoRA** (en cours) :
- Licence Meta acceptee sur HuggingFace
- 4,593,664 / 3,217,349,632 parametres entrainables (0.14%)
- Entrainement en cours sur GPU AMD RX 7900 XTX via ROCm

### 2. Corrections et evolutions

- Remplacement initial de Llama par Qwen2.5-3B (licence gated non approuvee)
- Ajout de Llama comme second Challenger apres approbation Meta
- Renommage dossiers modeles : `models/challenger/` → `models/challenger-qwen/` et `models/challenger-llama/`
- Mise a jour de toute la documentation (PLAN_ETAPES, README, etc.)

---

## Sessions precedentes

### Session 2026-03-13
- Entrainement Champion DeBERTa-v3-base reussi
- Developpement API FastAPI (14 endpoints, 27 tests)
- Corrections techniques (passlib/bcrypt, SQLite compat, JSONB/JSON)

### Session 2026-03-10
- Collecte ciblee avec 120 credits API NewsData.io (779 articles)
- Affinement systeme d'annotation (100+ indicateurs ponderes)
- Golden Dataset final : 5808 articles (22 Green IT / 5786 Non Green IT)

---

## Prochaines etapes

### Priorite : Terminer ETAPE 3

#### 3.3 - Finaliser les entrainements
1. Attendre fin entrainement Challenger 2 (Llama 3.2 3B)
2. Benchmark final 3 modeles dans MLflow : F1 vs Latence vs CO2
3. Selectionner le modele vainqueur

#### 3.4 - Packaging
1. Packaging du modele gagnant (safetensors) → `models/production/`
2. Push via DVC
3. Redaction de la Model Card

### Ensuite : Finaliser ETAPE 4
1. Integration effective du modele IA vainqueur dans /analyze
2. Documentation OpenAPI complete
3. Tests complementaires si necessaire

---

## Points importants

### Architecture 3 modeles
Le projet compare 3 approches de classification :
- **Encoder (DeBERTa)** : modele leger, rapide, full fine-tuning
- **Causal LM + LoRA (Qwen)** : gros modele generatif, fine-tuning efficient
- **Causal LM + LoRA (Llama)** : idem, modele Meta gated

Le vainqueur est selectionne par F1 score (metrique primaire), avec latence et CO2 comme criteres secondaires.

### Desequilibre du dataset
Le ratio 22/5786 (0.4% Green IT) est tres desequilibre. Strategie appliquee :
- Oversampling de la classe minoritaire (22 → 1152, ratio cible 20%)
- Applique sur le train set uniquement (test set intact)

### Nomenclature
Le projet utilise des noms de colonnes en **francais** dans toute la stack (SQL, SQLAlchemy, Pydantic).

---

**Redige par KaRn1zC - 2026-04-10**
