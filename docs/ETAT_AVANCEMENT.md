# Etat d'avancement du projet - GreenTech Intelligence

> **Derniere mise a jour** : 2026-04-14

---

## Ou nous en sommes

### ETAPE 1 : Installation & Configuration - **TERMINEE**
Toutes les dependances et outils sont installes et configures.

### ETAPE 2 : Data Factory & Gestion de Donnees (Bloc E1) - **TERMINEE**
Pipeline complet operationnel : 3 sources de collecte, nettoyage PySpark, ingestion PostgreSQL.

### ETAPE 3 : Intelligence Artificielle (Blocs E2 & E3) - **TERMINEE**

#### Section 3.1 : Veille Technologique & Benchmark - **TERMINEE**
- Veille Inoreader + Perplexity Pro configuree
- Benchmark services IA realise (choix : HuggingFace Serverless API)
- Module summarizer.py developpe et fonctionnel

#### Section 3.2 : Preparation des Donnees & MLOps - **TERMINEE**
- **Golden Dataset** : CREE et annote (5808 articles, 22 Green IT / 5786 Non Green IT)
- **DVC** : Initialise, remote MinIO configure, dataset versionne et pousse (s3://models/dvc)

#### Section 3.3 : Entrainement & Competition des Modeles - **TERMINEE**

Architecture a 4 modeles en competition :

| Modele | Type | Params | Methode | Statut | F1 | Accuracy | CO2 |
|--------|------|--------|---------|--------|-----|----------|-----|
| **Champion** DeBERTa-v3-base | Encoder | 86M | Full fine-tuning | ENTRAINE | 0.44 | 99.6% | 97.8g |
| **Challenger 1** Qwen2.5-3B | Causal LM | 3085M | LoRA (r=16) | ENTRAINE | 0.40 | 99.74% | 108.8g |
| **Challenger 2** Llama 3.2 3B | Causal LM | 3213M | LoRA (r=16) | ENTRAINE | 0.667 | 99.83% | 112.0g |
| **Challenger 3** Qwen3-4B | Causal LM (multilingue) | ~4000M | LoRA (r=16, attention only) | PRET A ENTRAINER | - | - | - |

- Oversampling de la classe minoritaire (22 → 1152, ratio 20%)
- Benchmark final initial execute sur 1162 articles de test
- **Vainqueur historique** : Llama 3.2 3B + LoRA (F1=0.667, precision 1.0, recall 0.50)
- **Nouveau modele cible depuis avril 2026** : Qwen3-4B + LoRA (Apache-2.0,
  multilingue natif, remplace Llama gated). Metriques en cours de mesure.
- **Note** : une tentative intermediaire avec `Qwen/Qwen3.5-4B` (15 avril 2026)
  a ete abandonnee — ce modele est en realite un VLM (image-text-to-text)
  avec attention lineaire hybride non supportee sous ROCm, provoquant des
  freezes systeme a l'entrainement LoRA.

#### Section 3.4 : Validation & Packaging - **TERMINEE**
- Tests Deepchecks ecrits et fonctionnels
- Packaging du modele gagnant (adapter_model.safetensors, 18 Mo) dans models/production/
- Push DVC vers MinIO (s3://models/dvc)
- Model Card redigee (models/production/README.md)

#### Section 3.5 : Deploiement MLOps (Monitoring) - **TERMINE**
- Metriques de production definies (14 metriques Prometheus)
- Configuration Prometheus preparee (scrape API, MLflow, MinIO)
- Stack Prometheus + Loki + Grafana operationnelle via docker-compose
- 2 dashboards Grafana provisionnes (Performance Systeme + Metier GreenTech)
- MLflow Tracking Server Docker (PostgreSQL backend + MinIO S3 artifacts)
- CodeCarbon integre dans le tracking MLflow (mesure CO2 par run)

### ETAPE 4 : Backend & API (Blocs E1 & E4) - **TERMINEE**

- Architecture API FastAPI complete (14 endpoints)
- Schemas Pydantic (article, user, analysis, stats)
- Authentification JWT (bcrypt + python-jose)
- Routes : auth (4), articles (3), stats (3), analyze (2), health (1), metrics (1)
- Suite de tests d'integration : **39 tests**, tous passent
- Middleware logging, CORS configurable via env, gestion d'erreurs globale
- Integration du modele IA vainqueur dans inference.py (dispatch automatique :
  Qwen3-4B + LoRA si adapter_config pointe vers qwen3-4b, Llama 3.2 3B + LoRA
  sinon, DeBERTa si pas d'adapter_config)
- Documentation OpenAPI/Swagger complete sur /docs et /redoc
- Dockerfile.api multi-stage (uv, python:3.12-slim, non-root)
- Competences C5 et C9 validees

### ETAPE 5 : Frontend & Application (Bloc E4) - **TERMINEE**

#### Section 5.1 : Initialisation - **TERMINEE**
- Vite + React 19 + TypeScript 6
- Tailwind CSS v4 via @tailwindcss/vite (plugin natif, pas de tailwind.config.js)
- shadcn/ui initialise (style Zinc, CSS variables oklch)
- lucide-react pour les icones
- react-router-dom v7 pour le routage

#### Section 5.2 : Composants UI - **TERMINEE**
- 11 composants shadcn : button, input, card, badge, skeleton, table, dialog, label, separator, tabs, sonner (toast)
- Layout principal : Header responsive (logo, nav, auth), Footer (mentions), conteneur centre max-w-5xl
- Theming dark mode pret (variables CSS oklch)

#### Section 5.3 : Pages - **TERMINEE**
- **LoginPage** : Formulaire login/register toggle, validation email/password, gestion erreurs, redirection auto
- **DashboardPage** : Zone d'analyse (URL/texte), camembert stats Green IT (recharts), articles recents, polling async
- **ArticleDetailPage** : Affichage complet, badge Green IT colore, resume IA, barre de confiance, metadonnees

#### Section 5.4 : Logique metier - **TERMINEE**
- Client API fetch avec intercepteur auth automatique (lib/api.ts)
- Token JWT localStorage (lib/auth.ts)
- Context React auth (hooks/useAuth.tsx) avec login/logout/getMe
- Routes protegees via ProtectedRoute
- Types TypeScript miroir des schemas Pydantic (types/api.ts)
- Loading states (Skeleton, Loader2 spinner) sur toutes les sections
- Toasts (Sonner) pour succes/erreurs d'analyse et d'authentification

#### Section 5.5 : Tests & Accessibilite - **TERMINEE**
- @axe-core/playwright installe et configure
- 3 tests WCAG (login, dashboard, detail article) — zero violation critique/serious
- Tests navigation clavier (tabulation champs login, soumission Enter)
- Tests responsive automatises (Mobile 375px, Tablette 768px, Desktop 1280px)
- Labels accessibles (Label shadcn, aria-label, aria-describedby, sr-only)
- Rapport HTML genere par Playwright (a11y-report/)

#### Qualite code frontend
- TypeScript : zero erreur (tsc --noEmit clean)
- ESLint : zero erreur (hooks, refresh, ts-eslint)
- Competences C10, C14-C17 validees

---

## Prochaines etapes

### Priorite : ETAPE 6 — DevOps, Deploiement & Maintenance (Bloc E5)
1. Dockerfile.frontend + orchestration docker-compose profile full
2. CI Pipeline : finaliser validation GitHub Actions (lint, tests, build, a11y, Docker)
3. CD Pipeline : liaison Render, deploy automatique sur push main
4. Configuration finale monitoring (scrape API live, alertes Grafana)
5. Simulation d'incidents (chaos engineering leger)
6. Documentation de maintenance (playbook Grafana, procedure MAJ modele IA)
7. Cocher C13, C18-C21 dans CHECKLIST_SUIVI.md

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
Le projet utilise des noms de colonnes en **francais** dans toute la stack (SQL, SQLAlchemy, Pydantic, TypeScript).

---

## Evolutions post-livraison initiale (2026-04-13)

### Classification hybride en deux etages
Le scoring mots-cles historique a ete refactore en **pre-filtre permissif**,
complete par un LLM judge (`Qwen/Qwen2.5-7B-Instruct`) qui tranche les
cas ambigus. Le golden dataset est desormais regenere a partir de la DB
post-LLM, pas du scoring seul.

- **Etage 1** (`scripts/auto_annotate_dataset.py`) : ~85% du corpus classifie
  en `NON_GREEN` direct, ~15% envoye au LLM comme `CANDIDATE`.
- **Etage 2** (`scripts/classify_candidates.py`) : LLM judge avec prompt
  zero-shot permissif et parser JSON tolerant aux erreurs d'echappement.
- **Export** (`scripts/export_golden_dataset.py`) : regeneration du CSV
  depuis l'etat final de la DB.

### Fallback local Qwen sur GPU AMD ROCm
Pour resister a l'epuisement du quota mensuel HF Inference Providers,
un dispatcher (`src/greentech/ai/services/llm_dispatcher.py`) bascule
automatiquement tous les appels LLM (classification + resumes) sur le
meme modele Qwen execute en local sur la RX 7900 XTX 24 Go
(`src/greentech/ai/services/llm_local.py`). Ce bascule est declenche par
la premiere erreur HTTP 402, reset a chaque nouveau processus Python.

### Resumes cibles sur les Green IT confirmes uniquement
`summarize_green_it_articles` (dans `summarizer.py`) ne genere les deux
resumes (general + ecologique) que pour les articles avec
`est_green_it=true` et au moins un resume manquant. Cela evite de saturer
le quota HF sur l'ensemble du corpus alors que les resumes ne sont
exploites que pour les Green IT dans l'interface.

### Pipeline enrichi
`scripts/retrain_pipeline.py` integre desormais les nouvelles etapes
`classify`, `summarize`, `export-golden` entre `annotate` et `train-cv`.
Le pipeline par defaut est :

```
collect → annotate → classify → summarize → export-golden → train-cv → auto-promote
```

Chaque etape ne traite que les articles nouveaux ou incomplets, de sorte
que le pipeline complet peut etre relance a l'identique sans retravailler
les articles deja classifies/resumes.

