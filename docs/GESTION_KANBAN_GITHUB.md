# Gestion du Kanban GitHub - GreenTech Intelligence

> **Document de référence pour la gestion des issues GitHub Projects**
> **Mise à jour** : 2026-02-09

---

## 📋 Vue d'ensemble

Ce document contient :
1. Le détail complet des 6 issues principales du projet
2. Les instructions pour mettre à jour le Kanban au fur et à mesure de l'avancement
3. Les règles de workflow à respecter

---

## 🏷️ Labels Créés

| Label | Couleur | Description |
|-------|---------|-------------|
| `étape-1` | `#0E8A16` (vert) | Étape 1 - Installation & Configuration |
| `étape-2` | `#1D76DB` (bleu) | Étape 2 - Data Factory |
| `étape-3` | `#5319E7` (violet) | Étape 3 - Intelligence Artificielle |
| `étape-4` | `#D93F0B` (orange) | Étape 4 - Backend & API |
| `étape-5` | `#FBCA04` (jaune) | Étape 5 - Frontend |
| `étape-6` | `#D876E3` (rose) | Étape 6 - DevOps & Déploiement |
| `bloc-e1` | `#C2E0C6` (vert clair) | Bloc E1 - Données |
| `bloc-e2` | `#BFDADC` (bleu clair) | Bloc E2 - Veille & SaaS |
| `bloc-e3` | `#C5DEF5` (bleu clair) | Bloc E3 - MLOps |
| `bloc-e4` | `#F9D0C4` (orange clair) | Bloc E4 - Application |
| `bloc-e5` | `#FEF2C0` (jaune clair) | Bloc E5 - Maintenance |
| `terminé` | `#0E8A16` (vert foncé) | Tâche terminée |
| `en-cours` | `#FBCA04` (jaune) | Tâche en cours |

---

## 📝 ISSUE #1 : ÉTAPE 1 - Installation & Configuration

### Informations Issue
- **Title** : `✅ ÉTAPE 1 : Installation & Configuration`
- **Labels** : `étape-1`, `bloc-e1`, `bloc-e2`, `bloc-e3`, `bloc-e4`, `bloc-e5`, `terminé`
- **Statut Initial** : Done ✅
- **Date Création** : 2026-02-09
- **Date Complétion** : 2026-02-09

### Description Complète
```markdown
## 🎯 Objectif
Installer et configurer l'environnement de développement complet pour le projet GreenTech Intelligence.

## 📦 Composants Installés

### Système & Matériel
- ✅ Windows 11 Pro
- ✅ PowerShell (version récente)
- ✅ ROCm/HIP SDK 7.1
- ✅ Drivers AMD Adrenalin Edition
- ✅ GPU AMD Radeon RX 7900 XTX opérationnel

### Outils de Développement
- ✅ VSCode + Extensions (Python, Ruff, Docker, MyST-Parser, Playwright)
- ✅ Git + GitHub
- ✅ GitHub Projects Kanban

### Environnement Python
- ✅ Python 3.12.10
- ✅ UV (gestionnaire de paquets)
- ✅ 344 packages installés

### Stack Python
- ✅ Data : httpx, scrapy, playwright, pyspark, sqlalchemy, asyncpg
- ✅ IA : torch 2.9.1+rocm, torchvision, torchaudio, scikit-learn, transformers, huggingface-hub, peft, accelerate
- ✅ MLOps : mlflow, dvc, dvc-s3, deepchecks, codecarbon
- ✅ Backend : fastapi, uvicorn, fastapi-users, loguru, pydantic, prometheus-client
- ✅ Documentation : sphinx, myst-parser, furo
- ✅ Dev Tools : ruff, pytest, pytest-asyncio, pytest-cov

### Infrastructure Docker
- ✅ PostgreSQL 15 (Port 5432)
- ✅ MinIO avec 4 buckets (raw-data, clean-data, models, mlflow)
- ✅ MLflow (Port 5000)
- ✅ Prometheus (Port 9090)
- ✅ Grafana (Port 3000)
- ✅ Loki (Port 3100)

## 📋 Actions Manuelles Restantes
- Outils de veille : Penpot, Looping, Inoreader, Perplexity Pro, Discord
- Compte Render (pour ÉTAPE 6)

## 📊 Statut
✅ **TERMINÉ** - 2026-02-09

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 1
- Voir : `docs/ACTIONS_MANUELLES_ETAPE1.md`
```

### Quand Mettre à Jour
**Issue déjà complète** - Aucune mise à jour nécessaire sauf si des outils supplémentaires sont installés plus tard.

---

## 📝 ISSUE #2 : ÉTAPE 2 - Data Factory & Gestion de Données

### Informations Issue
- **Title** : `ÉTAPE 2 : Data Factory & Gestion de Données (Bloc E1)`
- **Labels** : `étape-2`, `bloc-e1`, `en-cours`
- **Statut Initial** : Ready → In Progress
- **Date Création** : 2026-02-09
- **Date Début** : À venir

### Description Complète
```markdown
## 🎯 Objectif
Développer la pipeline complète de collecte, nettoyage et mise à disposition des données.

## 📋 Sous-tâches

### 2.1 Conception & Conformité
- [ ] Rédiger les spécifications techniques de collecte
- [ ] Réaliser le MCD/MLD avec Looping (Modèle Merise)
- [ ] Générer le script SQL depuis le MLD
- [ ] Rédiger le registre RGPD
- [ ] Définir les procédures d'anonymisation

### 2.2 Infrastructure de Stockage
- [ ] Créer la base `greentech_db` dans PostgreSQL
- [ ] Créer l'utilisateur dédié avec droits restreints
- [ ] Vérifier les buckets MinIO (raw-data, clean-data)
- [ ] Documenter l'installation

### 2.3 Module 0 : Configuration Dynamique (SQL)
- [ ] Créer la table `search_config`
- [ ] Script d'initialisation SQL
- [ ] Fonction Python `get_config_from_db()`
- [ ] Tests de connexion SQLAlchemy async

### 2.4 Module 1 : Collecte API
- [ ] Script `api_collector.py` avec httpx
- [ ] Configuration requêtes HTTP (headers, timeouts, retry)
- [ ] Parsing JSON
- [ ] Sauvegarde brute vers MinIO raw-data

### 2.5 Module 2 : Scraping Hybride
- [ ] Spider Scrapy avec Playwright
- [ ] Gestion JS rendering
- [ ] Respect robots.txt + rate limiting
- [ ] Sauvegarde HTML vers MinIO raw-data

### 2.6 Module 3 : Ingestion Fichiers
- [ ] Script de lecture CSV/JSON
- [ ] Upload vers MinIO raw-data

### 2.7 Traitement Big Data (Spark)
- [ ] Configuration session PySpark + connecteur S3/MinIO
- [ ] Script de nettoyage (suppression HTML, formats, doublons)
- [ ] Agrégation des 3 sources
- [ ] Sauvegarde vers MinIO clean-data (Parquet)

### 2.8 Script d'Import SQL
- [ ] Script SQLAlchemy async d'import vers PostgreSQL
- [ ] Gestion des conflits (Upsert)
- [ ] Requêtes de vérification
- [ ] Documentation complète

### 2.9 API de Mise à Disposition
- [ ] Spécifications OpenAPI
- [ ] Endpoints CRUD articles
- [ ] Validation requêtes (Pydantic)
- [ ] Sécurité OWASP Top 10
- [ ] Documentation Swagger/ReDoc

## 📚 Compétences Validées
- **C1** : Automatiser l'extraction de données
- **C2** : Développer des requêtes SQL d'extraction
- **C3** : Agrégation et nettoyage des données
- **C4** : Création de la base de données (et RGPD)
- **C5** : Développer une API de mise à disposition (REST)

## 📊 Statut
🚧 **À DÉMARRER**

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 2
- Structure : `src/greentech/data/`
```

### Quand Mettre à Jour

#### Au démarrage de l'ÉTAPE 2
1. **Déplacer l'issue de "Ready" vers "In Progress"**
2. **Modifier la description** : Changer `🚧 **À DÉMARRER**` par `🚧 **EN COURS** - Démarré le YYYY-MM-DD`
3. **Ajouter le label** : `en-cours`

#### Pendant le développement
**À chaque sous-tâche complétée**, éditer l'issue et cocher la case correspondante :
- Exemple : `- [x] Rédiger les spécifications techniques de collecte`

**Fréquence recommandée** : Mettre à jour quotidiennement ou à chaque fin de session de développement.

#### À la fin de l'ÉTAPE 2
1. **Vérifier que toutes les cases sont cochées**
2. **Modifier la description** : Changer le statut par `✅ **TERMINÉ** - Date de fin`
3. **Retirer le label** : `en-cours`
4. **Ajouter le label** : `terminé`
5. **Déplacer l'issue vers "Done"**
6. **Fermer l'issue** (bouton "Close issue")
7. **Cocher dans `docs/PLAN_ETAPES.md`** : Toutes les cases de l'ÉTAPE 2
8. **Cocher dans `docs/CHECKLIST_SUIVI.md`** : Les compétences C1, C2, C3, C4, C5

---

## 📝 ISSUE #3 : ÉTAPE 3 - Intelligence Artificielle

### Informations Issue
- **Title** : `ÉTAPE 3 : Intelligence Artificielle (Blocs E2 & E3)`
- **Labels** : `étape-3`, `bloc-e2`, `bloc-e3`
- **Statut Initial** : Backlog
- **Date Création** : 2026-02-09

### Description Complète
```markdown
## 🎯 Objectif
Intégrer un service IA SaaS pour le résumé automatique et développer un modèle custom de classification Green IT avec MLOps.

## 📋 Sous-tâches

### 3.1 Veille Technologique & Benchmark (Bloc E2 - C6, C7, C8)
- [ ] Configuration Inoreader (flux RSS Green IT, Sustainable AI)
- [ ] Configuration Perplexity Pro (synthèses hebdo)
- [ ] Rédaction synthèse mensuelle (Markdown)
- [ ] Benchmark services IA de résumé (OpenAI, Mistral, Hugging Face)
- [ ] Document de benchmark (`docs/specs/benchmark_ia.md`)
- [ ] Module `summarizer.py` (Hugging Face Serverless API)

### 3.2 Préparation Données & MLOps (Bloc E3 - Data Ops)
- [ ] Création Golden Dataset (200 articles annotés)
- [ ] Labeling manuel : Green IT (1) vs Non Green IT (0)
- [ ] Initialisation DVC
- [ ] Configuration remote DVC vers MinIO
- [ ] Versioning dataset (`dataset.csv.dvc`)
- [ ] Push données vers stockage distant

### 3.3 Entraînement Modèles (PC Fixe - ROCm GPU)
- [ ] Lancement serveur MLflow local
- [ ] Intégration CodeCarbon (mesure CO2)
- [ ] Fine-tuning DeBERTa-v3-base — Champion (full fine-tuning)
- [ ] Fine-tuning Qwen2.5-3B avec LoRA — Challenger 1
- [ ] Fine-tuning Llama 3.2 3B avec LoRA — Challenger 2
- [ ] Benchmark final 3 modèles (F1 vs Latence vs CO2)
- [ ] Sélection du modèle vainqueur

### 3.4 Validation & Packaging (Bloc E3 - C12)
- [ ] Suite de tests Deepchecks (Data Leakage, Biais, Robustesse)
- [ ] Génération rapport de validation
- [ ] Conversion modèle optimisé (safetensors/ONNX)
- [ ] Push modèle via DVC
- [ ] Rédaction Model Card

### 3.5 Déploiement MLOps (Bloc E3 - C11, C13)
- [ ] Définition métriques production (drift, latence, % Green)
- [ ] Configuration Prometheus pour métriques modèle
- [ ] Préparation dashboard Grafana
- [ ] Configuration alertes (latence > 2s)

## 📚 Compétences Validées
- **C6** : Veille technique et réglementaire
- **C7** : Identifier des services IA (Benchmark)
- **C8** : Paramétrer un service IA
- **C11** : Monitorer le modèle IA
- **C12** : Tests automatisés du modèle
- **C13** : Chaîne de livraison continue (CI/CD pour IA)

## 📊 Statut
⏳ **EN ATTENTE** (après ÉTAPE 2)

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 3
- Structure : `src/greentech/ai/`
```

### Quand Mettre à Jour

#### Quand l'ÉTAPE 2 est terminée
1. **Déplacer l'issue de "Backlog" vers "Ready"**
2. **Modifier la description** : Changer `⏳ **EN ATTENTE**` par `🚧 **PRÊTE À DÉMARRER**`

#### Au démarrage de l'ÉTAPE 3
1. **Déplacer de "Ready" vers "In Progress"**
2. **Modifier la description** : Changer le statut par `🚧 **EN COURS** - Démarré le YYYY-MM-DD`
3. **Ajouter le label** : `en-cours`

#### Pendant et à la fin
Même processus que l'ÉTAPE 2 (cocher cases, mettre à jour statut, déplacer vers Done, fermer, cocher PLAN_ETAPES et CHECKLIST).

---

## 📝 ISSUE #4 : ÉTAPE 4 - Backend & API

### Informations Issue
- **Title** : `ÉTAPE 4 : Backend & API (Blocs E1 & E4)`
- **Labels** : `étape-4`, `bloc-e1`, `bloc-e3`, `bloc-e4`
- **Statut Initial** : Backlog
- **Date Création** : 2026-02-09

### Description Complète
```markdown
## 🎯 Objectif
Construire l'API REST sécurisée FastAPI qui expose les données et les fonctionnalités d'IA.

## 📋 Sous-tâches

### 4.1 Conception Architecture API (Bloc E4 - C14, C15)
- [ ] Définition des endpoints OpenAPI
- [ ] Spécifications sécurité (OAuth2 + JWT)
- [ ] Validation des entrées (Pydantic)
- [ ] Diagramme de flux données

### 4.2 Développement Serveur API (Bloc E1 - C5 & Bloc E4 - C17)
- [ ] Initialisation app FastAPI (titre, version, CORS)
- [ ] Connexion PostgreSQL (SQLAlchemy async)
- [ ] Modèles Pydantic (schemas)
- [ ] Configuration Loguru (logs structurés)

### 4.3 Implémentation Sécurité (Bloc E4 - C17)
- [ ] Configuration FastAPI Users (Bearer Transport)
- [ ] Création/Login utilisateurs
- [ ] Protection routes sensibles
- [ ] Implémentation OWASP Top 10

### 4.4 Exposition des Données (Bloc E1 - C5)
- [ ] GET /articles (pagination, filtres)
- [ ] GET /articles/{id}
- [ ] GET /stats
- [ ] Tests requêtes SQL

### 4.5 Intégration IA dans API (Bloc E3 - C9, C10)
- [ ] POST /analyze (URL ou Texte)
- [ ] Orchestration : Scraping → Nettoyage → Modèle IA → Résumé SaaS
- [ ] Chargement modèle au démarrage
- [ ] Tests d'intégration

### 4.6 Documentation & Tests (Bloc E4 - C17, C18)
- [ ] Documentation Swagger UI (/docs)
- [ ] Documentation ReDoc (/redoc)
- [ ] Tests unitaires (pytest)
- [ ] Tests d'intégration endpoints
- [ ] Vérification codes erreur (401, 422, 500)

## 📚 Compétences Validées
- **C5** : Développer une API de mise à disposition (REST)
- **C9** : Développer une API exposant un modèle IA
- **C10** : Intégrer l'API IA dans une application
- **C14** : Analyser le besoin
- **C15** : Concevoir le cadre technique
- **C17** : Développer composants et interfaces
- **C18** : Automatiser les tests (CI)

## 📊 Statut
⏳ **EN ATTENTE** (après ÉTAPE 3)

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 4
- Structure : `src/greentech/api/`
```

### Quand Mettre à Jour
Même processus que l'ÉTAPE 3 (attendre fin ÉTAPE 3, déplacer vers Ready, puis In Progress, cocher cases, finaliser).

---

## 📝 ISSUE #5 : ÉTAPE 5 - Frontend & Application

### Informations Issue
- **Title** : `ÉTAPE 5 : Frontend & Application (Bloc E4)`
- **Labels** : `étape-5`, `bloc-e3`, `bloc-e4`
- **Statut Initial** : Backlog
- **Date Création** : 2026-02-09

### Description Complète
```markdown
## 🎯 Objectif
Réaliser l'interface utilisateur avec React et Shadcn/UI, avec un focus fort sur l'accessibilité WCAG.

## 📋 Sous-tâches

### 5.1 Initialisation Frontend (Bloc E4 - C14, C15)
- [ ] Génération projet Vite (React + TypeScript)
- [ ] Configuration Tailwind CSS
- [ ] Initialisation Shadcn/UI + lucide-react
- [ ] Wireframes sur Penpot

### 5.2 Composants UI (Bloc E4 - C17)
- [ ] Installation composants Shadcn (button, input, card, badge, table, dialog, form, toast)
- [ ] Layout principal (Header, Footer, Container)
- [ ] Composants responsive

### 5.3 Pages & Parcours Utilisateur (Bloc E4 - C14, C17)
- [ ] Page Login (formulaire + stockage JWT)
- [ ] Page Dashboard (zone d'analyse + derniers articles + graphiques)
- [ ] Page Détail Article (résultat IA + résumé + score confiance)
- [ ] Navigation au clavier

### 5.4 Intégration Logique Métier (Bloc E3 - C10 & Bloc E4 - C17)
- [ ] Client HTTP (fetch/axios + token auto)
- [ ] Gestion état (useState/useEffect ou React Query)
- [ ] Gestion chargement (Skeletons/Spinners)
- [ ] Gestion erreurs (Toasts)

### 5.5 Tests & Accessibilité (Bloc E4 - C14, C18)
- [ ] Installation @axe-core/playwright
- [ ] Script de test Playwright + Axe
- [ ] Scan accessibilité (contraste, labels, structure HTML)
- [ ] Génération rapport conformité WCAG
- [ ] Tests responsive (Mobile, Tablette, Desktop)

## 📚 Compétences Validées
- **C10** : Intégrer l'API IA dans une application
- **C14** : Analyser le besoin (User Stories + Wireframes + Objectifs accessibilité WCAG)
- **C15** : Concevoir le cadre technique (POC)
- **C17** : Développer composants et interfaces (accessibilité, éco-conception, sécurité)
- **C18** : Automatiser les tests (CI)

## 📊 Statut
⏳ **EN ATTENTE** (après ÉTAPE 4)

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 5
- Structure : `frontend/`
```

### Quand Mettre à Jour
Même processus que les étapes précédentes (attendre fin ÉTAPE 4, déplacer vers Ready, etc.).

---

## 📝 ISSUE #6 : ÉTAPE 6 - DevOps, Déploiement & Maintenance

### Informations Issue
- **Title** : `ÉTAPE 6 : DevOps, Déploiement & Maintenance (Bloc E5)`
- **Labels** : `étape-6`, `bloc-e4`, `bloc-e5`
- **Statut Initial** : Backlog
- **Date Création** : 2026-02-09

### Description Complète
```markdown
## 🎯 Objectif
Industrialiser le projet avec CI/CD automatisé, déploiement sur Render et surveillance proactive.

## 📋 Sous-tâches

### 6.1 Automatisation Tests (CI Pipeline) - Bloc E4 - C18
- [ ] Fichier `.github/workflows/ci.yml`
- [ ] Déclencheurs (push, pull request)
- [ ] Étape Linting (ruff)
- [ ] Étape Tests Backend (pytest)
- [ ] Étape Tests IA (deepchecks)
- [ ] Étape Tests Accessibilité (playwright + axe-core)

### 6.2 Conteneurisation & Packaging - Bloc E4 - C19
- [ ] Dockerfile API (multi-stage build)
- [ ] Dockerfile Frontend (NGINX/serve)
- [ ] Finalisation docker-compose.yml
- [ ] Tests build local

### 6.3 Livraison Continue (CD Pipeline) - Bloc E4 - C19
- [ ] Fichier `.github/workflows/cd.yml`
- [ ] Liaison compte Render au GitHub
- [ ] Configuration Web Service (API Docker)
- [ ] Configuration Static Site (Frontend)
- [ ] Déploiement automatique sur push main

### 6.4 Monitoring & Observabilité - Bloc E5 - C20
- [ ] Configuration finale Prometheus (scraper `/metrics`)
- [ ] Pilote logging Docker vers Loki
- [ ] Dashboard Grafana "Performance Système"
- [ ] Dashboard Grafana "Métier GreenTech"
- [ ] Configuration alertes (latence > 2s)
- [ ] Tests accessibilité dashboards

### 6.5 Maintenance & Gestion Incidents - Bloc E5 - C21
- [ ] Simulation coupure BDD
- [ ] Simulation indisponibilité API Hugging Face
- [ ] Vérification messages erreur utilisateur
- [ ] Playbook de débogage (lecture logs Grafana)
- [ ] Documentation procédure mise à jour modèle IA
- [ ] Tests de récupération

## 📚 Compétences Validées
- **C18** : Automatiser les tests (CI)
- **C19** : Livraison continue (CD)
- **C20** : Surveiller l'application (Monitoring App)
- **C21** : Résoudre les incidents techniques

## 📊 Statut
⏳ **EN ATTENTE** (après ÉTAPE 5)

## 📚 Documentation
- Voir : `docs/PLAN_ETAPES.md` - Section ÉTAPE 6
- Structure : `.github/workflows/`, `Dockerfile.*`, `config/`
```

### Quand Mettre à Jour
Même processus que les étapes précédentes (attendre fin ÉTAPE 5, déplacer vers Ready, etc.).

---

## 🔄 Workflow Général de Mise à Jour

### Règle de Base
**À chaque fin de session de développement**, mets à jour ton Kanban GitHub :

1. **Coche les cases** des sous-tâches terminées dans l'issue en cours
2. **Ajoute un commentaire** sur l'issue si nécessaire (blocage, question, avancement)
3. **Déplace les cartes** selon l'avancement

### Transitions d'Étape

Quand tu termines une ÉTAPE complète :

```
ÉTAPE N terminée
    ↓
1. Cocher toutes les cases de l'issue N
2. Modifier statut → "✅ TERMINÉ - Date"
3. Retirer label "en-cours" + Ajouter label "terminé"
4. Déplacer vers "Done"
5. Fermer l'issue
6. Cocher PLAN_ETAPES.md (ÉTAPE N)
7. Cocher CHECKLIST_SUIVI.md (compétences validées)
    ↓
ÉTAPE N+1 commence
    ↓
1. Déplacer issue N+1 de "Backlog" vers "Ready"
2. Modifier statut → "🚧 PRÊTE À DÉMARRER"
3. Déplacer vers "In Progress" quand on commence
4. Modifier statut → "🚧 EN COURS - Date"
5. Ajouter label "en-cours"
```

---

## 📊 Dashboard Recommandé

Dans l'onglet **"Backlog"** de ton Kanban, tu verras toujours :

- **Done** : Étapes complétées (1 actuellement)
- **In Review** : Étapes en validation (optionnel si travail solo)
- **In Progress** : Étape en cours (1 max recommandé pour focus)
- **Ready** : Prochaine étape (1 actuellement : ÉTAPE 2)
- **Backlog** : Étapes futures (3-6 actuellement)

---

## ⚠️ Rappels Importants

1. **Ne jamais avoir plus de 1 grande étape en "In Progress"** - Focus sur une étape à la fois
2. **Mettre à jour quotidiennement** les cases cochées pour suivre l'avancement réel
3. **Cocher CHECKLIST_SUIVI.md** en même temps que les issues pour la cohérence
4. **Fermer les issues terminées** pour garder un historique propre
5. **Utiliser les labels** pour filtrer facilement par bloc de compétences

---

## 🎯 Prochaine Action

**Maintenant que ton Kanban est configuré**, tu vas :

1. **Ouvrir l'issue #2 (ÉTAPE 2)** dans GitHub
2. **La déplacer de "Ready" vers "In Progress"**
3. **Ajouter le label** : `en-cours`
4. **Commencer le développement** de la structure Python

---

## 🔔 RAPPEL AUTOMATIQUE DE FIN D'ÉTAPE

**⚠️ IMPORTANT** : À la fin de chaque ÉTAPE complète, tu dois SYSTÉMATIQUEMENT :

### Checklist de Clôture d'Étape (9 actions obligatoires)

```
□ 1. Ouvrir ce fichier (docs/GESTION_KANBAN_GITHUB.md)
□ 2. Aller à la section "ISSUE #N" de l'étape terminée
□ 3. Suivre les instructions "Quand Mettre à Jour" > "À la fin de l'ÉTAPE N"
□ 4. Cocher toutes les cases de l'issue GitHub
□ 5. Modifier la description : Statut → "✅ TERMINÉ - Date"
□ 6. Labels : Retirer "en-cours" + Ajouter "terminé"
□ 7. Déplacer l'issue vers "Done" et fermer l'issue
□ 8. Cocher docs/PLAN_ETAPES.md (ÉTAPE N complète)
□ 9. Cocher docs/CHECKLIST_SUIVI.md (compétences C1, C2, etc.)
```

### Aide-mémoire Rapide

**Quand demander la mise à jour ?**
- ✅ Dès que toutes les sous-tâches d'une ÉTAPE sont terminées
- ✅ Avant de passer à l'ÉTAPE suivante
- ✅ À chaque session de développement (pour cocher les sous-tâches en cours)

**Pourquoi c'est important ?**
- Traçabilité du projet pour le diplôme
- Synchronisation entre GitHub Projects et la documentation
- Validation des compétences (C1-C21) au fur et à mesure

---

**Document rédigé par KaRn1zC - 2026-02-09**
**À consulter à chaque changement d'étape !**
