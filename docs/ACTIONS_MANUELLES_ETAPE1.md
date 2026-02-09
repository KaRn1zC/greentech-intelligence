# Actions Manuelles Restantes - ÉTAPE 1

> **Dernière mise à jour** : 2026-02-09
> **Statut ÉTAPE 1** : 100% complété ✅

---

## 🎯 Actions à Réaliser

### 1. ✅ Créer GitHub Projects (Kanban) - COMPLÉTÉ

**Ce que tu as obtenu** :

Le template "Kanban" de GitHub t'a créé un projet complet avec :

#### Onglets disponibles :
- **📋 Backlog** : Vue principale de gestion des tâches
- **🎯 Priority board** : Vue par priorité
- **👥 Team items** : Vue par équipe/assignation
- **🗓️ Roadmap** : Vue chronologique/planning
- **📝 My items** : Tes tâches personnelles

#### Colonnes dans l'onglet "Backlog" (workflow Kanban) :
- **Backlog (0/5)** : Tâches futures non priorisées (limite 5)
- **Ready (0)** : Tâches prêtes à être prises
- **In progress (0/3)** : Tâches en cours (limite 3 simultanées)
- **In review (0/5)** : Tâches en revue/validation (limite 5)
- **Done (0)** : Tâches terminées

#### Comment l'utiliser :

1. **Créer des tâches** : Clique sur **"+ Add item"** dans la colonne "Backlog"
2. **Déplacer les tâches** : Glisse-dépose entre colonnes selon l'avancement
3. **Limites WIP** : Les nombres (0/3, 0/5) sont des limites de travail en cours (Work In Progress) - bonnes pratiques Agile
4. **Vues multiples** : Utilise les onglets selon ton besoin (Backlog pour le quotidien, Roadmap pour la vision globale)

#### Première action recommandée :
Crée une première tâche pour tester :
- Colonne : **Backlog**
- Titre : `ÉTAPE 2 : Créer la structure du projet Python`

✅ **Ton Kanban est opérationnel et conforme aux pratiques Agile !**

**Note** : Ce setup est **parfait** pour la validation du Bloc E4 - C16 (Coordination Agile)

---

### 2. ✅ Configurer les Outils de Veille et Méthodologie - EN COURS

Ces outils sont nécessaires pour les blocs E2 (Veille) et E4 (Conception).

#### 2.1 ✅ Penpot (Wireframing) - TERMINÉ
- ✅ Compte créé
- ✅ Projet "GreenTech Intelligence Wireframes" créé
- URL : https://penpot.app/

#### 2.2 ✅ Looping (Modélisation MCD/MLD) - TERMINÉ
- ✅ Installé sur Windows
- À utiliser pour l'ÉTAPE 2.1 (Modélisation des données)
- URL : https://www.looping-mcd.fr/

#### 2.3 ✅ Inoreader (Agrégateur RSS) - TERMINÉ

**Statut** : ✅ Compte créé | ✅ Configuration des flux RSS terminée

**Limitations compte gratuit** ([source](https://www.inoreader.com/pricing)) :
- ✅ 150 abonnements maximum
- ✅ Recherche disponible
- ✅ Organisation en dossiers et tags
- ❌ 1 seul filtre/règle automatique (Pro requis pour plus)
- ❌ Pas de monitoring feeds (Pro uniquement)

**Recommandation** : Le compte gratuit est **suffisant** pour la veille technique du projet (on va configurer ~10-12 flux).

---

##### Guide détaillé de configuration des flux RSS

**Objectif** : Configurer des sources fiables pour la veille Green IT, Sustainable AI et Model Efficiency.

**IMPORTANT** : Différence entre les onglets dans "Add new" ([source](https://www.inoreader.com/blog/2020/10/first-steps-with-inoreader-adding-feeds.html)) :
- **"Website"** : Colle directement l'URL du flux RSS (format `/feed/` ou `/rss`) - **À UTILISER EN PRIORITÉ**
- **"Web Feed"** : Crée un flux RSS artificiel depuis une page web qui n'a pas de RSS natif (fonctionnalité avancée)

---

##### Étape 1 : Comprendre le système d'ajout

1. **Connecte-toi** sur https://www.inoreader.com/
2. Dans la barre latérale gauche, clique sur **"+ Add new"**
3. **Sélectionne l'onglet "Website"** (c'est l'onglet par défaut)
4. Tu verras un champ de recherche avec le texte "Enter feed URL or search by name"

---

##### Étape 2 : Créer l'organisation en dossiers

**AVANT d'ajouter des flux**, crée la structure de dossiers :

1. Dans la barre latérale gauche, clique-droit sur **"Subscriptions"** ou un espace vide
2. Sélectionne **"New folder"** ou **"Add folder"**
3. Crée ces 3 dossiers :
   - 📁 **Green IT**
   - 📁 **Sustainable AI**
   - 📁 **Cloud & Efficiency**

---

##### Étape 3 : Ajouter les flux RSS - Thématique 1 (Green IT)

**MÉTHODE** : Dans l'onglet "Website", colle l'URL complète du flux RSS, puis appuie sur **Entrée**.

**Flux 1 : GreenIT.fr** (Français - Référence française)
1. Clique sur **"+ Add new"** → Onglet **"Website"**
2. Colle : `https://www.greenit.fr/feed/`
3. Appuie sur **Entrée**
4. Une fenêtre apparaît avec le flux détecté
5. Dans le menu déroulant **"Add to folder"**, sélectionne **"Green IT"**
6. Clique sur **"Subscribe"**

**Flux 2 : EcoInfo CNRS** (Français - Académique)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://ecoinfo.cnrs.fr/feed/`
3. **Entrée** → Sélectionne dossier **"Green IT"** → **Subscribe**

**Flux 3 : The Green Web Foundation** (Anglais)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://www.thegreenwebfoundation.org/news/feed/`
3. **Entrée** → Sélectionne dossier **"Green IT"** → **Subscribe**

**Flux 4 : Numérique Responsable (Français)**
1. **"+ Add new"** → **"Website"**
2. Recherche par nom : tape `numérique responsable` (sans URL)
3. Appuie sur **Entrée**
4. Inoreader va chercher des flux correspondants
5. Si un flux pertinent apparaît, sélectionne-le et ajoute au dossier **"Green IT"**
6. **Si aucun résultat** : Passe cette source (compte gratuit limite les recherches)

---

##### Étape 4 : Ajouter les flux RSS - Thématique 2 (Sustainable AI)

**Flux 5 : Hugging Face Blog** (Anglais - ML)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://huggingface.co/blog/feed.xml`
3. **Entrée** → Sélectionne dossier **"Sustainable AI"** → **Subscribe**

**Flux 6 : Google AI Blog** (Anglais)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://blog.research.google/feeds/posts/default`
3. **Entrée** → Sélectionne dossier **"Sustainable AI"** → **Subscribe**

**Flux 7 : Towards Data Science** (Anglais - Medium)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://towardsdatascience.com/feed`
3. **Entrée** → Sélectionne dossier **"Sustainable AI"** → **Subscribe**

---

##### Étape 5 : Ajouter les flux RSS - Thématique 3 (Cloud & Efficiency)

**Flux 8 : PyTorch Blog** (Anglais)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://pytorch.org/blog/feed.xml`
3. **Entrée** → Sélectionne dossier **"Cloud & Efficiency"** → **Subscribe**

**Flux 9 : TensorFlow Blog** (Anglais)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://blog.tensorflow.org/feeds/posts/default`
3. **Entrée** → Sélectionne dossier **"Cloud & Efficiency"** → **Subscribe**

**Flux 10 : The New Stack** (Anglais - DevOps/Cloud)
1. **"+ Add new"** → **"Website"**
2. Colle : `https://thenewstack.io/feed/`
3. **Entrée** → Sélectionne dossier **"Cloud & Efficiency"** → **Subscribe**

---

##### Étape 6 : Vérifier l'ajout et organiser

1. **Vérifier les flux** :
   - Dans la barre latérale gauche, clique sur un dossier (ex: "Green IT")
   - Tu devrais voir les flux ajoutés avec des articles
   - Si un flux affiche "0 article" ou "0 follow" : c'est peut-être un flux inactif, tu peux le supprimer

2. **Marquer les articles non pertinents** :
   - Clique-droit sur un dossier
   - Sélectionne **"Mark all as read"** pour marquer tous les anciens articles comme lus
   - Ça te permet de partir sur une base propre

3. **Utiliser les étoiles** :
   - Quand tu lis un article intéressant, clique sur l'**étoile** ⭐ pour le sauvegarder
   - Tous tes articles étoilés seront accessibles dans **"Starred items"** (barre latérale)

---

##### Étape 7 : Limitations compte gratuit et alternatives

**⚠️ Limitations rencontrées** :
- **1 seul filtre/règle** : Tu ne peux créer qu'UNE règle automatique avec le compte gratuit
- **Certains flux peuvent être vides** : Si un flux affiche "0 article", c'est soit qu'il est inactif, soit qu'Inoreader ne peut pas y accéder

**💡 Alternative gratuite si besoin** :
Si Inoreader gratuit est trop limité, tu peux utiliser :
- **Feedly** (gratuit : 100 sources) : https://feedly.com/
- **The Old Reader** (gratuit : illimité) : https://theoldreader.com/

**Mais pour le projet** : Les 10 flux configurés dans Inoreader sont **suffisants** pour valider la compétence C6 (Veille technique)

---

##### ✅ Résumé de la Structure Finale

```
📁 Green IT (3-4 flux)
   ├─ GreenIT.fr ✅ (https://www.greenit.fr/feed/)
   ├─ EcoInfo CNRS ✅ (https://ecoinfo.cnrs.fr/feed/)
   ├─ The Green Web Foundation ✅ (https://www.thegreenwebfoundation.org/news/feed/)
   └─ [Optionnel: Numérique Responsable si trouvé]

📁 Sustainable AI (3 flux)
   ├─ Hugging Face Blog ✅ (https://huggingface.co/blog/feed.xml)
   ├─ Google AI Blog ✅ (https://blog.research.google/feeds/posts/default)
   └─ Towards Data Science ✅ (https://towardsdatascience.com/feed)

📁 Cloud & Efficiency (3 flux)
   ├─ PyTorch Blog ✅ (https://pytorch.org/blog/feed.xml)
   ├─ TensorFlow Blog ✅ (https://blog.tensorflow.org/feeds/posts/default)
   └─ The New Stack ✅ (https://thenewstack.io/feed/)
```

**Total : 9-10 flux RSS organisés en 3 thématiques**

**C'est suffisant pour** :
- ✅ Valider la compétence **C6** (Veille technique et réglementaire)
- ✅ Rédiger les synthèses mensuelles requises pour l'ÉTAPE 3

---

##### Utilisation Quotidienne Recommandée

- **Matin** (10 min) : Parcourir les nouveaux articles marqués "Important"
- **Hebdomadaire** (30 min) : Lire en détail 3-5 articles et prendre des notes
- **Mensuel** (1-2h) : Rédiger une synthèse dans `docs/veille/synthese_YYYY-MM.md`

---

#### 2.4 ✅ Perplexity Pro (Recherche IA) - CONFIGURÉ
- ✅ Compte créé
- URL : https://www.perplexity.ai/
- **Utilisation** : Deep research hebdomadaire (voir section ci-dessous)

---

##### 🔄 Routine de Veille Hebdomadaire avec Perplexity

**Planning** : Tous les lundis à 11:00 AM

**Objectif** : Effectuer une veille approfondie sur les avancées de la semaine précédente (lundi au dimanche) couvrant les thématiques du projet GreenTech Intelligence.

**⚠️ IMPORTANT** : Deux approches possibles (voir détails dans `docs/veille/PROMPT_PERPLEXITY.md`)

### Option 1 : Tâche Planifiée Perplexity (Automatique)

**Avantages** : Automatique
**Inconvénients** : Limite de ~2000 caractères, qualité réduite

**Prompt court pour tâche planifiée** (~1150 caractères) :

```
Effectue une veille technique hebdomadaire pour le projet GreenTech Intelligence (plateforme de classification automatique d'articles selon leur pertinence Green IT).

Stack : Python 3.12, FastAPI, PostgreSQL, PyTorch (AMD ROCm), Transformers (Hugging Face), Spark, MLflow, DVC, CodeCarbon, Docker, Prometheus, Grafana.

Thématiques à analyser (semaine du [DATE] au [DATE]) :

1. Green IT : réglementations environnementales, métriques carbone IT, éco-conception, hébergement vert
2. Sustainable AI : optimisation modèles (quantization, pruning), NLP efficient, benchmarks énergétiques, fine-tuning (LoRA, QLoRA)
3. Cloud & Infrastructure : optimisations PyTorch, support AMD ROCm, Spark, DevOps durables
4. APIs IA : services Hugging Face, alternatives éco-responsables, comparaisons énergétiques

Format de sortie :
- 5-7 actualités majeures (titre, source, date, résumé, pertinence projet)
- 2-3 analyses approfondies (contexte technique, implications projet, ressources officielles)
- Opportunités d'implémentation (bibliothèques Python, techniques d'optimisation, métriques)
- Ressources complémentaires (articles, repos GitHub, documentations)

Prioriser sources fiables (papers, blogs officiels, conférences), données quantitatives, compatibilité Windows + AMD ROCm.
```

**Configuration tâche planifiée Perplexity** :
1. "Create new scheduled task"
2. Fréquence : Hebdomadaire, Lundi, 11:00 AM
3. Modèle : Deep Research
4. Coller le prompt ci-dessus (remplacer `[DATE]` par `semaine dernière`)

---

### Option 2 : Méthode Manuelle (✅ RECOMMANDÉE)

**Avantages** : Qualité supérieure, flexibilité, prompt complet illimité
**Inconvénients** : 5 minutes de setup manuel chaque lundi

**Prompt complet** : Voir fichier `docs/veille/PROMPT_PERPLEXITY.md` (Option 2)

**Workflow** :
1. Ouvrir `docs/veille/PROMPT_PERPLEXITY.md`
2. Copier le prompt complet (Option 2)
3. Remplacer les dates
4. Nouvelle conversation Perplexity + mode Deep Research
5. Coller et lancer
6. Copier résultat dans `docs/veille/YYYY-MM-DD_synthese.md`

**📚 Documentation Complète** :

- **Prompts complets** : `docs/veille/PROMPT_PERPLEXITY.md` (versions courte et longue)
- **Template synthèse** : `docs/veille/TEMPLATE_synthese_hebdomadaire.md`
- **Guide d'utilisation** : `docs/veille/README.md`
- **Validation C6** : Détails dans le README du dossier veille

---

#### 2.5 Discord (Communication projet)
- Créer un serveur Discord dédié au projet (optionnel si travail solo)
- Inviter les éventuels collaborateurs ou formateurs

**Priorité** : 🟡 **MOYENNE** - Requis pour ÉTAPE 3 (Veille technique - C6)

---

### 3. ✅ Créer un compte Render (Hébergement) - TERMINÉ

**Actions complétées** :

- ✅ Compte créé sur https://render.com/
- ✅ Lié à GitHub

**Note** : Sera utilisé lors de l'ÉTAPE 6 (Déploiement)

---

## ✅ Ce qui a été complété

### Installation Système & GPU
- ✅ Windows 11 Pro
- ✅ PowerShell
- ✅ ROCm/HIP SDK 7.1 installé
- ✅ PyTorch 2.9.1 avec ROCm 7.2 configuré
- ✅ **GPU AMD Radeon RX 7900 XTX détecté et fonctionnel**

### Extensions VSCode
- ✅ Python (Microsoft)
- ✅ Ruff (Linting & Formattage)
- ✅ Docker (Gestion des conteneurs)
- ✅ MyST-Parser (Prévisualisation documentation Sphinx)
- ✅ Playwright Test for VSCode

### Dépendances Python (344 packages installés)
- ✅ httpx, scrapy, playwright, scrapy-playwright
- ✅ pyspark
- ✅ sqlalchemy, asyncpg
- ✅ **torch 2.9.1+rocmsdk20260116, torchvision 0.24.1, torchaudio 2.9.1 (ROCm)**
- ✅ rocm-sdk-core, rocm-sdk-devel, rocm-sdk-libraries-custom, rocm 7.2.0
- ✅ scikit-learn
- ✅ transformers, huggingface-hub
- ✅ peft, accelerate
- ✅ deepchecks
- ✅ mlflow, dvc, dvc-s3, codecarbon
- ✅ fastapi, uvicorn, fastapi-users
- ✅ loguru, pydantic, pydantic-settings, prometheus-client
- ✅ sphinx, myst-parser, furo
- ✅ ruff, pytest, pytest-asyncio, pytest-cov

### Infrastructure Docker
- ✅ PostgreSQL 15 (Port 5432)
- ✅ MinIO (Port 9000 API, 9001 Console)
  - ✅ Buckets créés : `raw-data`, `clean-data`, `models`, `mlflow`
- ✅ MLflow (Port 5000)
- ✅ Prometheus (Port 9090)
- ✅ Grafana (Port 3000) - Credentials : `admin/admin123`
- ✅ Loki (Port 3100)

### Configuration
- ✅ Git + GitHub configuré
- ✅ pyproject.toml complet
- ✅ docker-compose.yml fonctionnel
- ✅ Fichier de provisionnement Grafana créé
- ✅ Datasources Prometheus et Loki préconfigurés dans Grafana

---

## 🔗 Accès aux Services Locaux

| Service | URL | Credentials |
|---------|-----|-------------|
| **MinIO Console** | http://localhost:9001 | `minioadmin` / `minioadmin123` |
| **PostgreSQL** | `localhost:5432` | `greentech` / `greentech_dev_password` |
| **MLflow** | http://localhost:5000 | Aucun |
| **Prometheus** | http://localhost:9090 | Aucun |
| **Grafana** | http://localhost:3000 | `admin` / `admin123` |
| **Loki** | http://localhost:3100 | Aucun (API) |

---

## 🚀 Prochaine Étape

Une fois les actions manuelles complétées (GitHub Projects recommandé, Outils de veille pour ÉTAPE 2), vous pourrez passer à :

**ÉTAPE 2 : Data Factory & Gestion de Données (Bloc E1)**
- Création de la structure du projet Python (`src/greentech/`)
- Modélisation des données (MCD/MLD avec Looping)
- Développement des collecteurs de données
- Pipeline de nettoyage avec Spark
- API de mise à disposition

Référez-vous à `docs/PLAN_ETAPES.md` pour le détail complet.

---

**Document rédigé par KaRn1zC - 2026-02-09**
