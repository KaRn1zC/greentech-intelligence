# Plan Etape par Etape - GreenTech Intelligence

> **Feuille de route centrale du projet.**
> Chaque micro-etape est indispensable a la validation du diplome.

---

## ETAPE 1 : Installation & Configuration

### 1.1 Systeme & Materiel (Le Socle)

- [x] Systeme d'exploitation : Windows 11 Pro (verifier les mises a jour)
- [x] Terminal : PowerShell (version la plus recente disponible)
- [x] Acceleration Graphique (PC Fixe uniquement) :
  - [x] Drivers AMD Software : Adrenalin Edition a jour
  - [x] Suite AMD ROCm 7.2.1 stable pour PyTorch (migration 7.1 MSI + 7.2 wheels pre-release -> 7.2.1 wheels-only le 2026-04-18, voir `docs/PROCEDURE_MAJ_ROCM.md`)

### 1.2 Outils de Developpement Generaux

- [x] Editeur de code : VSCode (Visual Studio Code)
- [x] Extensions VSCode obligatoires :
  - [x] Python (Microsoft)
  - [x] Ruff (Linting & Formattage)
  - [x] Docker (Gestion des conteneurs)
  - [x] MyST-Parser (Previsualisation documentation Sphinx)
  - [x] Playwright Test for VSCode
- [x] Gestionnaire de Version : Git pour Windows (Git SCM)
- [x] Plateforme de Depot : Creation d'un compte et d'un depot "GreenTech-Intelligence" sur GitHub
- [x] Gestion de Projet : Initialisation d'un projet "Kanban" dans l'onglet Projects de GitHub

### 1.3 Environnement Python & Gestion de Paquets

- [x] Gestionnaire de projet : uv (CLI Astral) - installe comme remplacant de pip/poetry
- [x] Configuration Python : Creation d'un environnement virtuel gere par uv

### 1.4 Stack Data (Collecte & Stockage)

Librairies a ajouter au projet Python via uv :

- [x] Requetes HTTP : httpx (Client HTTP moderne et asynchrone)
- [x] Scraping Framework : scrapy (Framework principal)
- [x] Rendu Navigateur (JS) : playwright + scrapy-playwright (Connecteur)
- [x] Big Data & Traitement : pyspark (Interface Python pour Apache Spark)
- [x] Base de Donnees (ORM) :
  - [x] sqlalchemy (ORM version 2.0+)
  - [x] asyncpg (Driver asynchrone pour PostgreSQL)

### 1.5 Stack Intelligence Artificielle (Blocs E2 & E3)

- [x] Frameworks de Calcul (PC Fixe - ROCm) :
  - [x] torch (PyTorch) version 2.9.1+rocm7.2.1 avec ROCm 7.2.1 stable
  - [x] torchvision 0.24.1+rocm7.2.1 & torchaudio 2.9.1+rocm7.2.1 (wheels ROCm stables)
- [x] Machine Learning Classique : scikit-learn
- [x] Modeles & Transformers :
  - [x] transformers (Librairie Hugging Face)
  - [x] huggingface_hub (API Serverless et telechargement de modeles)
- [x] Fine-tuning :
  - [x] peft (LoRA)
  - [x] accelerate
- [x] Tests & Qualite IA : deepchecks (Validation des donnees et des modeles)
- [x] MLOps & Suivi :
  - [x] mlflow (Tracking des experiences)
  - [x] dvc + dvc-s3 (Data Version Control + Plugin stockage objet)
  - [x] codecarbon (Mesure de l'empreinte carbone - Green IT)

### 1.6 Stack Backend & API (Bloc E4)

- [x] Framework API : fastapi
- [x] Serveur : uvicorn[standard]
- [x] Authentification : fastapi-users[sqlalchemy]
- [x] Utilitaires :
  - [x] python-multipart (Gestion upload fichiers)
  - [x] loguru (Logging avance)
  - [x] pydantic + pydantic-settings
  - [x] prometheus-client (Metriques)

### 1.7 Stack Frontend (Application Client)

Environnement Node.js (via npm) :

- [x] Build Tool : Vite (Template React TypeScript)
- [x] Framework UI : React 19
- [x] Composants & Design :
  - [x] tailwindcss v4 (CSS Utility-first via @tailwindcss/vite)
  - [x] shadcn-ui (Librairie de composants, style Zinc)
  - [x] lucide-react (Icones)
- [x] Accessibilite : @axe-core/playwright

### 1.8 Documentation

- [x] Generateur : sphinx
- [x] Parser Markdown : myst-parser
- [x] Theme Visuel : furo

### 1.9 Infrastructure & DevOps (Docker & SaaS)

- [x] Conteneurisation (Docker Desktop) :
  - [x] Image Docker postgres:15 (Base de donnees relationnelle)
  - [x] Image Docker minio/minio (Stockage Objet type S3)
  - [x] Image Docker prom/prometheus (Monitoring metriques)
  - [x] Image Docker grafana/grafana (Tableaux de bord)
  - [x] Image Docker grafana/loki (Agregation de logs)
- [x] Hebergement : Creation d'un compte sur Render
- [x] CI/CD : Configuration des workflows GitHub Actions (fichiers YAML)

### 1.10 Outils de Veille & Methodologie

- [x] Maquettage : Compte sur Penpot (Wireframing)
- [x] Modelisation BDD : Logiciel Looping (MCD/MLD)
- [x] Veille Techno :
  - [x] Compte Inoreader (Agregateur RSS avec 9-10 flux configures)
  - [x] Compte Perplexity Pro (Recherche IA + Deep Research hebdomadaire configure)

---

## ETAPE 2 : Data Factory & Gestion de Donnees (Bloc E1)

> Couvre la collecte, le nettoyage, et le stockage des donnees.
> A realiser apres l'installation complete.

### 2.1 Conception & Conformite (Avant de coder)

- [x] **Specifications Techniques** : Redaction d'un document Markdown listant les contraintes techniques des sources de donnees (API, Web, Fichiers) et les regles d'extraction
- [x] **Modelisation des Donnees** :
  - [x] Realisation du Modele Conceptuel de Donnees (MCD) sur Looping (Entites : Article, Source, Auteur, Analyse)
  - [x] Generation du Modele Logique de Donnees (MLD) et du script SQL correspondant
- [x] **Registre RGPD & Confidentialite** :
  - [x] Redaction du registre des traitements de donnees (document texte)
  - [x] Definition des procedures de tri pour identifier et anonymiser les eventuelles donnees personnelles (noms d'auteurs, e-mails)

### 2.2 Infrastructure de Stockage

- [x] **Deploiement Base Relationnelle (PostgreSQL)** :
  - [x] Lancement du conteneur Docker PostgreSQL
  - [x] Creation de la base de donnees greentech_db
  - [x] Creation d'un utilisateur dedie avec droits restreints
- [x] **Deploiement Systeme Big Data (MinIO - Stockage Objet)** :
  - [x] Lancement du conteneur Docker MinIO
  - [x] Creation de deux "Buckets" via l'interface console :
    - [x] raw-data : Pour stocker les fichiers bruts (HTML, JSON non traites)
    - [x] clean-data : Pour stocker les donnees nettoyees par Spark

### 2.3 Programmation de la Collecte (Extraction)

#### Module 0 : Configuration Dynamique (Source SQL - PostgreSQL)

- [x] **Creation de la table de configuration** :
  - [x] Developpement d'un script SQL d'initialisation pour la table search_config (Mots-cles, URLs)
  - [x] Insertion des donnees de reference (ex: "Green IT", "Sustainable AI")
- [x] **Developpement du connecteur d'extraction (SQLAlchemy)** :
  - [x] Programmation d'une fonction Python get_config_from_db pour executer la requete SELECT
  - [x] Injection dynamique des parametres recuperes dans les modules API et Scraping (Validation source SQL)

#### Module 1 : Collecte via API (Bibliotheque httpx)

- [x] Developpement d'un script Python pour interroger une API d'actualites technologiques publique
- [x] Configuration des requetes HTTP (headers, timeouts, gestion des erreurs 4xx/5xx)
- [x] Filtrage initial des donnees JSON recues (parsing)
- [x] Sauvegarde des reponses brutes dans le bucket MinIO raw-data

#### Module 2 : Scraping Hybride (Bibliotheque scrapy + playwright)

- [x] Developpement d'un "Spider" Scrapy ciblant un blog technique moderne (site dynamique)
- [x] Integration de Playwright pour charger le JavaScript et effectuer le rendu navigateur des pages
- [x] Extraction du contenu HTML complet des articles
- [x] Gestion des contraintes techniques (User-Agent, delais entre requetes pour eviter le blocage)
- [x] Sauvegarde du HTML brut dans le bucket MinIO raw-data

#### Module 3 : Ingestion de Fichiers (Script Python)

- [x] Developpement d'un script pour lire un dataset historique local (format JSON telecharge depuis Kaggle)
- [x] Programmation de la lecture et du parsing du fichier
- [x] Upload des donnees brutes vers le bucket MinIO raw-data

### 2.4 Traitement Big Data & Nettoyage (Apache Spark)

- [x] **Initialisation Spark** : Configuration d'une session PySpark locale connectee a MinIO (via connecteurs S3)
- [x] **Lecture (Extraction Big Data)** :
  - [x] Programmation de la lecture des donnees brutes depuis le Data Lake MinIO via Spark
  - [x] (Note : Cette etape valide la competence "Connexion et extraction depuis un systeme Big Data")
- [x] **Script de Nettoyage Automatise** :
  - [x] Programmation de la lecture des fichiers bruts depuis MinIO raw-data
  - [x] Developpement des fonctions de nettoyage :
    - [x] Suppression des balises HTML residuelles
    - [x] Identification et suppression des entrees corrompues ou incompletes
    - [x] Homogeneisation des formats de dates (ISO 8601) et des encodages textes
  - [x] Agregation des trois sources (API, Scraping, Fichier) en un DataFrame unique normalise
- [x] **Sauvegarde "Propre"** : Ecriture du jeu de donnees final nettoye dans le bucket MinIO clean-data (format Parquet ou JSON)

### 2.5 Mise a disposition structuree (SQL)

- [x] **Script d'Ingestion SQL (SQLAlchemy)** :
  - [x] Configuration de la connexion asynchrone a la base PostgreSQL
  - [x] Developpement du script qui lit les donnees propres depuis MinIO ou le DataFrame Spark final
  - [x] Insertion des metadonnees structurees (Titre, URL, Date, Auteur, Resume) dans les tables PostgreSQL
  - [x] Gestion des conflits (Upsert) pour eviter les doublons
- [x] **Validation & Documentation** :
  - [x] Execution de requetes SQL de verification pour confirmer que les donnees sont bien extraites et stockees
  - [x] Redaction de la documentation technique expliquant la logique algorithmique, les choix de nettoyage et les commandes pour lancer les scripts
  - [x] Push de l'ensemble du code source (scripts, Dockerfiles, docs) sur le depot distant GitHub

---

## ETAPE 3 : Intelligence Artificielle - Blocs E2 & E3

> Valide l'integration de services IA, la veille technologique, et le deploiement MLOps de modeles custom sur materiel specifique (AMD ROCm).

### 3.1 Veille Technologique & Benchmark (Bloc E2)

- [x] **Configuration de la Veille** :
  - [x] Creation de flux RSS sur Inoreader ciblant "Green IT", "Sustainable AI", "Model Efficiency"
  - [x] Configuration de taches de recherche automatique sur Perplexity Pro pour synthetiser les tendances hebdo
  - [x] Redaction d'une synthese mensuelle (Document Markdown)
- [x] **Benchmark des Services IA (Etude Comparative)** :
  - [x] Redaction d'un document comparant les solutions de resumes automatiques (OpenAI, Mistral, Hugging Face)
  - [x] Criteres d'analyse : Cout, Facilite d'integration, Impact Carbone, Qualite du resume
  - [x] Selection finale : Hugging Face Serverless Inference API (pour le resume)
- [x] **Integration du Service SaaS (Script Python)** :
  - [x] Developpement d'un module Python summarizer.py utilisant la librairie huggingface_hub
  - [x] Connexion a l'API Serverless pour envoyer le texte nettoye des articles
  - [x] Recuperation et stockage du resume genere dans la base PostgreSQL

### 3.2 Preparation des Donnees & MLOps (Data Ops)

- [x] **Creation du Dataset d'Or (Golden Dataset)** :
  - [x] Constitution initiale : 5808 articles depuis 3 sources (arXiv, NewsData.io, TechCrunch)
  - [x] Annotation binaire par scoring multi-criteres (100+ indicateurs ponderes) : "Green IT" (1) ou "Non Green IT" (0)
  - [x] Verification manuelle des resultats et correction iterative des faux positifs/negatifs
  - [x] **Avril 2026 : migration vers The Guardian + Dev.to** : 1316 articles NewsData
    supprimes (contenu tronque en free tier), sources remplacees par The Guardian
    Open Platform (5000 req/jour) et Dev.to (pas de cle requise) pour garantir un
    contenu integral exploitable par le classifieur.
- [x] **Mise en place du Versioning (DVC)** :
  - [x] Initialisation de DVC (dvc init) dans le projet
  - [x] Configuration du "Remote" DVC vers le bucket MinIO clean-data ou un dossier dedie
  - [x] Versioning du fichier dataset annote (golden_dataset.csv.dvc)
  - [x] Push des donnees vers le stockage distant (MinIO s3://models/dvc)

### 3.3 Entrainement & Competition des Modeles (Sur PC Fixe - ROCm)

> Cette phase utilise specifiquement le GPU AMD 7900 XTX via ROCm.

- [x] **Configuration de l'Experience (MLFlow)** :
  - [x] Lancement du serveur MLFlow local pour tracker les metriques (Perte, Precision)
  - [x] Integration de la librairie codecarbon pour mesurer la consommation electrique reelle du GPU pendant l'entrainement
- [x] **Entrainement DeBERTa-v3-base (encoder seq-cls EN-only, fevrier 2026)** :
  - [x] Script de Fine-tuning classique utilisant transformers.Trainer
  - [x] Correction du chargement fp16 → fp32 (transformers 5.1.0 charge en fp16 par defaut, causant loss=0/NaN)
  - [x] Oversampling de la classe minoritaire (22 Green IT → 20% du dataset)
  - [x] Entrainement reussi : F1=0.44, Accuracy=99.6%, Precision=0.40, Recall=0.50
  - [x] Artefacts logges dans MLflow/MinIO, CodeCarbon: 97.8g CO2eq
- [x] **Entrainement Qwen2.5-3B + LoRA (decoder generatif, fevrier 2026)** :
  - [x] Script de Fine-tuning utilisant PEFT (LoRA) pour adapter le modele generatif a la classification
  - [x] Parametrage specifique pour PyTorch sur ROCm (device='cuda', bf16=True)
  - [x] Execution de l'entrainement : F1=0.40, Accuracy=99.74%, Precision=1.00, Recall=0.25
  - [x] Artefacts logges dans MLflow/MinIO, CodeCarbon: 108.8g CO2eq
- [x] **Entrainement Llama 3.2 3B + LoRA (decoder gated, mars 2026)** :
  - [x] Script de Fine-tuning utilisant PEFT (LoRA), meme architecture que Qwen2.5
  - [x] Parametrage specifique pour PyTorch sur ROCm (device='cuda', bf16=True)
  - [x] Acces licence Meta accepte sur HuggingFace (modele gated)
  - [x] Execution de l'entrainement : F1=0.667, Accuracy=99.83%, Precision=1.00, Recall=0.50
  - [x] Artefacts logges dans MLflow/MinIO, CodeCarbon: 112.0g CO2eq
- [x] **Benchmark Final & Selection (mars 2026)** :
  - [x] Execution du script de comparaison sur le jeu de test (3 modeles, 1162 articles)
  - [x] Analyse des metriques MLFlow : F1 vs Latence vs CO2
  - [x] Selection du modele initial : Llama 3.2 3B + LoRA (F1=0.667)
- [x] **Migration vers Qwen3-4B (avril 2026)** :
  - [x] Ajout de `Qwen3Classifier` dans `training.py` (base `Qwen/Qwen3-4B`, Apache-2.0, multilingue natif FR/EN/DE/ES/ZH)
  - [x] Creation du module `baseline.py` pour l'evaluation zero-shot reutilisable
  - [x] Creation du script `scripts/benchmark_baseline.py` avec run MLflow dedie
  - [x] Basculement du pipeline `retrain_pipeline.py` vers `qwen3` (`TRAIN_MODEL_TYPE`)
  - [x] Detection automatique de la famille Qwen3-4B dans `inference.py` pour le chargement LoRA
  - [x] Ajout de `gradient_checkpointing=True` et `max_length=512` dans `LoRAClassifier.train()` pour tenir la VRAM sur RX 7900 XTX
  - [x] Abandon de `Qwen/Qwen3.5-4B` (VLM incompatible ROCm) apres deux freezes systeme au premier step d'entrainement — voir `docs/PROCEDURE_MAJ_MODELE.md` pour le post-mortem

### 3.4 Validation & Packaging (Qualite Modele)

- [x] **Tests Automatises du Modele (Deepchecks)** :
  - [x] Ecriture d'une suite de tests pour verifier l'integrite du modele (Data Leakage, Biais, Robustesse au bruit)
  - [x] Generation d'un rapport de validation automatique
- [x] **Packaging pour Inference** :
  - [x] Sauvegarde du modele gagnant en safetensors (adapter_model.safetensors, 18 Mo) dans models/production/
  - [x] Push du modele valide via DVC vers le stockage partage (MinIO s3://models/dvc)
  - [x] Redaction de la "Model Card" (Documentation du modele : donnees utilisees, limites, metriques)

### 3.5 Deploiement MLOps (Monitoring)

- [x] **Definition des Metriques de Production** :
  - [x] Identification des indicateurs cles a surveiller (Drift des donnees, Temps de reponse, Pourcentage de classification "Green")
- [x] **Configuration du Monitoring (Prometheus)** :
  - [x] Preparation des exporteurs pour envoyer les metriques d'inference vers Prometheus
  - [x] (L'integration effective se fera lors du developpement de l'API a l'etape suivante)
- [x] **Stack Monitoring Complete (Docker)** :
  - [x] Prometheus (metriques), Loki (logs), Grafana (dashboards) operationnels
  - [x] 2 dashboards Grafana provisionnes : "Metier GreenTech" + "Performance Systeme"
  - [x] Datasources Grafana auto-provisionnees (Prometheus + Loki)
  - [x] Integration Loguru → Loki via sink HTTP (logger.py)
  - [x] MLflow Tracking Server Docker (PostgreSQL backend + MinIO S3 artifacts)
  - [x] CodeCarbon integre dans le tracking MLflow (mesure CO2 par run)

---

## ETAPE 4 : Backend & API (Blocs E1 & E4)

> Construire le coeur fonctionnel de l'application : une API REST securisee qui expose les donnees et les fonctionnalites d'IA.

### 4.1 Conception de l'Architecture API (Design)

- [x] **Definition des Endpoints (Specification OpenAPI)** :
  - [x] Identification des routes necessaires :
    - [x] GET /articles : Liste des articles analyses (pagination)
    - [x] GET /articles/{id} : Details d'un article
    - [x] POST /analyze : Point d'entree pour l'analyse IA (URL ou Texte)
    - [x] GET /stats : Statistiques globales (Ratio Green IT)
    - [x] POST /auth/login : Authentification
- [x] **Securisation (OWASP)** :
  - [x] Adoption du standard OAuth2 avec jetons JWT pour l'authentification
  - [x] Definition des regles de validation des entrees (Input Validation) via Pydantic pour eviter les injections

### 4.2 Developpement du Serveur API (FastAPI)

- [x] **Initialisation du Projet FastAPI** :
  - [x] Configuration de l'application principale (titre, version, description)
  - [x] Configuration des regles CORS pour autoriser le futur Frontend React
- [x] **Connexion Base de Donnees (SQLAlchemy + Asyncpg)** :
  - [x] Integration de la connexion asynchrone a PostgreSQL configuree a l'etape 2
  - [x] Creation des modeles Pydantic (Schemas) pour serialiser les reponses JSON
- [x] **Mise en place des Logs (Loguru)** :
  - [x] Configuration du logger pour remplacer les print() par des logs structures (INFO, ERROR, DEBUG)
  - [x] Interception des logs systeme Uvicorn pour centralisation

### 4.3 Implementation de la Securite (Auth)

- [x] **Gestion Utilisateurs (FastAPI Users)** :
  - [x] Configuration du gestionnaire d'authentification (Bearer Transport)
  - [x] Implementation des strategies de creation de compte et de login
  - [x] Connexion de la table User PostgreSQL via l'ORM SQLAlchemy
- [x] **Protection des Routes** :
  - [x] Ajout de dependances de securite sur les routes sensibles (ex: POST /analyze necessite d'etre connecte)

### 4.4 Exposition des Donnees (Bloc E1 - C5)

- [x] **Developpement des Endpoints de Lecture** :
  - [x] Implementation de la logique de recuperation des articles depuis la base de donnees
  - [x] Ajout des filtres (par date, par statut "Green IT")
  - [x] Gestion de la pagination pour ne pas surcharger le client

### 4.5 Integration de l'IA dans l'API (Bloc E3 - C9/C10)

- [x] **Endpoint d'Analyse (POST /analyze) - Le Coeur du Reacteur** :
  - [x] Developpement de la logique d'orchestration :
    1. [x] Reception de l'URL ou du texte
    2. [x] Declenchement du Scraping (si URL) via le module cree a l'etape 2
    3. [x] Declenchement du nettoyage
    4. [x] Chargement du Modele IA vainqueur (DeBERTa, Qwen ou Llama) pour inference
    5. [x] Appel de l'API SaaS Hugging Face pour le resume
    6. [x] Agregation des resultats et sauvegarde en base
  - [x] Optimisation : Chargement du modele IA au demarrage de l'API (pour eviter de le recharger a chaque requete)

### 4.6 Documentation & Tests d'Integration

- [x] **Documentation Automatique (Swagger UI / ReDoc)** :
  - [x] Verification de la generation automatique de la documentation interactive sur /docs
  - [x] Ajout de descriptions et d'exemples pour chaque endpoint
- [x] **Tests API (Pytest + HTTPX)** :
  - [x] Ecriture de tests d'integration pour verifier que les endpoints repondent correctement (Code 200)
  - [x] Test des cas d'erreur (Code 401 Unauthorized, Code 422 Validation Error)
  - [x] Verification de la chaine complete (Input -> BDD -> Output)

---

## ETAPE 5 : Frontend & Application (Bloc E4)

> Realisation de l'interface utilisateur (Application Cliente) en utilisant React et Shadcn/UI, avec un focus fort sur l'accessibilite (Audit Axe-core).

### 5.1 Initialisation de l'environnement Frontend (React & Vite)

- [x] **Creation du projet** :
  - [x] Generation du squelette de l'application via Vite avec le template React + TypeScript
  - [x] Nettoyage des fichiers par defaut pour partir sur une base propre
- [x] **Configuration du Styling (Tailwind CSS)** :
  - [x] Installation et initialisation de Tailwind CSS (v4 via @tailwindcss/vite)
  - [x] Configuration du plugin Vite Tailwind (remplace tailwind.config.js en v4)
  - [x] Ajout des directives Tailwind dans le fichier CSS principal (index.css)
- [x] **Installation du Design System (Shadcn/UI)** :
  - [x] Initialisation de la CLI Shadcn/UI pour configurer les variables CSS globales (theme, radius)
  - [x] Choix d'un style neutre (Zinc) pour un rendu professionnel "Green IT"
  - [x] Installation de la bibliotheque d'icones lucide-react

### 5.2 Developpement des Composants UI (Design System)

- [x] **Installation des composants Shadcn necessaires** :
  - [x] Ajout des composants de base : button, input, card, badge, skeleton, table, dialog, label, separator, tabs, sonner (toast)
- [x] **Creation du Layout Principal** :
  - [x] Developpement d'un composant de mise en page global incluant :
    - [x] Une barre de navigation (Header) responsive avec logo, nav et auth state
    - [x] Un pied de page (Footer) avec mentions legales
    - [x] Un conteneur principal centre et responsive (max-w-5xl)

### 5.3 Developpement des Pages & Parcours Utilisateur

- [x] **Page d'Authentification (Login)** :
  - [x] Creation d'un formulaire de connexion securise (login + register toggle)
  - [x] Integration de la logique de stockage du Token JWT (localStorage)
  - [x] Redirection automatique apres connexion reussie
- [x] **Page Dashboard (Accueil)** :
  - [x] Developpement de la zone d'action principale : Champ de saisie (URL/Texte) + Bouton "Analyser"
  - [x] Developpement de la section "Dernieres Analyses" affichant les articles sous forme de liste interactive
  - [x] Integration de graphiques statistiques (Camembert "Ratio Green IT") via recharts
- [x] **Page Detail Article** :
  - [x] Affichage complet du resultat de l'analyse IA
  - [x] Mise en valeur visuelle du statut (Vert = Green IT, Rouge = Non Green) via Badge colore
  - [x] Affichage du resume genere par l'IA SaaS
  - [x] Affichage du score de confiance du modele IA (barre de progression)

### 5.4 Integration de la Logique Metier (API & State)

- [x] **Connexion Backend (Client HTTP)** :
  - [x] Configuration d'une instance fetch avec l'URL de base de l'API FastAPI (lib/api.ts)
  - [x] Mise en place d'intercepteurs pour injecter automatiquement le Token d'authentification dans les headers
- [x] **Gestion de l'Etat et des Requetes** :
  - [x] Utilisation de Hooks (useEffect, useState, useCallback) pour recuperer les donnees de l'API
  - [x] Gestion des etats de chargement (Skeletons/Spinners) pendant l'analyse IA
  - [x] Gestion des erreurs (affichage de Toasts via Sonner en cas d'echec serveur)

### 5.5 Tests & Accessibilite (Bloc E4 - A8)

- [x] **Audit d'Accessibilite Automatise (Axe-core)** :
  - [x] Installation de la bibliotheque @axe-core/playwright
  - [x] Ecriture d'un script de test Playwright qui navigue sur chaque page de l'application locale (3 pages)
  - [x] Injection du moteur Axe pour scanner la page et detecter les violations WCAG (Contraste couleurs, Labels manquants, Structure HTML)
  - [x] Generation d'un rapport de conformite accessibilite (HTML via Playwright reporter)
- [x] **Verification Responsive & UX** :
  - [x] Tests automatises de l'interface sur differentes resolutions (Mobile 375px, Tablette 768px, Desktop 1280px)
  - [x] Verification de la navigation au clavier (Tabulation) pour tous les elements interactifs

---

## ETAPE 6 : DevOps, Deploiement & Maintenance (Bloc E5)

> Industrialiser le projet en automatisant les tests, le deploiement, et en mettant en place une surveillance proactive de l'application en production.

### 6.1 Automatisation des Tests (CI Pipeline)

- [x] **Configuration de GitHub Actions** :
  - [x] Creation du fichier de workflow YAML (.github/workflows/ci.yml)
  - [x] Definition des declencheurs (push sur main/develop, pull request)
- [x] **Integration des Etapes de Controle Qualite** :
  - [x] Etape de Linting : Execution automatique de ruff (backend) et eslint (frontend)
  - [x] Etape de Tests Unitaires/Integration : Execution de pytest avec couverture (PostgreSQL service)
  - [x] Etape de Tests IA : Execution de deepchecks (job conditionnel sur push main)
  - [x] Etape de Tests Accessibilite : Execution des tests Playwright/Axe-core avec rapport HTML
  - [x] Etape de Build Docker : Construction des images API et Frontend
  - [x] Etape de Securite : Scan pip-audit des dependances

### 6.2 Conteneurisation & Packaging (Docker)

- [x] **Dockerisation du Backend (API)** :
  - [x] Redaction du Dockerfile pour l'API Python (Dockerfile.api)
  - [x] Optimisation via Multi-stage build (builder + runtime, python:3.12-slim, user non-root)
- [x] **Dockerisation du Frontend (React)** :
  - [x] Redaction du Dockerfile pour l'application React (frontend/Dockerfile)
  - [x] Configuration du serveur web NGINX pour distribuer les fichiers statiques buildes
  - [x] Fichier nginx.conf avec gzip, cache assets, SPA fallback, headers securite, proxy API
- [x] **Orchestration Locale (Docker Compose)** :
  - [x] Finalisation du fichier docker-compose.yml pour lancer toute la stack en une commande (API + Front + BDD + MinIO + MLflow + Monitoring)
  - [x] Profil "full" pour lancer API + Frontend en plus de l'infra

### 6.3 Livraison Continue (CD Pipeline)

- [x] **Configuration de l'Hebergement (Render)** :
  - [x] Creation du Blueprint Render (render.yaml) pour deploiement automatique
  - [x] Configuration du Web Service pour l'API Docker
  - [x] Configuration du Static Site pour le Frontend React
- [x] **Automatisation du Deploiement** :
  - [x] Pipeline CD GitHub Actions (.github/workflows/cd.yml) declenche apres CI reussi
  - [x] Verification automatique de la disponibilite (health check API + Frontend)

### 6.4 Monitoring & Observabilite (La Tour de Controle)

- [x] **Collecte de Metriques (Prometheus)** :
  - [x] Configuration finale de Prometheus pour scraper l'API FastAPI (/metrics), MLflow, MinIO
  - [x] Surveillance des indicateurs cles : Latence HTTP, Taux d'erreurs 5xx, Temps d'inference IA, Etat des targets
- [x] **Centralisation des Logs (Loki)** :
  - [x] Configuration du pilote de logging Docker (json-file) pour les conteneurs
  - [x] Integration Loguru → Loki via sink HTTP dans le code Python
- [x] **Tableaux de Bord (Grafana)** :
  - [x] Dashboard "Performance Systeme" (latence, erreurs, targets up/down)
  - [x] Dashboard "Metier GreenTech" (articles analyses, ratio Green/Non-Green, inference)
  - [x] Regles d'alertes Prometheus (latence > 2s, erreurs 5xx > 5%, API/DB/MinIO down, inference lente)

### 6.5 Maintenance & Gestion d'Incidents (A9)

- [x] **Simulation d'Incidents (Chaos Engineering leger)** :
  - [x] Documentation des scenarios de test (coupure BDD, coupure HF API, surcharge)
  - [x] Verification que l'API gere les erreurs gracieusement (try/except, messages utilisateur)
- [x] **Documentation de Maintenance** :
  - [x] Playbook de debogage (docs/PLAYBOOK_MAINTENANCE.md) : lecture logs Grafana, incidents courants, commandes utiles
  - [x] Procedure de mise a jour du modele IA (docs/PROCEDURE_MAJ_MODELE.md) : entrainement, validation, deploiement blue-green, rollback

---

## ETAPE 7 BONUS : Optimisation Demonstration & Refonte Modele (Avril-Mai 2026)

> **Objectif** : Ameliorer la precision du classifieur Green IT pour une demonstration credible au jury de soutenance.
> **Note** : Toutes les competences C1-C21 sont deja validees (voir CHECKLIST_SUIVI.md). Cette etape bonus est une optimisation qualitative qui RENFORCE les competences existantes (C1, C3, C7, C11, C12) sans en invalider ni en ajouter aucune.
> **Reference detaillee** : voir section "BONUS" dans `docs/CHECKLIST_SUIVI.md` pour la checklist complete (B1 a B5).
> **Cible chiffree** : passer de 17 articles Green IT (0.3%) a 200-500 (3-8%), atteindre un MCC > 0.75 stable.
> **Ordre obligatoire** : 7.1 -> 7.2 -> 7.3 -> 7.4, puis 7.5 si temps disponible.

### 7.1 Mise a Jour Infrastructure ROCm (correspond a B1) - TERMINE 2026-04-18

- [x] Verifier la version ROCm actuellement installee (7.1.51803 HIP SDK + wheels 7.2 pre-release)
- [x] Consulter la documentation AMD pour la derniere version stable Windows et la matrice de compatibilite PyTorch (cible : 7.2.1 stable)
- [x] Version plus recente disponible (7.2.1) :
  - [x] Sauvegarder l'environnement actuel (snapshot pip + backup pyproject.toml + uv.lock au 20260418)
  - [x] Desinstaller proprement le HIP SDK 7.1 MSI (reboot effectue)
  - [x] **Decouverte** : ROCm 7.2.1 sur Windows est wheels-only (pas de MSI requis). Le HIP SDK natif n'est plus necessaire, les wheels `rocm_sdk_core/devel/libraries_custom` contiennent le runtime.
  - [x] Mettre a jour torch + torchvision + torchaudio vers les wheels ROCm 7.2.1 (2.9.1+rocm7.2.1 / 0.24.1+rocm7.2.1)
  - [x] Test de validation : import torch, detection GPU, inference rapide Qwen3-4B (28,7 tok/s en bf16)
  - [x] Test entrainement : reporte au re-entrainement effectif du modele de production (K-fold CV servira de validation finale sustained load)
- [x] Documenter la procedure complete dans `docs/PROCEDURE_MAJ_ROCM.md`
- [x] Mettre a jour la documentation interne et la section 1.1 de ce plan avec la nouvelle version (7.2.1)

### 7.2 Enrichissement Dataset Green IT (correspond a B2)

> Toutes les nouvelles sources doivent rester dans les 5 categories C1 deja validees (REST/JSON, scraping, fichiers, BDD, Big Data) et fournir le format `titre + abstract` ou `titre + resume` (resume <= 450 tokens).
>
> **Note 2026-04-19** : le dataset Hugging Face (`climatebert/climate_detection`) a ete ecarte du plan apres validation prealable. Format incompatible (pas de titre, pas d'URL, paragraphes corporate non-tech) et labels "climat general" qui corrompraient les labels Green IT. Les 7 autres sources fournissent largement le volume cible (5 000-8 000 articles attendus).

- [x] **Phase de recherche & validation prealable des 7 sources** (B2.1 terminee 2026-04-19) :
  - [x] arXiv API (REST/JSON) : 8 queries Green IT ciblees, Atom XML, `summary` = abstract (150-300 mots). Volume estime 300-1500 articles.
  - [x] Crossref API (REST/JSON) : Polite Pool via `mailto`, filtre `has-abstract:true + journal-article + from-pub-date:2020`. Volume estime 500-1000.
  - [x] Extension The Guardian : sections `environment` (5932 articles 2024+) et `technology` (2674) validees. Sub-sections `technology/green-computing` n'existent pas cote API, usage du filtre `section=` direct.
  - [x] GreenIT.fr : 1001 posts sitemap, WordPress, HTML statique, 100% Green IT, 5000+ chars/article.
  - [x] Greensoftware.foundation : 170 articles (17 pages x 10), HTML statique, 100% Green IT, 8500 chars/article.
  - [x] Sustainablewebdesign.org : 131 items (50 posts + 81 guidelines), WordPress, HTML statique.
  - [x] Climateaction.tech : 71 posts, WordPress, HTML statique, 4500 chars/article.
- [x] **Implementation des collecteurs REST/JSON** (B2.2 terminee 2026-04-19) :
  - [x] `src/greentech/data/collectors/arxiv_collector.py` (feedparser + httpx, pagination 3s delay, filtre `cs.*|eess.*|stat.ML`)
  - [x] `src/greentech/data/collectors/crossref_collector.py` (httpx, Polite Pool, strip JATS, top 200/query)
  - [x] Extension de `guardian_collector.py` : parametre `sections`, dedup URL, 2 sections par defaut (`environment`, `technology`)
  - [x] Tests unitaires : 30 tests pytest (9 arXiv + 16 Crossref + 5 Guardian), tous verts
- [x] **Implementation des spiders Scrapy (B2.3 terminee 2026-04-19)** :
  - [x] `src/greentech/data/collectors/spiders/base.py` : `StaticArticleSpider` base class (sitemap/pagination discovery, extraction titre/contenu/date/auteur avec fallback trafilatura, tags p/h/ul/li/blockquote pour capture complete)
  - [x] `src/greentech/data/collectors/spiders/greenit_fr_spider.py` (1 001 posts FR, sitemap WordPress)
  - [x] `src/greentech/data/collectors/spiders/greensoftware_spider.py` (170 articles EN, pagination /articles/N)
  - [x] `src/greentech/data/collectors/spiders/sustainable_web_spider.py` (131 items EN, 2 sitemaps)
  - [x] `src/greentech/data/collectors/spiders/climate_action_tech_spider.py` (71 posts EN, sitemap WordPress)
  - [x] `src/greentech/data/collectors/static_scraping_collector.py` : orchestrator lancant les 4 spiders en un seul CrawlerProcess
  - [x] Migration SQL idempotente : `scripts/sql/migration_003_b2_3_spiders.sql` (4 nouvelles sources)
  - [x] Tests unitaires : 23 tests pytest (`test_static_spiders.py`), tous verts
  - [x] Architecture Scrapy HTTP (pas Playwright) : ~5x plus rapide + Green IT compliant. Hooks Playwright-ready dans la base class pour activation future si un site ajoute du JS critique
  - [x] **Optimisation re-run** : flag `skip_existing=True` par defaut, pre-charge toutes les URLs de `articles` en BDD via asyncpg au demarrage, filtre avant scheduling HTTP. Sur les re-runs, saute 99%+ des articles deja connus (gain de temps ~massif). Override CLI via `-a skip_existing=false` pour forcer un re-scrape complet.
- [x] **Mise a jour metadata BDD (B2.2)** : 2 nouvelles sources (`arXiv API`, `Crossref`) + 9 mots-cles `arxiv_api` + 8 mots-cles `crossref` dans `search_config`. Migration SQL idempotente dans `scripts/sql/migration_002_b2_sources_config.sql` appliquee.
- [x] **Lancement collecte massive (B2.6)** : `scripts/retrain_pipeline.py collect` mis a jour pour inclure tous les collecteurs B2 (guardian + devto + arxiv + crossref + techcrunch + static_scraping 4 sites). `spark_cleaner` generalise sur `scraping/` (accepte TechCrunch + 4 spiders B2.3). `sql_ingester` : ajout des mappings pour les 6 nouvelles sources (arXiv API, Crossref, GreenIT.fr, GSF, SWD, CAT) dans `SOURCE_NAME_MAPPING` et dans les heuristiques par URL. Pipeline end-to-end valide sur les smoke tests B2.2+B2.3 (306 articles inseres, 0 erreurs).
- [x] **Nettoyage Spark (B2.7)** : `spark_cleaner.clean_scraping_data` refactore pour gerer deux formats (TechCrunch `contenu_html` + 4 spiders `contenu`), et utiliser la langue de l'article plutot qu'un hardcode `en` (necessaire pour greenit.fr = FR). Lecture du prefixe MinIO `scraping/` au lieu de `scraping/techcrunch` pour embarquer toutes les sources.
- [x] **Generation des resumes de classification (B2.8, terminee 2026-04-19)** : 5 977 articles traites sur 5-9h, 0 resume bidon grace aux filtres low-entropy + withdrawn/retracted ajoutes a `classification_summarizer.py`. Cleanup BDD de 76 articles junk (tests API + preprints retires) avant lancement.
- [x] **Re-classification du corpus etendu (B2.9, TERMINE 2026-04-21 00:32)** :
  - [x] Etage 1 (pre-filtre keywords) : `auto_annotate_dataset.py` — **TERMINE 2026-04-19 22:38** sur 11 667 articles (6 091 NON_GREEN + 4 531 candidats pour le LLM judge)
  - [x] Etage 2 (LLM judge Qwen) : `classify_candidates.py` — **TERMINE 2026-04-20 23:04** en 3h10 sur les 4 528 candidats restants (91 batches de 50, kill-safe). Verdicts : 1 001 Green IT + 3 527 Non Green IT, 0 echec. Fallback Qwen2.5-3B local sur RX 7900 XTX (HF Serverless en quota HTTP 402).
  - [x] Generation resumes Green IT (confirmes uniquement) : `generate_green_summaries.py` — **TERMINE 2026-04-21 00:32** en 1h28 sur 1 002 articles Green IT, 0 echec (8-9 sec/article via Qwen local).
  - [x] Export golden dataset : `export_golden_dataset.py` — **TERMINE 2026-04-21 00:32** (initial) puis **re-execute 2026-04-21 01:46** apres nettoyage linguistique. `data/golden_dataset.csv` contient **11 664 articles** dont **1 018 Green IT (8.73 %)** et 10 646 Non Green IT, 0 article exclu. Dataset final bilingue EN/FR (EN 74.75 % / FR 25.25 %).
- [ ] **Annotation manuelle** des articles borderline :
  - [ ] Identification des borderline (score LLM judge entre 0.3 et 0.7)
  - [ ] Decision sur le volume avec validation utilisateur
  - [ ] Outil CLI : `scripts/manual_annotation_helper.py`
  - [ ] Procedure documentee : `docs/ANNOTATION_MANUELLE.md`
  - [ ] Re-export et versioning DVC
- [ ] **Mise a jour documentation** :
  - [ ] `docs/SPECIFICATIONS_DATA.md` : nouvelles sources
  - [ ] `docs/REGISTRE_RGPD.md` : verifier nouvelles donnees personnelles
  - [ ] Documentation interne : sections Data et Commandes
  - [ ] Documentation Sphinx complete

### 7.3 Optimisation Pipeline d'Entrainement (correspond a B3)

> **Protocole unifie** (fige 2026-04-21 apres synthese de 3 recherches paralleles : imbalanced text classification 2024-2026 + LoRA Qwen3-4B + mDeBERTa fine-tuning bilingue). Les 4 decisions structurantes sont actees.

- [ ] **Stratification croisee `(langue x label)`** (priorite 1, gain principal sur sigma MCC) : remplacer `StratifiedKFold` par `MultilabelStratifiedKFold` du package `iterative-stratification`. Chaque fold doit contenir la distribution exacte 75 % EN / 25 % FR et 8.73 % Green IT. Sans cela, l'ecart-type K-fold explose au-dela de la cible 0.10.
- [ ] **Loss ponderee `class_weight=[1.0, 10.46]`** (CrossEntropy) remplace l'oversampling x84. Ratio calcule sur le train set de chaque fold. Les 3 agents de recherche convergent : BCE pondere / CE ponderee est la SOTA sur ratio modere 1:10, Focal Loss n'apporte de gain qu'au-dela de 1:50 (et degrade la calibration sans temperature scaling).
- [ ] **Back-translation EN<->FR sur les positifs** via `Helsinki-NLP/opus-mt` (MarianMT, 2x 75M params) :
  - [ ] Nouveau module `src/greentech/data/processors/back_translator.py` + script `scripts/augment_positives.py`
  - [ ] Pour chaque positif, generer 1 variante via langue pivot (EN positif -> FR -> EN, FR positif -> EN -> FR)
  - [ ] Filtre qualite : rejeter si similarite cosine `sentence-transformers` (original, retraduit) hors [0.85, 0.99]
  - [ ] 1 018 positifs -> ~2 036 positifs effectifs, ratio 1:10.5 -> ~1:5.25
  - [ ] Les variantes vont UNIQUEMENT dans le train split de chaque fold (jamais val/test) pour eviter toute fuite d'evaluation
  - [ ] N'appliquer que sur le `resume` (150-220 mots), pas sur le `titre` (trop court)
  - [ ] Temps estime : ~20-30 min sur RX 7900 XTX (MarianMT 75M params, batch 32)
- [ ] **Calibration post-training** : nouveau module `src/greentech/ai/mlops/calibration.py`
  - [ ] **Temperature scaling** : 1 parametre T scale, optimise sur val MCC apres training. Platt/isotonic ecartes (overfit avec <1 000 positifs).
  - [ ] **Threshold tuning** : scan seuils [0.05, 0.95] pas 0.01, retenir argmax MCC sur val
  - [ ] Persister `temperature.json` + `optimal_threshold.json` dans le dossier du modele
  - [ ] `inference.py` : charger au startup, appliquer T puis seuil
- [ ] **Ensemble K-fold K=5** : pour la production, moyenner les logits des 5 modeles K-fold plutot qu'entrainer un modele final sur le full train. Gain MCC +0.03 a +0.07 documente.
  - [ ] Qwen3-4B : fusionner les 5 adapters LoRA via `PeftModel.merge_and_unload()` -> 1 seul modele prod (cout inference 1x)
  - [ ] mDeBERTa : moyenner les logits a l'inference (cout ~5x, ~5.5 Go VRAM sur RX 7900 XTX, OK)
- [ ] **3 seeds par fold** (15 trainings par modele) : variance inter-seed sur DeBERTa petit dataset est ±0.03-0.05 MCC. Moyenner sur 3 seeds stabilise sigma < 0.10.
- [ ] **Validation Deepchecks renforcee** : verification data leakage, distribution drift entre folds, robustesse au bruit. Inchange.
- [ ] **Decision finale** : conserver le modele avec MCC moyen K-fold le plus eleve ET ecart-type < 0.10. Critere secondaire : latence < 200 ms sur RX 7900 XTX.

### 7.4 Benchmark Final & Selection du modele (correspond a B4)

- [x] **Selection de la version DeBERTa adaptee** : **`mdeberta-v3-base`** (multilingue) retenu le 2026-04-21. Justification : dataset final bilingue EN 74.75 % / FR 25.25 % (1 018 Green IT dont 600 en FR). `deberta-v3-base` EN-pur encoderait mal les 600 Green IT francais et fausserait le benchmark en faveur de Qwen3. mDeBERTa couvre 100 langues dont EN + FR avec la meme architecture que DeBERTa-v3-base (278M params, encoder-only, DisentangledSelfAttention), garantissant un benchmark equitable encoder-vs-decoder contre Qwen3-4B (decoder generatif, 4B params).
- [ ] **Benchmark BRUT (zero-shot)** sur le nouveau dataset : run MLflow `baseline-comparison-2026-04` avec mDeBERTa-v3-base et Qwen3-4B
- [ ] **Entrainement des 2 modeles avec le protocole unifie B3** (stratification langue x label, class_weight, 3 seeds, ensemble K=5, calibration) :
  - [ ] **Qwen3-4B + LoRA** (`Qwen3Classifier` actualise) :
    - `target_modules="all-linear"` (attention + MLP : `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
    - `r=32, lora_alpha=64, lora_dropout=0.05`
    - Head : `AutoModelForSequenceClassification` num_labels=2 (latence / 5-10 vs generation)
    - `lr=1e-4`, scheduler cosine, warmup_ratio=0.06
    - 3 epochs, early stopping sur val MCC (patience 1)
    - `batch=2, grad_accum=16` (effectif 32), `max_length=512`, `bf16=True`
    - `model.enable_input_require_grads()` AVANT `get_peft_model()` (piege PEFT + gradient checkpointing, issue HF #42947)
    - `gradient_checkpointing=True, use_reentrant=False`
    - Run MLflow `qwen3-final-2026-04`
    - Fusion des 5 adapters K-fold via `PeftModel.merge_and_unload()` -> 1 modele prod unique
  - [ ] **mDeBERTa-v3-base** (nouvelle classe `MDeBERTaClassifier`) :
    - `lr=2e-5`, scheduler linear, `warmup_ratio=0.06`
    - `batch=16, grad_accum=2` (effectif 32), `max_length=384` (couvre 98 % des resumes FR + titre)
    - 5 epochs, early stopping sur val MCC (patience 2)
    - `weight_decay=0.01`, dropout 0.1 (default)
    - **Precision** : `bf16=True` si `transformers >= 4.48` (bug #35332 corrige), sinon `fp32` (fp16 = NaN garanti, strictement interdit)
    - `attn_implementation="sdpa"` (Flash-Attention indisponible sur RDNA3)
    - `gradient_checkpointing=True`
    - Run MLflow `mdeberta-final-2026-04`
    - Ensemble : moyenne des logits des 5 modeles K-fold a l'inference (cout ~5x, ~5.5 Go VRAM sur RX 7900 XTX, OK)
- [ ] **Benchmark comparatif des modeles entraines** : `scripts/benchmark_models.py`, evaluation sur test set fige, metriques completes (MCC, F1, precision, recall, latence p50/p95/p99, VRAM peak, CO2 CodeCarbon). Produit `docs/BENCHMARK_FINAL_2026-04.md` + `models/benchmark_final_metrics.json`.
- [ ] **Selection du modele retenu** : MCC moyen K-fold le plus eleve ET ecart-type < 0.10 ET latence < 200 ms
- [ ] **Promotion du modele retenu** : `models/production/` + tag DVC + push MinIO, avec `temperature.json` et `optimal_threshold.json` associes
- [ ] **Validation end-to-end** : tests Deepchecks + API + Frontend + dashboards Grafana
- [ ] **Mise a jour documentation finale** : Model Card, documentation interne, section 3.3 de ce plan, tag Git

### 7.5 (BONUS) Refonte Agentic avec LangGraph (correspond a B5)

> A realiser uniquement si 7.1 a 7.4 sont valides ET temps disponible avant la soutenance.

- [ ] **Conception architecture** : `docs/ARCHITECTURE_AGENTIC.md` avec schema agents, etat partage, flux, fallback, observabilite
- [ ] **Setup LangGraph** : `uv add langgraph langchain-core`, package `src/greentech/ai/agents/`, etat partage
- [ ] **Implementation des 5 agents** :
  - [ ] Agent Cleaner (nettoyage + detection langue + traduction)
  - [ ] Agent Summarizer (resume classification + resume Green IT)
  - [ ] Agent Judge Green IT (pre-filtre + LLM judge)
  - [ ] Agent Classifier (modele fine-tune retenu en production)
  - [ ] Agent Orchestrator (workflow LangGraph + retry + logs)
- [ ] **Tests unitaires & integration** : par agent + orchestrator + fallback + reprise
- [ ] **Integration dans l'API** : refactor `routes/analyze.py` pour appeler l'orchestrator
- [ ] **Observabilite agentic** : metriques Prometheus par agent + dashboard Grafana dedie + alertes
- [ ] **Documentation finale** : Sphinx + demo interactive + documentation interne + tag Git
