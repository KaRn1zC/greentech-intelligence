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
  - [x] Suite AMD ROCm (HIP SDK) version 7.1 + ROCm SDK 7.2 pour PyTorch

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
  - [x] torch (PyTorch) version 2.9.1+rocmsdk20260116 avec ROCm 7.2
  - [x] torchvision 0.24.1 & torchaudio 2.9.1 (versions ROCm)
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

- [ ] Build Tool : Vite (Template React TypeScript)
- [ ] Framework UI : React
- [ ] Composants & Design :
  - [ ] tailwindcss (CSS Utility-first)
  - [ ] shadcn-ui (Librairie de composants)
  - [ ] lucide-react (Icones)
- [ ] Accessibilite : @axe-core/playwright

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

- [ ] **Creation du Dataset d'Or (Golden Dataset)** :
  - [ ] Selection manuelle de 200 articles varies depuis la base
  - [ ] Annotation binaire manuelle (Labeling) : "Green IT" (1) ou "Non Green IT" (0)
- [x] **Mise en place du Versioning (DVC)** :
  - [x] Initialisation de DVC (dvc init) dans le projet
  - [x] Configuration du "Remote" DVC vers le bucket MinIO clean-data ou un dossier dedie
  - [ ] Versioning du fichier dataset annote (dataset.csv.dvc)
  - [ ] Push des donnees vers le stockage distant pour synchronisation entre PC Fixe et Portable

### 3.3 Entrainement & Competition des Modeles (Sur PC Fixe - ROCm)

> Cette phase utilise specifiquement le GPU AMD 7900 XTX via ROCm.

- [x] **Configuration de l'Experience (MLFlow)** :
  - [x] Lancement du serveur MLFlow local pour tracker les metriques (Perte, Precision)
  - [x] Integration de la librairie codecarbon pour mesurer la consommation electrique reelle du GPU pendant l'entrainement
- [x] **Entrainement du Challenger (Llama 3.2 3B)** :
  - [x] Script de Fine-tuning utilisant PEFT (LoRA) pour adapter le modele generatif a la classification
  - [x] Parametrage specifique pour PyTorch sur ROCm (device='cuda')
  - [x] Enregistrement des logs et du modele final
- [x] **Entrainement du Champion (DeBERTa-v3-base)** :
  - [x] Script de Fine-tuning classique utilisant transformers.Trainer
  - [x] Optimisation des hyperparametres pour la classification de sequence
  - [x] Enregistrement des logs et du modele final
- [ ] **Benchmark Final & Selection** :
  - [ ] Execution du script de comparaison sur le jeu de test
  - [ ] Analyse des graphiques MLFlow : Comparaison Precision vs Latence vs CO2
  - [ ] Selection du modele vainqueur pour la production

### 3.4 Validation & Packaging (Qualite Modele)

- [x] **Tests Automatises du Modele (Deepchecks)** :
  - [x] Ecriture d'une suite de tests pour verifier l'integrite du modele (Data Leakage, Biais, Robustesse au bruit)
  - [x] Generation d'un rapport de validation automatique
- [ ] **Packaging pour Inference** :
  - [ ] Conversion/Sauvegarde du modele gagnant dans un format optimise (ex: safetensors ou ONNX si applicable)
  - [ ] Push du modele valide via DVC vers le stockage partage
  - [ ] Redaction de la "Model Card" (Documentation du modele : donnees utilisees, limites, metriques)

### 3.5 Deploiement MLOps (Monitoring)

- [x] **Definition des Metriques de Production** :
  - [x] Identification des indicateurs cles a surveiller (Drift des donnees, Temps de reponse, Pourcentage de classification "Green")
- [x] **Configuration du Monitoring (Prometheus)** :
  - [x] Preparation des exporteurs pour envoyer les metriques d'inference vers Prometheus
  - [x] (L'integration effective se fera lors du developpement de l'API a l'etape suivante)

---

## ETAPE 4 : Backend & API (Blocs E1 & E4)

> Construire le coeur fonctionnel de l'application : une API REST securisee qui expose les donnees et les fonctionnalites d'IA.

### 4.1 Conception de l'Architecture API (Design)

- [ ] **Definition des Endpoints (Specification OpenAPI)** :
  - [ ] Identification des routes necessaires :
    - [ ] GET /articles : Liste des articles analyses (pagination)
    - [ ] GET /articles/{id} : Details d'un article
    - [ ] POST /analyze : Point d'entree pour l'analyse IA (URL ou Texte)
    - [ ] GET /stats : Statistiques globales (Ratio Green IT)
    - [ ] POST /auth/login : Authentification
- [ ] **Securisation (OWASP)** :
  - [ ] Adoption du standard OAuth2 avec jetons JWT pour l'authentification
  - [ ] Definition des regles de validation des entrees (Input Validation) via Pydantic pour eviter les injections

### 4.2 Developpement du Serveur API (FastAPI)

- [ ] **Initialisation du Projet FastAPI** :
  - [ ] Configuration de l'application principale (titre, version, description)
  - [ ] Configuration des regles CORS pour autoriser le futur Frontend React
- [ ] **Connexion Base de Donnees (SQLAlchemy + Asyncpg)** :
  - [ ] Integration de la connexion asynchrone a PostgreSQL configuree a l'etape 2
  - [ ] Creation des modeles Pydantic (Schemas) pour serialiser les reponses JSON
- [ ] **Mise en place des Logs (Loguru)** :
  - [ ] Configuration du logger pour remplacer les print() par des logs structures (INFO, ERROR, DEBUG)
  - [ ] Interception des logs systeme Uvicorn pour centralisation

### 4.3 Implementation de la Securite (Auth)

- [ ] **Gestion Utilisateurs (FastAPI Users)** :
  - [ ] Configuration du gestionnaire d'authentification (Bearer Transport)
  - [ ] Implementation des strategies de creation de compte et de login
  - [ ] Connexion de la table User PostgreSQL via l'ORM SQLAlchemy
- [ ] **Protection des Routes** :
  - [ ] Ajout de dependances de securite sur les routes sensibles (ex: POST /analyze necessite d'etre connecte)

### 4.4 Exposition des Donnees (Bloc E1 - C5)

- [ ] **Developpement des Endpoints de Lecture** :
  - [ ] Implementation de la logique de recuperation des articles depuis la base de donnees
  - [ ] Ajout des filtres (par date, par statut "Green IT")
  - [ ] Gestion de la pagination pour ne pas surcharger le client

### 4.5 Integration de l'IA dans l'API (Bloc E3 - C9/C10)

- [ ] **Endpoint d'Analyse (POST /analyze) - Le Coeur du Reacteur** :
  - [ ] Developpement de la logique d'orchestration :
    1. [ ] Reception de l'URL ou du texte
    2. [ ] Declenchement du Scraping (si URL) via le module cree a l'etape 2
    3. [ ] Declenchement du nettoyage
    4. [ ] Chargement du Modele IA Custom (DeBERTa ou Llama) pour inference
    5. [ ] Appel de l'API SaaS Hugging Face pour le resume
    6. [ ] Agregation des resultats et sauvegarde en base
  - [ ] Optimisation : Chargement du modele IA au demarrage de l'API (pour eviter de le recharger a chaque requete)

### 4.6 Documentation & Tests d'Integration

- [ ] **Documentation Automatique (Swagger UI / ReDoc)** :
  - [ ] Verification de la generation automatique de la documentation interactive sur /docs
  - [ ] Ajout de descriptions et d'exemples pour chaque endpoint
- [ ] **Tests API (Pytest + HTTPX)** :
  - [ ] Ecriture de tests d'integration pour verifier que les endpoints repondent correctement (Code 200)
  - [ ] Test des cas d'erreur (Code 401 Unauthorized, Code 422 Validation Error)
  - [ ] Verification de la chaine complete (Input -> BDD -> Output)

---

## ETAPE 5 : Frontend & Application (Bloc E4)

> Realisation de l'interface utilisateur (Application Cliente) en utilisant React et Shadcn/UI, avec un focus fort sur l'accessibilite (Audit Axe-core).

### 5.1 Initialisation de l'environnement Frontend (React & Vite)

- [ ] **Creation du projet** :
  - [ ] Generation du squelette de l'application via Vite avec le template React + TypeScript
  - [ ] Nettoyage des fichiers par defaut pour partir sur une base propre
- [ ] **Configuration du Styling (Tailwind CSS)** :
  - [ ] Installation et initialisation de Tailwind CSS
  - [ ] Configuration du fichier tailwind.config.js pour definir les chemins des contenus
  - [ ] Ajout des directives Tailwind dans le fichier CSS principal
- [ ] **Installation du Design System (Shadcn/UI)** :
  - [ ] Initialisation de la CLI Shadcn/UI pour configurer les variables CSS globales (theme, radius)
  - [ ] Choix d'un style neutre (Slate ou Zinc) pour un rendu professionnel "Green IT"
  - [ ] Installation de la bibliotheque d'icones lucide-react

### 5.2 Developpement des Composants UI (Design System)

- [ ] **Installation des composants Shadcn necessaires** :
  - [ ] Ajout des composants de base via la CLI : button, input, card, badge, table, dialog, form, toast
- [ ] **Creation du Layout Principal** :
  - [ ] Developpement d'un composant de mise en page global incluant :
    - [ ] Une barre de navigation (Header) responsive
    - [ ] Un pied de page (Footer) avec mentions legales
    - [ ] Un conteneur principal centre et responsive

### 5.3 Developpement des Pages & Parcours Utilisateur

- [ ] **Page d'Authentification (Login)** :
  - [ ] Creation d'un formulaire de connexion securise
  - [ ] Integration de la logique de stockage du Token JWT (localStorage ou Cookie)
  - [ ] Redirection automatique apres connexion reussie
- [ ] **Page Dashboard (Accueil)** :
  - [ ] Developpement de la zone d'action principale : Champ de saisie (URL/Texte) + Bouton "Analyser"
  - [ ] Developpement de la section "Dernieres Analyses" affichant les articles sous forme de cartes ou tableau
  - [ ] Integration de graphiques statistiques (ex: Camembert "Ratio Green IT") via une librairie de chart compatible
- [ ] **Page Detail Article** :
  - [ ] Affichage complet du resultat de l'analyse IA
  - [ ] Mise en valeur visuelle du statut (Vert = Green IT, Rouge = Non Green)
  - [ ] Affichage du resume genere par l'IA SaaS
  - [ ] Affichage du score de confiance du modele IA

### 5.4 Integration de la Logique Metier (API & State)

- [ ] **Connexion Backend (Client HTTP)** :
  - [ ] Configuration d'une instance fetch ou axios avec l'URL de base de l'API FastAPI
  - [ ] Mise en place d'intercepteurs pour injecter automatiquement le Token d'authentification dans les headers
- [ ] **Gestion de l'Etat et des Requetes** :
  - [ ] Utilisation de Hooks (useEffect, useState) ou d'une librairie (React Query) pour recuperer les donnees de l'API
  - [ ] Gestion des etats de chargement (Skeletons/Spinners) pendant l'analyse IA
  - [ ] Gestion des erreurs (affichage de "Toasts" en cas d'echec serveur)

### 5.5 Tests & Accessibilite (Bloc E4 - A8)

- [ ] **Audit d'Accessibilite Automatise (Axe-core)** :
  - [ ] Installation de la bibliotheque @axe-core/playwright
  - [ ] Ecriture d'un script de test Playwright qui navigue sur chaque page de l'application locale
  - [ ] Injection du moteur Axe pour scanner la page et detecter les violations WCAG (Contraste couleurs, Labels manquants, Structure HTML)
  - [ ] Generation d'un rapport de conformite accessibilite
- [ ] **Verification Responsive & UX** :
  - [ ] Test manuel de l'interface sur differentes resolutions (Mobile, Tablette, Desktop)
  - [ ] Verification de la navigation au clavier (Tabulation) pour tous les elements interactifs

---

## ETAPE 6 : DevOps, Deploiement & Maintenance (Bloc E5)

> Industrialiser le projet en automatisant les tests, le deploiement, et en mettant en place une surveillance proactive de l'application en production.

### 6.1 Automatisation des Tests (CI Pipeline)

- [ ] **Configuration de GitHub Actions** :
  - [ ] Creation du fichier de workflow YAML (.github/workflows/ci.yml)
  - [ ] Definition des declencheurs (push sur main, pull request)
- [ ] **Integration des Etapes de Controle Qualite** :
  - [ ] Etape de Linting : Execution automatique de ruff pour verifier le style du code
  - [ ] Etape de Tests Unitaires/Integration : Execution de pytest pour valider le Backend
  - [ ] Etape de Tests IA : Execution de deepchecks pour valider la robustesse du modele
  - [ ] Etape de Tests Accessibilite : Execution des tests Playwright/Axe-core pour valider le Frontend

### 6.2 Conteneurisation & Packaging (Docker)

- [ ] **Dockerisation du Backend (API)** :
  - [ ] Redaction du Dockerfile pour l'API Python
  - [ ] Optimisation via "Multi-stage build" pour reduire la taille de l'image finale
- [ ] **Dockerisation du Frontend (React)** :
  - [ ] Redaction du Dockerfile pour l'application React
  - [ ] Configuration du serveur web leger (NGINX ou module serve) pour distribuer les fichiers statiques buildes
- [ ] **Orchestration Locale (Docker Compose)** :
  - [ ] Finalisation du fichier docker-compose.yml pour lancer toute la stack en une commande (API + Front + BDD + MinIO + Monitoring)

### 6.3 Livraison Continue (CD Pipeline)

- [ ] **Configuration de l'Hebergement (Render)** :
  - [ ] Liaison du compte Render au depot GitHub
  - [ ] Creation d'un "Web Service" pour l'API Docker
  - [ ] Creation d'un "Static Site" pour le Frontend React
- [ ] **Automatisation du Deploiement** :
  - [ ] Configuration du declenchement automatique du deploiement a chaque push valide sur la branche main
  - [ ] Verification de la disponibilite de l'application en ligne (URL publique)

### 6.4 Monitoring & Observabilite (La Tour de Controle)

- [ ] **Collecte de Metriques (Prometheus)** :
  - [ ] Configuration finale de Prometheus pour "scraper" les metriques exposees par l'API FastAPI (/metrics)
  - [ ] Surveillance des indicateurs cles : CPU, Memoire, Latence des requetes HTTP, Nombre d'erreurs 500
- [ ] **Centralisation des Logs (Loki)** :
  - [ ] Configuration du pilote de logging Docker pour envoyer les logs des conteneurs vers Loki
  - [ ] Verification de la remontee des logs structures (JSON) generes par Loguru
- [ ] **Tableaux de Bord (Grafana)** :
  - [ ] Creation d'un Dashboard "Performance Systeme" (Utilisation ressources)
  - [ ] Creation d'un Dashboard "Metier GreenTech" (Nombre d'articles analyses, Ratio Green/Non-Green, Temps moyen d'analyse IA)
  - [ ] Configuration d'alertes visuelles (ex: seuil rouge si latence > 2s)

### 6.5 Maintenance & Gestion d'Incidents (A9)

- [ ] **Simulation d'Incidents (Chaos Engineering leger)** :
  - [ ] Test de coupure volontaire de la base de donnees ou de l'API externe Hugging Face
  - [ ] Verification que l'application ne crashe pas brutalement et affiche un message d'erreur utilisateur comprehensible
- [ ] **Documentation de Maintenance** :
  - [ ] Redaction d'une procedure de debogage (Playbook) expliquant comment lire les logs dans Grafana pour identifier une erreur
  - [ ] Documentation de la procedure de mise a jour du modele IA en production sans interruption de service
