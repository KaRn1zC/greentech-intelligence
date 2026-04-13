# Checklist de Suivi - Projet Chef d'Oeuvre (E1 a E5)

> **Chaque case doit etre cochee pour garantir la validation du diplome.**
> Mise a jour reguliere au fil de l'avancement du projet.

---

## BLOC E1 : Gerer des donnees de l'information (Data)

### A1. Programmer la collecte de donnees

#### C1. Automatiser l'extraction de donnees

- [x] Identifier les contraintes techniques propres aux sources de donnees (documentation, confidentialite, etc.)
- [x] Rediger les specifications techniques pour l'extraction des donnees
- [x] Construire les requetes HTTP pour recuperer des donnees depuis un service web (API REST)
- [x] Programmer la lecture d'un fichier de donnees dans un script
- [x] Programmer le telechargement du HTML d'une ou plusieurs pages web (Scraping)
- [x] Etablir une connexion programmatique a un Systeme de Gestion de Base de Donnees (SGBD)
- [x] Etablir une connexion programmatique a un systeme Big Data (Hive, Impala, etc.)
- [x] Programmer le filtrage/parsing des donnees depuis l'API
- [x] Programmer le filtrage/parsing des donnees depuis les fichiers
- [x] Programmer le filtrage/parsing des donnees depuis le HTML (Scraping)
- [x] Executer des requetes d'extraction de type SQL via le script
- [x] Executer des requetes d'extraction depuis le systeme Big Data via le script
- [x] Le script est fonctionnel et recupere toutes les donnees visees
- [x] Le script est structure (initialisation, traitement, gestion erreurs, sauvegarde)
- [x] Le script est versionne et accessible sur un depot Git distant

#### C2. Developper des requetes SQL d'extraction

- [x] Ecrire les requetes SQL pour extraire les donnees de la base de donnees relationnelle
- [x] Ecrire les requetes pour extraire les donnees du systeme Big Data
- [x] Les requetes sont fonctionnelles et extraient exactement les donnees visees
- [x] Documenter les requetes (choix de selections, filtrages, jointures)
- [x] La documentation explicite les optimisations appliquees aux requetes

#### C3. Agregation et nettoyage des donnees

- [x] Rediger les specifications techniques pour l'agregation
- [x] Programmer l'agregation des donnees de toutes les sources en un jeu unique
- [x] Programmer l'identification des entrees corrompues (partielles/manquantes)
- [x] Programmer la suppression des entrees corrompues
- [x] Programmer l'identification des formats non normalises
- [x] Programmer l'homogeneisation des formats (dates, unites, etc.)
- [x] Le script produit un jeu de donnees final unique, nettoye et normalise
- [x] Le script est versionne sur Git
- [x] La documentation du script est complete (dependances, logique algo, choix de nettoyage)

### A2. Developper la mise a disposition technique des donnees

#### C4. Creation de la base de donnees (et RGPD)

- [x] Rediger les specifications techniques pour le stockage
- [x] Modeliser la structure des donnees selon la methode Merise (MCD/MLD)
- [x] Le modele physique est fonctionnel et integre sans erreur
- [x] Choisir un SGBD adapte aux contraintes
- [x] Creer la base de donnees
- [x] Documenter la procedure d'installation du SGBD
- [x] Rediger/mettre a jour le registre des traitements de donnees personnelles (RGPD)
- [x] Rediger les procedures de tri des donnees personnelles (detection, suppression, conformite RGPD)
- [x] Programmer le script d'import des donnees dans la base
- [x] Le script d'import est fonctionnel
- [x] Le script d'import est versionne et documente (dependances, commandes)

#### C5. Developper une API de mise a disposition (REST)

- [x] Rediger les specifications techniques de l'API et de l'acces aux donnees
- [x] Configurer les acces aux donnees depuis le serveur API
- [x] Developper la reception et la validation des requetes client
- [x] Developper les requetes BDD declenchees par l'API
- [x] Developper les reponses de l'API au client
- [x] Developper les regles d'autorisation/acces aux endpoints
- [x] Securiser l'API (ex: Top 10 OWASP)
- [x] L'API est fonctionnelle (acces et mise a disposition des donnees)
- [x] La documentation technique couvre tous les endpoints
- [x] La documentation couvre l'authentification/autorisation
- [x] La documentation respecte les standards (ex: OpenAPI/Swagger)

---

## BLOC E2 : Integrer des services d'IA (Veille & SaaS)

### A3. Accompagner le choix et l'integration d'un service IA

#### C6. Veille technique et reglementaire

- [x] Definir la/les thematiques de veille (liees au projet)
- [x] Planifier des temps de veille reguliers (ex: hebdo)
- [x] Choisir un outil d'agregation de flux (RSS, newsletter, etc.)
- [x] Choisir un outil de partage/communication des syntheses
- [x] Identifier des sources fiables (dont accessibilite, securite, data)
- [x] Qualifier la fiabilite des sources
- [x] Configurer les outils d'agregation
- [x] Rediger des syntheses des informations collectees
- [x] Communiquer les syntheses aux parties prenantes (format accessible)
- [x] Les sources identifiees respectent les criteres de fiabilite (auteur identifie, date recente, sources citees, confirmation par d'autres sites de confiance)

#### C7. Identifier des services IA (Benchmark)

- [x] Reformuler le besoin (problematique technique/fonctionnelle)
- [x] Identifier les contraintes (moyens, techniques, ops)
- [x] Realiser un benchmark des services existants (etudies vs non etudies)
- [x] Expliciter les raisons d'exclusion des services
- [x] Le benchmark detaille l'adequation fonctionnelle
- [x] Le benchmark detaille la demarche eco-responsable des services
- [x] Le benchmark detaille les contraintes techniques/pre-requis
- [x] Rediger des conclusions preconisant une ou plusieurs solutions

#### C8. Parametrer un service IA

- [x] Creer l'environnement d'execution (Compte SaaS, VPS, etc.)
- [x] Installer/configurer les dependances (SDK, outils)
- [x] Gerer les acces (comptes, groupes, droits)
- [x] Installer/configurer les outils de monitoring du service
- [x] Le service est accessible et authentifie
- [x] Le service repond aux besoins fonctionnels
- [x] Le monitoring est operationnel
- [x] La documentation technique couvre installation, acces, dependances
- [x] La documentation est accessible (normes handicap)

---

## BLOC E3 : Deployer un modele d'IA (MLOps)

### A4. Realiser l'integration d'un modele IA

#### C9. Developper une API exposant un modele IA

- [x] Analyser les specs fonctionnelles/techniques
- [x] Concevoir l'architecture de l'API (endpoints, regles)
- [x] Choisir les outils/langages
- [x] Installer l'environnement de developpement
- [x] L'API verifie et transforme les parametres envoyes par le client
- [x] L'API execute le modele a partir de la requete
- [x] L'API renvoie le resultat au client
- [x] Developper l'authentification/autorisation
- [x] Securiser l'API (OWASP)
- [x] Developper des tests d'integration pour les endpoints
- [x] Les tests passent sans bug
- [x] Le code est versionne sur Git distant
- [x] La documentation (OpenAPI) est redigee et accessible
- [x] La documentation respecte les standards du modele choisi (OpenAPI/Swagger)
- [x] La documentation est communiquee dans un format respectant les recommandations d'accessibilite (Valentin Haüy / Microsoft)

#### C10. Integrer l'API IA dans une application

- [x] Installer l'environnement de l'application cliente
- [x] Programmer l'authentification vers l'API IA
- [x] Programmer la communication avec les endpoints
- [x] Integrer les adaptations d'interface necessaires
- [x] Tester l'accessibilite des interfaces modifiees
- [x] Developper des tests d'integration (cote app) sur le perimetre API
- [x] Les tests s'executent en totalite sans erreur
- [x] Le code est versionne sur le Git de l'application

### A5. Deploiement MLOps

#### C11. Monitorer le modele IA

- [x] Lister les metriques a monitorer (perf modele, stabilite data, sante systeme)
- [x] Definir les declencheurs de reentrainement
- [x] Choisir l'outil de monitoring et de consolidation
- [x] Integrer les collecteurs de donnees
- [x] Integrer l'outil de restitution (Dashboard type Grafana/Streamlit)
- [x] Au moins un vecteur de restitution des metriques en temps reel est propose
- [x] Les metriques monitorees sont expliquees sans erreur d'interpretation
- [x] Configurer les alertes (email, notif)
- [x] L'outil de restitution respecte les enjeux d'accessibilite
- [x] La chaine de monitoring est d'abord testee dans un bac a sable / environnement dedie
- [x] La chaine de monitoring est testee et fonctionnelle
- [x] Code versionne sur Git
- [x] Documentation technique (install/maintenance) et utilisateur redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite

#### C12. Tests automatises du modele

- [x] Definir le perimetre des tests (format data, entrainement, eval)
- [x] Choisir les outils de test (ex: unittest, pytest)
- [x] Configurer l'environnement de test
- [x] Integrer les tests (assertions, mocks, fixtures)
- [x] Les tests couvrent le perimetre defini
- [x] Code et Donnees (si possible DVC) versionnes
- [x] Documentation technique des tests redigee
- [x] Documentation couvre la procedure d'installation, l'execution des tests et le calcul de la couverture
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite

#### C13. Chaine de livraison continue (CI/CD pour IA)

- [x] Definir etapes, taches et declencheurs (Pipeline)
- [x] Parametrer la chaine (variables d'env, versions)
- [x] Integrer l'etape de test des donnees
- [x] Integrer l'etape de test/entrainement/validation du modele
- [x] Integrer la generation de rapports (matrix confusion, accuracy) dans la livraison
- [x] Integrer l'etape de livraison (ex: Pull Request automatique) avec rapports d'evaluation attaches
- [x] Les fichiers de config CI/CD sont reconnus et executes
- [x] Code versionne sur Git
- [x] Documentation de la chaine CI/CD redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite

---

## BLOC E4 : Realiser une application (Dev App)

### A6. Concevoir l'application

#### C14. Analyser le besoin

- [x] La modelisation des donnees respecte un formalisme (Merise/Entite-Relation)
- [x] La modelisation des parcours utilisateurs respecte un formalisme (Wireframes/Schema)
- [x] Les User Stories couvrent contexte, scenarios et criteres de validation
- [x] Les objectifs d'accessibilite sont integres aux criteres d'acceptation (User Stories)
- [x] Les objectifs d'accessibilite s'appuient sur un standard (WCAG/RG2AA)

#### C15. Concevoir le cadre technique

- [x] Les specs techniques couvrent architecture, dependances, environnement
- [x] Les choix techniques favorisent l'eco-responsabilite (Green IT)
- [x] Les flux de donnees sont representes (Diagramme de flux)
- [x] Une Preuve de Concept (POC) est realisee
- [x] La POC est accessible et fonctionnelle en pre-prod
- [x] La conclusion de la POC permet de decider de la poursuite du projet

### A7. Developper interfaces et fonctionnalites

#### C16. Coordonner la realisation (Agile)

- [x] Les cycles, roles et rituels Agile sont respectes
- [x] Les outils de pilotage (Kanban, Backlog) sont disponibles et a jour
- [x] Les objectifs des rituels sont partages
- [x] Les elements de pilotage sont accessibles a tous

#### C17. Developper composants et interfaces

- [x] L'environnement de dev respecte les specs
- [x] Les interfaces respectent les maquettes
- [x] Les comportements (validation, navigation) respectent les specs
- [x] Les composants metier fonctionnent comme prevu
- [x] La gestion des droits/acces est developpee
- [x] Les flux de donnees sont integres selon les specs
- [x] Les developpements respectent l'eco-conception (eco-index / Green IT)
- [x] Le Top 10 OWASP est implemente
- [x] Des tests unitaires/integration couvrent le metier et les acces
- [x] Sources versionnees sur Git
- [x] Documentation technique (install, arch, tests) redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite (Valentin Haüy / Microsoft)

### A8. Tests et Controle

#### C18. Automatiser les tests (CI)

- [x] L'outil de CI est coherent avec la stack
- [x] La chaine integre toutes les etapes prealables (build)
- [x] La chaine execute les tests automatiquement
- [x] Config versionnee sur Git
- [x] Documentation de la CI (install/config/test) redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite (Valentin Haüy / Microsoft)

#### C19. Livraison continue (CD)

- [x] Les fichiers de config CD sont executes par le systeme
- [x] Les etapes de packaging (build, docker, minification) sont integrees
- [x] L'etape de livraison (ex: Deploy ou Pull Request) est integree
- [x] Sources versionnees sur Git
- [x] Documentation de la CD (toutes etapes/declencheurs) redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite

---

## BLOC E5 : Maintenance et Monitoring

### A9. Maintien en condition operationnelle

#### C20. Surveiller l'application (Monitoring App)

- [x] La documentation liste les metriques, seuils et valeurs d'alerte
- [x] La documentation justifie le choix des outils
- [x] Les outils (collecteurs, logs, dashboard) sont installes et operationnels (minima en local)
- [x] Les regles de journalisation (Logs) sont integrees au code source, en fonction des metriques a surveiller
- [x] Les alertes sont configurees et fonctionnelles selon les seuils
- [x] Le monitoring respecte les normes de gestion des donnees personnelles (RGPD)
- [x] Documentation d'installation du monitoring redigee
- [x] Documentation communiquee dans un format respectant les recommandations d'accessibilite

#### C21. Resoudre les incidents techniques

- [x] La cause du probleme est identifiee
- [x] Le probleme est reproduit en environnement de dev
- [x] La procedure de debogage est documentee
- [x] La solution documentee explicite chaque etape de resolution
- [x] La solution (fix) est versionnee sur Git (ex: Merge Request)

---

## Checklist Ressources & Stack Technique

### 1. Gestion de Projet & Methodologie

- [x] Methode Agile : Scrum
- [x] Outil de gestion de tickets/Backlog : Github Projects
- [x] Outil de communication : Discord
- [x] Outil de wireframing/maquettage : Penpot
- [x] Outil de modelisation donnees/flux : Looping

### 2. Developpement & Code

- [x] Gestionnaire de version (VCS) : Git
- [x] Plateforme de depot distant : GitHub
- [x] IDE / Editeur de code : VSCode
- [x] OS : Windows 11 Pro (PC Fixe et Portable)
- [x] CPU : AMD Ryzen 9 7950X (PC Fixe) et AMD Ryzen AI 9 HX 370 avec NPU inclus (PC Portable)
- [x] RAM : 32 Gb DDR5 (PC Fixe et Portable)
- [x] GPU : AMD Radeon RX 7900 XTX 24 Go (PC Fixe) et chipset graphique inclus dans le CPU (PC Portable)
- [x] Terminal : Powershell (version la plus recente)
- [x] Gestionnaire de paquets/projet Python : uv (CLI Astral)
- [x] Generateur de documentation : Sphinx
- [x] Documentation en Markdown : MyST-Parser (extension Sphinx)
- [x] Theme de documentation : Furo (theme Sphinx)

### 3. Data (Bloc E1)

- [x] Langage de script de collecte : Python
- [x] Librairie de requetes HTTP : HTTPX
- [x] Framework de crawl/scraping : Scrapy
- [x] Scraping de sites dynamiques : Playwright
- [x] Integration navigateur dans le crawler : scrapy-playwright
- [x] SGBD Relationnel : PostgreSQL
- [x] Systeme Big Data : Apache Spark (Traitement) + MinIO (Stockage Objet)
- [x] ORM SQL : SQLAlchemy ORM 2.0 (Async)

### 4. Intelligence Artificielle (Blocs E2 & E3)

- [x] Source de Veille IA : Inoreader + Task deep search auto en mode weekly avec Perplexity Pro (abonnement)
- [x] Service IA Pre-existant (SaaS/PaaS) : HuggingFace Serverless Inference API
- [x] Modele IA (Code Custom) : Scikit-learn, PyTorch, Hugging Face + AMD ROCm 7.1 et 7.2 (version native Windows)
- [x] Librairie de test IA : Deepchecks
- [x] Outil de MLOps / Monitoring IA : MLFlow + Loguru (suivi experiences et logs) + Prometheus + Grafana + Loki (monitoring et logs avances pour production/modeles IA)
- [x] Outil de versionning de donnees : DVC (Data Version Control)

### 5. Backend & API

- [x] Langage Backend : Python
- [x] Framework API REST : FastAPI
- [x] Outil de documentation API : Swagger UI + ReDoc
- [x] Librairie de securite/Auth : FastAPI Users (JWT bcrypt)
- [x] Librairies et outils utiles : Ruff, Loguru

### 6. Frontend (Application cliente)

- [x] Framework Frontend : React 19 via build tool Vite 8
- [x] Librairie UI / CSS : Shadcn/UI + Tailwind CSS v4
- [x] Outils d'accessibilite (Test) : axe-core via Playwright dans pipeline CI/CD

### 7. DevOps & Infrastructure (Blocs E4 & E5)

- [x] Plateforme CI/CD : GitHub Actions (ci.yml + cd.yml)
- [x] Solution de Conteneurisation : Docker, Docker-compose (multi-stage builds)
- [x] Outil de Monitoring Applicatif : Prometheus + Grafana + Loki (alertes configurees)
- [x] Hebergement / Deploiement (Pre-prod) : Render (Blueprint render.yaml)
