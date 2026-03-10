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
- [ ] Le script est versionne et accessible sur un depot Git distant

#### C2. Developper des requetes SQL d'extraction

- [x] Ecrire les requetes SQL pour extraire les donnees de la base de donnees relationnelle
- [x] Ecrire les requetes pour extraire les donnees du systeme Big Data
- [x] Les requetes sont fonctionnelles et extraient exactement les donnees visees
- [ ] Documenter les requetes (choix de selections, filtrages, jointures)
- [ ] La documentation explicite les optimisations appliquees aux requetes

#### C3. Agregation et nettoyage des donnees

- [x] Rediger les specifications techniques pour l'agregation
- [x] Programmer l'agregation des donnees de toutes les sources en un jeu unique
- [x] Programmer l'identification des entrees corrompues (partielles/manquantes)
- [x] Programmer la suppression des entrees corrompues
- [x] Programmer l'identification des formats non normalises
- [x] Programmer l'homogeneisation des formats (dates, unites, etc.)
- [x] Le script produit un jeu de donnees final unique, nettoye et normalise
- [ ] Le script est versionne sur Git
- [ ] La documentation du script est complete (dependances, logique algo, choix de nettoyage)

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
- [ ] Le script d'import est versionne et documente (dependances, commandes)

#### C5. Developper une API de mise a disposition (REST)

- [ ] Rediger les specifications techniques de l'API et de l'acces aux donnees
- [ ] Configurer les acces aux donnees depuis le serveur API
- [ ] Developper la reception et la validation des requetes client
- [ ] Developper les requetes BDD declenchees par l'API
- [ ] Developper les reponses de l'API au client
- [ ] Developper les regles d'autorisation/acces aux endpoints
- [ ] Securiser l'API (ex: Top 10 OWASP)
- [ ] L'API est fonctionnelle (acces et mise a disposition des donnees)
- [ ] La documentation technique couvre tous les endpoints
- [ ] La documentation couvre l'authentification/autorisation
- [ ] La documentation respecte les standards (ex: OpenAPI/Swagger)

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
- [ ] Rediger des syntheses des informations collectees
- [ ] Communiquer les syntheses aux parties prenantes (format accessible)

#### C7. Identifier des services IA (Benchmark)

- [ ] Reformuler le besoin (problematique technique/fonctionnelle)
- [ ] Identifier les contraintes (moyens, techniques, ops)
- [ ] Realiser un benchmark des services existants (etudies vs non etudies)
- [ ] Expliciter les raisons d'exclusion des services
- [ ] Le benchmark detaille l'adequation fonctionnelle
- [ ] Le benchmark detaille la demarche eco-responsable des services
- [ ] Le benchmark detaille les contraintes techniques/pre-requis
- [ ] Rediger des conclusions preconisant une ou plusieurs solutions

#### C8. Parametrer un service IA

- [ ] Creer l'environnement d'execution (Compte SaaS, VPS, etc.)
- [ ] Installer/configurer les dependances (SDK, outils)
- [ ] Gerer les acces (comptes, groupes, droits)
- [ ] Installer/configurer les outils de monitoring du service
- [ ] Le service est accessible et authentifie
- [ ] Le service repond aux besoins fonctionnels
- [ ] Le monitoring est operationnel
- [ ] La documentation technique couvre installation, acces, dependances
- [ ] La documentation est accessible (normes handicap)

---

## BLOC E3 : Deployer un modele d'IA (MLOps)

### A4. Realiser l'integration d'un modele IA

#### C9. Developper une API exposant un modele IA

- [ ] Analyser les specs fonctionnelles/techniques
- [ ] Concevoir l'architecture de l'API (endpoints, regles)
- [ ] Choisir les outils/langages
- [ ] Installer l'environnement de developpement
- [ ] L'API verifie et transforme les parametres envoyes par le client
- [ ] L'API execute le modele a partir de la requete
- [ ] L'API renvoie le resultat au client
- [ ] Developper l'authentification/autorisation
- [ ] Securiser l'API (OWASP)
- [ ] Developper des tests d'integration pour les endpoints
- [ ] Les tests passent sans bug
- [ ] Le code est versionne sur Git distant
- [ ] La documentation (OpenAPI) est redigee et accessible

#### C10. Integrer l'API IA dans une application

- [ ] Installer l'environnement de l'application cliente
- [ ] Programmer l'authentification vers l'API IA
- [ ] Programmer la communication avec les endpoints
- [ ] Integrer les adaptations d'interface necessaires
- [ ] Tester l'accessibilite des interfaces modifiees
- [ ] Developper des tests d'integration (cote app) sur le perimetre API
- [ ] Les tests s'executent en totalite sans erreur
- [ ] Le code est versionne sur le Git de l'application

### A5. Deploiement MLOps

#### C11. Monitorer le modele IA

- [ ] Lister les metriques a monitorer (perf modele, stabilite data, sante systeme)
- [ ] Definir les declencheurs de reentrainement
- [ ] Choisir l'outil de monitoring et de consolidation
- [ ] Integrer les collecteurs de donnees
- [ ] Integrer l'outil de restitution (Dashboard type Grafana/Streamlit)
- [ ] Configurer les alertes (email, notif)
- [ ] L'outil de restitution respecte les enjeux d'accessibilite
- [ ] La chaine de monitoring est testee et fonctionnelle
- [ ] Code versionne sur Git
- [ ] Documentation technique (install/maintenance) et utilisateur redigee

#### C12. Tests automatises du modele

- [ ] Definir le perimetre des tests (format data, entrainement, eval)
- [ ] Choisir les outils de test (ex: unittest, pytest)
- [ ] Configurer l'environnement de test
- [ ] Integrer les tests (assertions, mocks, fixtures)
- [ ] Les tests couvrent le perimetre defini
- [ ] Code et Donnees (si possible DVC) versionnes
- [ ] Documentation technique des tests redigee

#### C13. Chaine de livraison continue (CI/CD pour IA)

- [ ] Definir etapes, taches et declencheurs (Pipeline)
- [ ] Parametrer la chaine (variables d'env, versions)
- [ ] Integrer l'etape de test des donnees
- [ ] Integrer l'etape de test/entrainement/validation du modele
- [ ] Integrer la generation de rapports (matrix confusion, accuracy) dans la livraison
- [ ] Integrer l'etape de livraison (ex: Pull Request automatique)
- [ ] Les fichiers de config CI/CD sont reconnus et executes
- [ ] Code versionne sur Git
- [ ] Documentation de la chaine CI/CD redigee et accessible

---

## BLOC E4 : Realiser une application (Dev App)

### A6. Concevoir l'application

#### C14. Analyser le besoin

- [x] La modelisation des donnees respecte un formalisme (Merise/Entite-Relation)
- [ ] La modelisation des parcours utilisateurs respecte un formalisme (Wireframes/Schema)
- [ ] Les User Stories couvrent contexte, scenarios et criteres de validation
- [ ] Les objectifs d'accessibilite sont integres aux criteres d'acceptation (User Stories)
- [ ] Les objectifs d'accessibilite s'appuient sur un standard (WCAG/RG2AA)

#### C15. Concevoir le cadre technique

- [ ] Les specs techniques couvrent architecture, dependances, environnement
- [ ] Les choix techniques favorisent l'eco-responsabilite (Green IT)
- [ ] Les flux de donnees sont representes (Diagramme de flux)
- [ ] Une Preuve de Concept (POC) est realisee
- [ ] La POC est accessible et fonctionnelle en pre-prod
- [ ] La conclusion de la POC permet de decider de la poursuite du projet

### A7. Developper interfaces et fonctionnalites

#### C16. Coordonner la realisation (Agile)

- [ ] Les cycles, roles et rituels Agile sont respectes
- [ ] Les outils de pilotage (Kanban, Backlog) sont disponibles et a jour
- [ ] Les objectifs des rituels sont partages
- [ ] Les elements de pilotage sont accessibles a tous

#### C17. Developper composants et interfaces

- [ ] L'environnement de dev respecte les specs
- [ ] Les interfaces respectent les maquettes
- [ ] Les comportements (validation, navigation) respectent les specs
- [ ] Les composants metier fonctionnent comme prevu
- [ ] La gestion des droits/acces est developpee
- [ ] Les developpements respectent l'eco-conception
- [ ] Le Top 10 OWASP est implemente
- [ ] Des tests unitaires/integration couvrent le metier et les acces
- [ ] Sources versionnees sur Git
- [ ] Documentation technique (install, arch, tests) redigee et accessible

### A8. Tests et Controle

#### C18. Automatiser les tests (CI)

- [ ] L'outil de CI est coherent avec la stack
- [ ] La chaine integre toutes les etapes prealables (build)
- [ ] La chaine execute les tests automatiquement
- [ ] Config versionnee sur Git
- [ ] Documentation de la CI (install/config/test) redigee et accessible

#### C19. Livraison continue (CD)

- [ ] Les fichiers de config CD sont executes par le systeme
- [ ] Les etapes de packaging (build, docker, minification) sont integrees
- [ ] L'etape de livraison (ex: Deploy ou Pull Request) est integree
- [ ] Sources versionnees sur Git
- [ ] Documentation de la CD (toutes etapes/declencheurs) redigee et accessible

---

## BLOC E5 : Maintenance et Monitoring

### A9. Maintien en condition operationnelle

#### C20. Surveiller l'application (Monitoring App)

- [ ] La documentation liste les metriques, seuils et valeurs d'alerte
- [ ] La documentation justifie le choix des outils
- [ ] Les outils (collecteurs, logs, dashboard) sont installes et operationnels (minima en local)
- [ ] Les regles de journalisation (Logs) sont integrees au code source
- [ ] Les alertes sont configurees et fonctionnelles selon les seuils
- [ ] Documentation d'installation du monitoring redigee et accessible

#### C21. Resoudre les incidents techniques

- [ ] La cause du probleme est identifiee
- [ ] Le probleme est reproduit en environnement de dev
- [ ] La procedure de debogage est documentee
- [ ] La solution documentee explicite chaque etape de resolution
- [ ] La solution (fix) est versionnee sur Git (ex: Merge Request)

---

## Checklist Ressources & Stack Technique

### 1. Gestion de Projet & Methodologie

- [ ] Methode Agile : Scrum
- [ ] Outil de gestion de tickets/Backlog : Github Projects
- [ ] Outil de communication : Discord
- [ ] Outil de wireframing/maquettage : Penpot
- [x] Outil de modelisation donnees/flux : Looping

### 2. Developpement & Code

- [ ] Gestionnaire de version (VCS) : Git
- [ ] Plateforme de depot distant : GitHub
- [ ] IDE / Editeur de code : VSCode
- [ ] OS : Windows 11 Pro (PC Fixe et Portable)
- [ ] CPU : AMD Ryzen 9 7950X (PC Fixe) et AMD Ryzen AI 9 HX 370 avec NPU inclus (PC Portable)
- [ ] RAM : 32 Gb DDR5 (PC Fixe et Portable)
- [ ] GPU : AMD Radeon RX 7900 XTX 24 Go (PC Fixe) et chipset graphique inclus dans le CPU (PC Portable)
- [ ] Terminal : Powershell (version la plus recente)
- [ ] Gestionnaire de paquets/projet Python : uv (CLI Astral)
- [ ] Generateur de documentation : Sphinx
- [ ] Documentation en Markdown : MyST-Parser (extension Sphinx)
- [ ] Theme de documentation : Furo (theme Sphinx)

### 3. Data (Bloc E1)

- [ ] Langage de script de collecte : Python
- [ ] Librairie de requetes HTTP : HTTPX
- [ ] Framework de crawl/scraping : Scrapy
- [ ] Scraping de sites dynamiques : Playwright
- [ ] Integration navigateur dans le crawler : scrapy-playwright
- [ ] SGBD Relationnel : PostgreSQL
- [ ] Systeme Big Data : Apache Spark (Traitement) + MinIO (Stockage Objet)
- [ ] ORM SQL : SQLAlchemy ORM 2.0 (Async)

### 4. Intelligence Artificielle (Blocs E2 & E3)

- [ ] Source de Veille IA : Inoreader + Task deep search auto en mode weekly avec Perplexity Pro (abonnement)
- [ ] Service IA Pre-existant (SaaS/PaaS) : HuggingFace Serverless Inference API
- [ ] Modele IA (Code Custom) : Scikit-learn, PyTorch, TensorFlow, Hugging Face + AMD ROCm 7.1 et 7.2 (version native Windows)
- [ ] Librairie de test IA : Deepchecks
- [ ] Outil de MLOps / Monitoring IA : MLFlow + Loguru (suivi experiences et logs) + Prometheus + Grafana + Loki (monitoring et logs avances pour production/modeles IA)
- [ ] Outil de versionning de donnees : DVC (Data Version Control)

### 5. Backend & API

- [ ] Langage Backend : Python
- [ ] Framework API REST : FastAPI
- [ ] Outil de documentation API : Swagger UI + ReDoc
- [ ] Librairie de securite/Auth : FastAPI Users
- [ ] Librairies et outils utiles : Ruff, Loguru

### 6. Frontend (Application cliente)

- [ ] Framework Frontend : React via build tool Vite
- [ ] Librairie UI / CSS : Shadcn/UI
- [ ] Outils d'accessibilite (Test) : axe-core via Playwright dans pipeline CI/CD

### 7. DevOps & Infrastructure (Blocs E4 & E5)

- [ ] Plateforme CI/CD : GitHub Actions
- [ ] Solution de Conteneurisation : Docker, Docker-compose
- [ ] Outil de Monitoring Applicatif : Prometheus + Grafana + Loki
- [ ] Hebergement / Deploiement (Pre-prod) : Render
