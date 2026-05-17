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
- [x] Modele IA (Code Custom) : Scikit-learn, PyTorch, Hugging Face + AMD ROCm 7.2.1 stable (wheels-only, migration 7.1 MSI + 7.2 pre-release -> 7.2.1 le 2026-04-18)
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

---

## BONUS : Optimisation Demonstration & Refonte Modele (Avril-Mai 2026)

> **Objectif** : Ameliorer la precision du classifieur Green IT pour une demonstration credible au jury de soutenance.
>
> **Contexte** : Toutes les competences C1-C21 sont deja validees. Cette section bonus est une amelioration qualitative qui RENFORCE les competences existantes (C1, C3, C7, C11, C12) sans en invalider ni en ajouter aucune.
>
> **Ordre d'execution obligatoire** : B1 -> B2 -> B3 -> B4 -> B5 (B5 optionnel selon temps).
>
> **Cible chiffree** : passer de 17 articles Green IT (0.3% du dataset) a 200-500 (3-8% du dataset), et atteindre un MCC > 0.75 stable (ecart-type < 0.05) sur le K-fold.

---

### B1. Mise a Jour Infrastructure ROCm

#### B1.1 Verification de la version disponible

- [x] Verifier la version actuelle installee : HIP SDK 7.1.51803-d3a86bd04 + wheels torch 7.2 pre-release (rocmsdk20260116)
- [x] Consulter la page officielle AMD pour la derniere version stable Windows :
  - https://rocm.docs.amd.com/projects/install-on-windows/en/latest/install/quick-start.html
  - https://www.amd.com/en/developer/resources/rocm-hub.html
- [x] Consulter la matrice de compatibilite PyTorch + ROCm : wheels 2.9.1+rocm7.2.1 disponibles sur repo.radeon.com/rocm/windows/rocm-rel-7.2.1/
- [x] Version 7.2.1 stable confirmee -> passage a B1.2. Release notes examinees : fixes de stabilite HIP positifs, known issue hipBLASLt potentielle (mesuree non impactante en B1.6).

#### B1.2 Sauvegarde de l'environnement actuel

- [x] Creer un snapshot des versions actuelles : `uv pip list > requirements.snapshot.20260418.txt` (357 packages)
- [x] Sauvegarder `pyproject.toml` : `cp pyproject.toml pyproject.toml.20260418.backup`
- [x] Sauvegarder `uv.lock` : `cp uv.lock uv.lock.20260418.backup`
- [x] Noter le nom exact des wheels torch+torchvision+torchaudio actuellement installes (pour rollback eventuel) : torch=2.9.1+rocmsdk20260116, torchvision=0.24.1+rocmsdk20260116, torchaudio=2.9.1+rocmsdk20260116

#### B1.3 Desinstallation propre de ROCm 7.1/7.2

- [x] Fermer tous les programmes utilisant le GPU (MLflow UI, scripts Python, etc.)
- [x] Lancer le AMD HIP SDK Uninstaller pour ROCm 7.1 (via script PowerShell Admin msiexec /x)
- [x] Lancer le AMD HIP SDK Uninstaller pour ROCm 7.2 (n/a : 7.2 n'etait pas installe en systeme)
- [x] Verifier la suppression : registre HKLM et dossier `C:/Program Files/AMD/ROCm/7.1/` vides
- [x] Reboot du PC apres desinstallation (effectue)
- [x] Validation post-reboot : variables HIP_PATH et HIP_PATH_71 supprimees au niveau Machine, PATH propre

#### B1.4 Installation de la nouvelle version [OBSOLETE - ROCm 7.2.1 wheels-only]

- [x] Decouverte 2026-04-18 : AMD a pivote ROCm 7.2.1 Windows vers une distribution **wheels-only** sur `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/`. Aucun installeur MSI n'est publie, uniquement les wheels `rocm_sdk_core/devel/libraries_custom` qui contiennent le runtime HIP self-contained.
- [x] Test de validation post-desinstallation MSI : `torch.cuda.is_available() == True` avec les anciens wheels -> confirme que les wheels sont self-contained et rendent le MSI inutile.
- [x] **Conclusion : B1.4 rendu obsolete par la distribution wheels-only.** Toute la migration se joue desormais dans `pyproject.toml` (B1.5). Pas de reboot supplementaire.

#### B1.5 Mise a jour des dependances PyTorch

- [x] Identifier les wheels compatibles via la doc officielle AMD : torch-2.9.1+rocm7.2.1, torchvision-0.24.1+rocm7.2.1, torchaudio-2.9.1+rocm7.2.1 sur `rocm-rel-7.2.1/`
- [x] Mettre a jour `pyproject.toml` :
  - [x] Section `[tool.uv.sources]` : 3 URLs modifiees (rocm-rel-7.2/ -> rocm-rel-7.2.1/, rocmsdk20260116 -> rocm7.2.1)
  - [x] `find-links` idem + commentaire migration date
- [x] Reinstaller : `uv sync --reinstall-package torch torchvision torchaudio rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-custom` (~1,9 Go telecharge, 7 packages migres)
- [x] Verification de base : `torch.__version__ == '2.9.1+rocm7.2.1'`, `torch.cuda.is_available() == True`, `torch.cuda.get_device_name(0) == 'AMD Radeon RX 7900 XTX'`

#### B1.6 Tests de validation

- [x] Test 1 : PyTorch + matmul 1024x1024 GPU OK (44 MB VRAM, device capability (11,0) gfx1100)
- [x] Test 2 : Inference Qwen2.5-1.5B-Instruct fallback lightweight - 23,8 tok/s en bf16 sur GPU
- [x] Test 3 : Inference Qwen3-4B (base du modele de production) - charge 8,1 s, 7,49 GB VRAM, 28,7 tok/s en bf16
- [x] Test 4 : LocalQwenClient dispatcher + Qwen2.5-3B fallback n1 - charge + generation asyncio en 14,5 s (5,96 GB VRAM)
- [x] Aucun crash, aucun freeze, aucun warning hipBLASLt : pas de regression vs ROCm 7.1
- [x] Test entrainement sustained load : reporte au re-entrainement effectif du modele de production (K-fold CV servira de validation finale)

#### B1.7 Documentation

- [x] Creer `docs/PROCEDURE_MAJ_ROCM.md` documentant la procedure complete (wheels-only, pas de MSI)
- [x] Mettre a jour la documentation interne (section "Tech Stack") : ROCm 7.2.1 stable
- [x] Mettre a jour `docs/PLAN_ETAPES.md` section 1.1 + section 7.1 : nouvelle version + migration terminee

---

### B2. Enrichissement Dataset Green IT

> **Contrainte format absolue** : tout article ingere doit fournir soit `(titre + abstract)` soit `(titre + resume genere via classification_summarizer)`.
> Le resume de classification reste plafonne a `CLASSIFICATION_MAX_TOKENS=450` (cf. documentation interne).
> Toutes les sources doivent rester dans les 5 categories C1 deja validees (REST/JSON, scraping, fichiers, BDD, Big Data).

#### B2.1 Phase de recherche & validation prealable des sources (TERMINEE 2026-04-19)

> **Note 2026-04-19** : le dataset Hugging Face a ete ecarte apres validation prealable. `climatebert/climate_detection` n'a ni titre ni URL, utilise des paragraphes corporate hors scope tech, et ses labels "climat general" corrompraient notre ground truth Green IT. Les 6 autres sources fournissent largement le volume cible (5 000-8 000 articles attendus).

##### Source 1 : arXiv API (REST/JSON) - GO

- [x] Identifier les categories arXiv pertinentes : `cs.*`, `eess.*`, `stat.ML` (filtrage post-fetch pour souplesse)
- [x] Construire la liste de queries Green IT (9 queries ciblees < 200 resultats chacune) :
  - "green computing", "sustainable AI", "green AI", "carbon-aware computing"
  - "energy-efficient ML", "green software engineering", "low-power neural network"
  - "data center sustainability", "sustainable computing"
  - *(Ecartees : "efficient inference" 1499 resultats et "model compression" 1585, trop larges et dilueraient le signal)*
- [x] Test pilote (carbon-aware computing) : 13 articles, Atom XML parse OK via feedparser
- [x] Verifier presence systematique du champ `<summary>` : OK, 150-300 mots par article
- [x] Volume estime : 300-1 500 articles uniques (apres dedup par arxiv_id sans version)

##### Source 2 : Crossref API (REST/JSON) - GO

- [x] URL de base : `https://api.crossref.org/works`
- [x] Queries via `query.title` (meilleure precision que `query.bibliographic` qui explose a 300k+ resultats)
- [x] Filtre `has-abstract:true,from-pub-date:2020,type:journal-article` : necessaire car ~40% des entrees Crossref n'ont pas d'abstract
- [x] Test pilote (green computing) : 198 articles, JATS stripping OK, abstracts 500-2000 chars
- [x] Polite Pool via `mailto=karn1zc@gmail.com` dans le User-Agent (setting `CROSSREF_MAILTO`)

##### Source 3 : The Guardian (extension) - GO

- [x] Sections validees :
  - `environment` (5 932 articles 2024+, thematiques climat/biodiversite/ecologie)
  - `technology` (2 674 articles 2024+, innovations tech incl. Green IT)
- [x] Sub-sections `technology/green-computing`, `business/green-business`, `sustainable-business` : **n'existent pas** cote API Guardian (verifie via endpoint `/sections`). Utilisation du filtre `section=` direct.
- [x] Champs `fields=bodyText,trailText,byline,...` complets (8000+ chars)
- [x] Volume additionnel estime : +2 000-5 000 articles Guardian (environment + technology)

##### Sources 4-7 : Scraping (4 sites) - GO

- [x] **GreenIT.fr** : pas de robots.txt (convention = allow all), sitemap Yoast avec 1 001 posts, WordPress, URL `/YYYY/MM/DD/slug/`, HTML statique, 5 000+ chars/article, **100% Green IT**
- [x] **Greensoftware.foundation** : robots.txt permissif (content-signal explicatif Cloudflare sans regle bloquante), 170 articles (17 pages × 10), HTML statique, 8 500 chars/article, **100% Green IT**
- [x] **Sustainablewebdesign.org** : robots.txt `User-agent: * / Disallow:` (= all allowed), sitemap Yoast, 131 items (50 posts + 81 guidelines), HTML statique, 8 500+ chars
- [x] **Climateaction.tech** : robots.txt `Disallow: /wp-admin/` seul, sitemap_index.xml OK, 71 posts, WordPress, HTML statique, 4 500 chars/article, 100% tech+climat

##### Source 8 : Dataset Hugging Face (ECARTEE)

- [x] Candidats identifies : `climatebert/climate_detection` (1 700 paragraphes), `climatebert/environmental_claims` (2 647 phrases), `tdiggelm/climate_fever` (1 535 claims)
- [x] **DECISION : Ecarte du plan B2**. Format incompatible (ni titre ni URL), labels "climat general" hors scope Green IT specifique, domaine corporate/fact-checking different des articles tech. Utilisable eventuellement comme corpus d'evaluation robustesse dans une phase ulterieure.
- [x] Documentation de la decision d'ecart : section B2.1 ci-dessus + note dans `docs/PLAN_ETAPES.md` section 7.2 (2026-04-19).

##### Validation globale prealable

- [x] Volume total apres collecte (6 sources) : 5 000-8 000 nouveaux articles attendus (avant filtrage LLM judge)
- [x] Proportion Green IT attendue : 800-1 500+ confirmes (largement au-dessus de la cible 200-500)
- [x] Format titre + abstract/resume respecte sur toutes les sources retenues

#### B2.2 Implementation des collecteurs REST/JSON (TERMINEE 2026-04-19)

##### arXiv Collector - GO

- [x] `src/greentech/data/collectors/arxiv_collector.py` (411 lignes, ruff-clean)
- [x] Herite de `BaseCollector`, async via httpx, parsing Atom XML via feedparser
- [x] Pagination `start/max_results` avec `MAX_RESULTS_PER_KEYWORD=500` et `PAGE_SIZE=100`
- [x] Delai `MIN_DELAY_BETWEEN_REQUESTS=3.0` conforme aux bonnes pratiques arXiv
- [x] Filtrage post-fetch : categories `cs.*`, `eess.*`, `stat.ML` + abstract >= 100 chars
- [x] URL canonique sans version (`arxiv.org/abs/ID` sans v1/v2) pour dedup naturelle
- [x] Sauvegarde brute dans MinIO `raw-data/api/arxiv_api/<date>/<timestamp>.json`
- [x] Source BDD `source_name='arxiv_api'` distincte de `arXiv Dataset` (type=file historique)
- [x] Tests unitaires : `tests/unit/data/collectors/test_arxiv_collector.py` (9 tests)
- [x] Smoke test manuel : `carbon-aware computing` -> 13 articles pertinents
- [x] Commande CLI : `uv run python -m greentech.data.collectors.arxiv_collector`

##### Crossref Collector - GO

- [x] `src/greentech/data/collectors/crossref_collector.py` (pattern identique a arxiv)
- [x] Polite Pool via setting `CROSSREF_MAILTO` injecte dans le User-Agent (`base (mailto:...)`)
- [x] Filtre serveur `has-abstract:true,from-pub-date:2020,type:journal-article` + tri `relevance:desc`
- [x] Top 200 par query (MAX_RESULTS_PER_KEYWORD=200, PAGE_SIZE=200)
- [x] Strip JATS (`<jats:p>`, `<jats:sec>`, etc.) via regex + normalisation espaces
- [x] Extraction date multi-fallback : `published-print -> published-online -> issued -> created`
- [x] Types acceptes : `journal-article`, `proceedings-article` (livres, chapitres rejetes)
- [x] URL canonique via resolveur DOI : `https://doi.org/<DOI>`
- [x] Tests unitaires : `tests/unit/data/collectors/test_crossref_collector.py` (16 tests)
- [x] Smoke test manuel : `green computing` -> 198 articles (top 200 Crossref), JATS propre
- [x] Commande CLI : `uv run python -m greentech.data.collectors.crossref_collector`

##### Extension Guardian Collector - GO

- [x] Parametre `sections: list[str] | None` ajoute a `GuardianCollector.collect()`
- [x] `DEFAULT_GREEN_IT_SECTIONS = ('environment', 'technology')` utilise par defaut dans `run_guardian_collection()`
- [x] Mode 2 passes : 1 requete plein-texte + N requetes par section, avec dedup URL au niveau collecteur
- [x] Signatures `_fetch_articles_with_retry` et `_fetch_articles` etendues avec `section: str | None = None`
- [x] Backward compatible : sections vides = mode historique (plein-texte uniquement)
- [x] Tests unitaires : `tests/unit/data/collectors/test_guardian_collector.py` (5 tests)
- [x] Smoke test manuel : `sustainable AI` + section `environment` -> 98 articles uniques (50 + 48 dedoublonnes)

##### Migration BDD (search_config + sources)

- [x] Script SQL idempotent : `scripts/sql/migration_002_b2_sources_config.sql` applique
- [x] `search_config_type_source_check` etendu pour accepter `arxiv_api` et `crossref`
- [x] 2 nouvelles sources inserees : `arXiv API` (type=api), `Crossref` (type=api), `est_active=true`
- [x] 9 mots-cles `arxiv_api` inseres (voir B2.1)
- [x] 8 mots-cles `crossref` inseres (selection top-pertinence)
- [x] `scripts/sql/init.sql` aligne pour les nouveaux deploiements (schema CHECK + INSERT sources + INSERT search_config)

#### B2.3 Implementation des spiders Scrapy (4 sites) (TERMINEE 2026-04-19)

> Architecture : Scrapy + HTTP standard (sans Playwright). Les 4 sites sont en HTML statique, verifie en B2.1. Le choix evite l'overhead Chromium (~5x plus rapide, ~5x moins de RAM, conforme au positionnement Green IT du projet). Les hooks Playwright restent accessibles via la base class si un site ajoute du JS critique plus tard.

- [x] **Base class commune** : `src/greentech/data/collectors/spiders/base.py`
  - [x] `StaticArticleSpider` : discovery par sitemap XML OU pagination HTML
  - [x] Parsing des sitemaps (urlset + sitemapindex), filtre URL par regex (`article_url_pattern`)
  - [x] Extraction title/contenu/date/auteur via chaine de selecteurs CSS + fallback og:title + fallback trafilatura
  - [x] Extraction contenu inclut `<p>, <h2-5>, <li>, <blockquote>` avec texte des enfants `<a>/<em>/<strong>` (XPath `.//text()`)
  - [x] Filtrage `MIN_CONTENT_LENGTH=300`, `MIN_TITLE_LENGTH=5`, troncature `MAX_TITLE_LENGTH=480` (contrainte BDD 500 chars)
  - [x] Telemetrie par categorie d'erreur (missing_title, empty_content, http_error, parsing_error)
  - [x] Dedup global par URL via `_seen_urls` set
- [x] **Spider GreenIT.fr** : `src/greentech/data/collectors/spiders/greenit_fr_spider.py`
  - [x] 3 sitemaps WordPress (post-sitemap + post-sitemap2 + post-sitemap3), 1 001 posts total
  - [x] URL pattern `/YYYY/MM/DD/slug/`, langue FR
  - [x] `DOWNLOAD_DELAY = 2s`, `ROBOTSTXT_OBEY = True`
  - [x] Sauvegarde MinIO `raw-data/scraping/greenit_fr/<date>/`
- [x] **Spider Greensoftware.foundation** : `greensoftware_spider.py`
  - [x] Pagination HTML `/articles/` + `/articles/2` ... `/articles/17`
  - [x] Article link selector `a[href^="/articles/"]` + filtre regex (exclut pages paginated)
  - [x] 170 articles EN
- [x] **Spider Sustainablewebdesign.org** : `sustainable_web_spider.py`
  - [x] 2 sitemaps : `post-sitemap.xml` (50 posts) + `guidelines-sitemap.xml` (81 guidelines)
  - [x] 131 items EN, pas de filtre URL
- [x] **Spider Climateaction.tech** : `climate_action_tech_spider.py`
  - [x] `sitemap_index.xml` -> `post-sitemap.xml` (71 posts)
  - [x] URL pattern `/blog/slug/`, langue EN, theme Neve
- [x] **Optimisation re-run : pre-check BDD par pre-fetch (2026-04-19)**
  - [x] Module partage `src/greentech/data/collectors/url_precheck.py` : `load_known_urls(source_name)` via asyncpg direct (< 500 ms pour 50 k URLs), fallback silencieux sur set vide si BDD indispo
  - [x] Flag `skip_existing: bool = True` ajoute dans `StaticArticleSpider` (override par defaut actif via `-a skip_existing=false`)
  - [x] Filtrage applique dans `_parse_sitemap()` et `_parse_listing()` : URL deja en BDD -> skip avant scheduling Request (evite fetch HTTP + parsing)
  - [x] Compteur `_skipped_existing` pour telemetrie par spider
  - [x] Tests unitaires : 3 nouveaux tests pytest (filtrage effectif, default True, override kwarg). 23 tests verts total pour les spiders statiques.
- [x] **Extension du pre-check BDD a TOUS les collecteurs (2026-04-19)**
  - [x] **Dev.to** (gain ENORME) : check apres la liste, evite le fetch detail par article
  - [x] **TechCrunch scraping** (gain ENORME) : check apres le RSS, evite le fetch Playwright (5-10s/page)
  - [x] **arXiv API** (gain modeste) : check pendant le parsing, evite pollution MinIO
  - [x] **Crossref** (gain modeste) : check pendant le parsing, evite pollution MinIO
  - [x] **Guardian** (gain modeste) : check dans `_parse_articles`, evite ingestion de doublons
  - [x] **File ingester** (gain faible) : check pendant la lecture JSON Lines, evite pousse de doublons vers MinIO
  - [x] Tous les collecteurs exposent `skip_existing: bool = True` par defaut, desactivable explicitement
- [x] **Correctness du pre-check : normalisation URL + tests (2026-04-19)**
  - [x] `normalize_url()` applique : http -> https, host lowercase, trailing slash retire, whitespace trim
  - [x] `url_is_known(url, known_set)` : helper centralise utilise par tous les collecteurs, compare les formes normalisees des deux cotes
  - [x] `load_known_urls()` stocke les URLs deja normalisees pour une comparaison O(1) consistante
  - [x] **PROUVE : pas de faux positif** (URL reellement absente ne declenche jamais un skip, le set ne contient que des URLs issues de la BDD)
  - [x] **Faux negatifs elimines** pour les cas reels observes : arXiv Kaggle https vs API http (incoherence pre-existante non detectee par l'ancien pre-check exact), trailing slash, casse host
  - [x] Tests unitaires : 42 tests pytest dans `test_url_precheck.py` couvrant (a) idempotence de `normalize_url`, (b) absence de faux positifs (5 cas : empty set, host different, path different, casse path, query diff), (c) equivalences critiques detectees (5 cas), (d) conversion de flags CLI par `coerce_bool`. Total projet : 134 tests verts.
- [x] **Orchestrator** : `src/greentech/data/collectors/static_scraping_collector.py`
  - [x] Lance les 4 spiders en un unique `CrawlerProcess` Scrapy (partage Twisted reactor)
  - [x] Agrege les articles par spider, upload MinIO separe par site (tracabilite)
  - [x] Bilan global (articles par site + erreurs par categorie)
  - [x] Compatible interface `BaseCollector` + commande CLI `uv run python -m greentech.data.collectors.static_scraping_collector`
- [x] **Tests unitaires** : `tests/unit/data/collectors/test_static_spiders.py`
  - [x] 20 tests pytest (normalize_whitespace, config par site, sitemap parsing, extraction titre/contenu/date/auteur, parse_article valide/rejete, meta-test heritage)
  - [x] Tous verts, aucune regression sur le reste du projet (89 tests total)
- [x] **Migration BDD** : `scripts/sql/migration_003_b2_3_spiders.sql` appliquee
  - [x] 4 nouvelles sources inserees (`GreenIT.fr`, `Green Software Foundation`, `Sustainable Web Design`, `Climate Action Tech`)
  - [x] `scripts/sql/init.sql` aligne pour les deploiements futurs
- [x] **Smoke tests live** : 3 articles par site verifies (contenus 3 000-9 000 chars, langue FR/EN correcte, dedup URL OK)

#### B2.4 Ingestion Hugging Face dataset (ECARTE)

> Supprime du plan apres validation prealable (B2.1). Le dataset `climatebert/climate_detection` a un format incompatible (ni titre ni URL, paragraphes corporate hors scope tech) et des labels "climat general" qui corrompraient le ground truth Green IT. Les 7 autres sources fournissent largement le volume cible. Cette source pourrait eventuellement servir de corpus d'evaluation robustesse dans une phase ulterieure.
- [x] Documentation de la decision d'ecart : note consignee dans B2.1 (CHECKLIST) + `docs/PLAN_ETAPES.md` section 7.2 (2026-04-19).

#### B2.5 Mise a jour des metadata BDD (TERMINEE 2026-04-19)

- [x] Mettre a jour `scripts/sql/init.sql` ou creer une migration :
  - [x] Ajouter les 8 nouvelles sources dans la table `sources` (nom, type, URL, active=true) — 10 sources actives en BDD (arXiv Dataset, arXiv API, Crossref, The Guardian, Dev.to, TechCrunch Climate, GreenIT.fr, Green Software Foundation, Sustainable Web Design, Climate Action Tech) + NewsData.io conservee desactivee
  - [x] Ajouter les nouveaux mots-cles dans la table `search_config` — 79 mots-cles repartis : 36 api + 15 guardian + 9 arxiv_api + 8 crossref + 8 devto + 3 scraping
- [x] Executer la migration sur la base existante — `migration_002_b2_sources_config.sql` + `migration_003_b2_3_spiders.sql` appliquees, `scripts/sql/init.sql` aligne pour les deploiements futurs
- [x] Verifier via `uv run python -c "..."` ou requete SQL directe — 11 664 articles repartis sur 10 sources actives, 0 article sans source

#### B2.6 Lancement collecte massive (TERMINEE 2026-04-19)

> **Note d'implementation** : l'orchestration n'a pas ete materialisee dans un nouveau `scripts/collect_all_sources.py` dedie. Elle est integree directement dans `scripts/retrain_pipeline.py collect`, qui appelle sequentiellement tous les collecteurs. Ce choix garantit que la collecte utilise les memes flags (`skip_existing=True`, pre-check URL) que le reste du pipeline et evite un script redondant.

- [x] Orchestration de la collecte (implementee dans `scripts/retrain_pipeline.py collect`) :
  - [x] arxiv_collector
  - [x] crossref_collector
  - [x] guardian_collector (avec nouvelles sections `environment` + `technology`)
  - [x] devto_collector (existant)
  - [x] api_collector legacy (NewsData, desactive conformement au plan)
  - [x] Les 4 nouveaux spiders Scrapy (via `static_scraping_collector` en un seul `CrawlerProcess`)
  - [x] file_ingester (arXiv Kaggle, conserve pour retro-compatibilite ; dataset HF ecarte en B2.4)
- [x] Executer la collecte (idempotente : skip articles deja en BDD via URL unique normalisee) — pre-check BDD actif sur tous les collecteurs (cf. B2.3)
- [x] Mesurer le volume reellement collecte — 11 664 articles totaux en BDD : arXiv Dataset 4 957, GreenIT.fr 2 945, Crossref 1 499, The Guardian 1 252, arXiv API 382, Green Software Foundation 193, Dev.to 135, Sustainable Web Design 130, TechCrunch Climate 105, Climate Action Tech 66

#### B2.7 Nettoyage Spark sur le nouveau volume (TERMINEE 2026-04-19)

- [x] Verifier que `src/greentech/data/processors/spark_cleaner.py` tient le nouveau volume — generalise pour lire le prefixe MinIO `scraping/` complet (plus uniquement `scraping/techcrunch`) et pour supporter deux formats (TechCrunch `contenu_html` + 4 spiders `contenu`)
- [x] Adapter la conf memoire si besoin (driver memory, executor memory) — configuration existante suffisante, aucune OOM observee sur les 11 664 articles
- [x] Lancer le nettoyage : `uv run python -m greentech.data.processors.spark_cleaner` — execute dans le pipeline `retrain_pipeline.py clean`
- [x] Verifier que les articles trop courts (< seuil deja existant) sont bien filtres — 76 articles junk (tests API + preprints retractes) supprimes avant B2.8 ; filtres low-entropy/withdrawn/retracted actifs dans `classification_summarizer.py`
- [x] Verifier la deduplication — URL normalisee (http->https, host lowercase, trailing slash) utilisee comme cle de dedup, 42 tests unitaires dans `test_url_precheck.py`

#### B2.8 Generation des resumes de classification (TERMINEE 2026-04-19)

- [x] Pour les articles complets sans abstract (issus du scraping principalement) : generer le resume via `classification_summarizer`
- [x] Lancer : `uv run python scripts/generate_classification_summaries.py` — 5 977 articles traites sur 5-9h, 0 resume bidon grace aux filtres low-entropy + withdrawn/retracted
- [x] Verifier que tous les articles ont desormais soit `abstract` soit `resume` <= 450 tokens — `CLASSIFICATION_MAX_TOKENS=450` respecte via `classification_summarizer.py`
- [x] Requete SQL de validation : compter les articles sans `(abstract OR resume)` — 11 664 articles avec resume non-null, 0 article sans resume (verifie 2026-04-21)

#### B2.9 Re-classification du corpus etendu (TERMINE 2026-04-21 00:32)

> **Etat final** : pipeline complet `classify summarize-green export-golden` execute d'une traite en 4h38 (278 min). classify 3h10, summarize-green 1h28, export-golden 2 sec. Zero echec sur l'ensemble. Le refactor kill-safe (commit par batch de 50) n'a pas eu a etre sollicite — aucune interruption n'est survenue.

- [x] **Etage 1 - Pre-filtre keywords** : `uv run python scripts/auto_annotate_dataset.py` (sur tout le corpus, ancien + nouveau) — **TERMINE 2026-04-19 22:38**
  - [x] 11 667 articles annotes (100 %)
  - [x] 6 091 marques `NON_GREEN` (filtrage permissif par mots-cles)
  - [x] 4 531 marques `CANDIDATE` en attente du LLM judge (etage 2)
  - [x] Dispatcher HF -> Qwen local verifie : fallback sur quota epuise (HTTP 402)
- [x] **Etage 2 - LLM judge Qwen** : `uv run python scripts/classify_candidates.py` — **TERMINE 2026-04-20 23:04** en 3h10 sur les 4 528 candidats restants (91 batches de 50, kill-safe)
  - [x] Dispatcher HF -> local verifie (fallback Qwen2.5-3B sur RX 7900 XTX 24 Go VRAM, bf16)
  - [x] **Refactor kill-safe 2026-04-20** : commit par batch de 50 articles via `classify_all_candidates(batch_size=50)`. Perte max sur interruption = 50 verdicts (~10 min au rythme local ~5/min). Idempotent : `fetch_candidates` ne renvoie que les articles encore `est_green_it IS NULL`. La run precedente (non-batch) du 2026-04-19 22:38 tournait depuis 3h sans commit et a perdu ~1 045 verdicts au kill — c'est ce qui a motive le refactor.
  - [x] **Run 2026-04-20 19:54 -> 23:04** : 4 528 candidats traites d'une traite, 91/91 batches commites, 0 echec. Verdicts : 1 001 Green IT + 3 527 Non Green IT.
- [x] **Generation resumes Green IT** : `uv run python scripts/generate_green_summaries.py` (sur les confirmes uniquement) — **TERMINE 2026-04-21 00:32** en 1h28 sur 1 002 articles (1 001 nouveaux + 1 restant du 2026-04-20 01:50). 0 echec, ~8-9 sec/article via Qwen2.5-3B local.
- [x] **Export golden dataset** : `uv run python scripts/export_golden_dataset.py` — **TERMINE 2026-04-21 00:32** en 2 sec. `data/golden_dataset.csv` : 11 667 articles, 1 018 Green IT (8.73 %), 10 649 Non Green IT, 0 exclu.
- [x] Verifier les chiffres finaux :
  - [x] Total articles : **11 664 articles** (dans la fourchette 8 000-15 000) apres nettoyage linguistique
  - [x] Articles Green IT confirmes : **1 018 (8.73 %)** — cible initiale 200-500 pulverisee, ratio en haut de la fourchette 3-8 % (point de depart : 17 Green IT soit 0.3 %).
  - [x] Repartition linguistique (FR/EN/autre) loguee — dataset **bilingue EN/FR** apres nettoyage : EN 8 719 (74.75 %, 418 Green IT), FR 2 945 (25.25 %, 600 Green IT). Nettoyage 2026-04-21 01:46 : 10 articles `ng` (faux positif langdetect, contenu 100 % EN) normalises en `en`, 3 articles `de`/`ru` supprimes (volume trop faible pour exploitation, 2 de + 1 ru, tous Non Green IT). Asymetrie forte : le FR est 4.2x plus dense en Green IT que l'EN (lie a greenit.fr, 1 000 posts 100 % Green IT en FR). Implication B4 : `mdeberta-v3-base` (multilingue) requis, pas `deberta-v3-base` EN-only.

#### B2.10 Annotation manuelle (procedure detaillee)

> **Decision 2026-04-22 (option C)** : apres analyse, le score
> ``score_confiance`` est en realite **deja persiste** en BDD pour les
> 5 574 articles classifies par le LLM judge (il l'etait des le run
> du 2026-04-20 via `apply_verdicts`). Pas besoin de re-runner pour
> obtenir les scores. La seule information manquante est la
> justification textuelle (`verdict.raison`) : elle est desormais
> captee via la migration 004 + modification de `classify_candidates.py`
> pour le **prochain** run naturel, sans re-run force immediat.

- [x] **Etape 1 - Identifier les borderline** :
  - [x] Modifier `classify_candidates.py` pour logger le score brut du LLM judge (probabilite/confiance) (2026-04-22) — le score etait deja persiste dans `score_confiance` depuis le run du 2026-04-20. Modification supplementaire : ajout de la persistence de `verdict.raison` (justification textuelle) dans `raison_llm_judge` + `annotation_source='llm_judge'` pour tracabilite. Actif des le prochain run LLM (pas re-run force). Migration SQL `migration_004_annotation_manuelle.sql` appliquee (5 574 + 6 090 articles backfillee).
  - [x] Identifier les articles avec score entre 0.3 et 0.7 (zone d'incertitude) — 1 325 articles borderline identifies via index partiel `idx_articles_borderline_llm`. Tous classes Non Green IT (le LLM hesite a dire OUI et tranche NON).
- [x] **Etape 2 - Decision sur le volume** :
  - [x] Compter les borderline. Si < 50 : tout annoter manuellement. Si 50-200 : echantillon stratifie. Si > 200 : echantillon de 100 max. — 1 325 articles borderline, tous a examiner selon decision utilisateur 2026-04-22 (pas d'echantillonnage). Priorite a GreenIT.fr (969 borderline, site 100 % Green IT = potentiels faux negatifs concentres).
  - [x] Communiquer le volume a l'utilisateur pour validation avant lancement — fait le 2026-04-22 avec ventilation par source. Audit exhaustif valide.
- [x] **Etape 3 - Outil d'annotation** (2026-04-22) :
  - [x] Creer `scripts/manual_annotation_helper.py` :
    - [x] CLI interactif via `rich` (Console, Panel, Table, Prompt, Markdown)
    - [x] Pour chaque article : affiche titre + resume + URL + contenu tronque + raison LLM (si peuplee) + score + decision LLM actuelle
    - [x] Demande input : `g` (Green IT) / `n` (Non Green IT) / `s` (skip) / `o` (ouvrir URL dans navigateur) / `q` (quit)
    - [x] Sauvegarde decision en BDD avec `annotation_source='manual'`, `annotated_at=now()`, `annotated_by='KaRn1zC'` (configurable via `--by`). Preserve `score_confiance` et `raison_llm_judge` pour mesurer le taux de correction humaine a posteriori.
    - [x] Possibilite de reprendre une session interrompue : les articles `annotation_source='manual'` sont automatiquement exclus au prochain lancement. Filtres CLI : `--source`, `--score-min`, `--score-max`, `--limit`.
- [x] **Etape 4 - Documentation** (2026-04-22) :
  - [x] Creer `docs/ANNOTATION_MANUELLE.md` avec :
    - [x] Procedure step-by-step illustree (commandes CLI par cas d'usage, raccourcis clavier, reprise de session)
    - [x] Criteres de decision Green IT (rappel de la definition inclusive du prompt systeme LLM, alignement strict pour eviter toute derive)
    - [x] Exemples de cas limites avec leur classification correcte (6 exemples couvrant Crossref, Guardian, GreenIT.fr, arXiv, TechCrunch, Dev.to)
    - [x] Commandes CLI exactes (commandes usuelles, raccourcis clavier, reprise, annulation)
    - [x] Justification de la fenetre [0.3 ; 0.7] vs [0.3 ; 0.8] (distance a 0.5, volume annotable, strategie d'elargissement conditionnelle)
    - [x] Priorite par source (GreenIT.fr en tete : 969 borderline sur 1 325)
- [ ] **Etape 5 - Versioning** :
  - [ ] Re-export du golden dataset apres annotation manuelle — **A FAIRE apres audit**
  - [ ] Tag DVC du nouveau dataset versionne (`golden_dataset.csv.dvc`) — **A FAIRE apres audit**
  - [ ] Push vers MinIO via `uv run dvc push` — **A FAIRE apres audit**

#### B2.11 Mise a jour documentation

- [ ] Mettre a jour `docs/SPECIFICATIONS_DATA.md` :
  - [ ] Ajouter les 8 nouvelles sources avec leur description, format, frequence
  - [ ] Mettre a jour la liste des contraintes techniques
- [ ] Mettre a jour `docs/REGISTRE_RGPD.md` :
  - [ ] Verifier si nouvelles sources contiennent des donnees personnelles (auteurs, emails)
  - [ ] Documenter les nouvelles regles d'anonymisation si necessaire
- [ ] Mettre a jour la documentation interne (sections "Data" et "Commandes") avec les nouveaux collecteurs
- [ ] Mettre a jour `docs/PLAN_ETAPES.md` section 2.3 (Programmation Collecte) avec les nouveaux modules
- [ ] Documentation Sphinx complete : `cd docs && uv run sphinx-build -b html . _build/html`

---

### B3. Optimisation Pipeline d'Entrainement (PROTOCOLE UNIFIE 2026-04-21)

> **Protocole fige apres synthese de 3 agents de recherche** (imbalanced text classification 2024-2026 + LoRA Qwen3-4B + mDeBERTa fine-tuning bilingue). Les 4 decisions structurantes sont actees : stratification croisee `(langue x label)`, class_weight sur CrossEntropy, back-translation EN<->FR, calibration (temperature scaling + threshold tuning), ensemble K-fold K=5, 3 seeds par fold.

#### B3.1 Stratification croisee (langue x label)

> Priorite 1. Gain principal sur l'ecart-type MCC (cible sigma < 0.10).

- [x] Installer `iterative-stratification` via `uv add iterative-stratification` — `iterative-stratification>=0.1.9` dans `pyproject.toml:73`
- [x] Remplacer `StratifiedKFold(K=5, stratify=y)` par `MultilabelStratifiedKFold(K=5)` sur le vecteur compose `(langue, label)` (une seule colonne multi-label encodee one-hot) — import `training.py:1562`, usage ligne 1646
- [x] Verifier que chaque fold contient la distribution correcte :
  - [x] ~75 % EN / ~25 % FR — `_log_fold_split_stats` logue les compteurs EN/FR par fold (ligne 1935)
  - [x] ~8.73 % positifs globaux, avec ~4.80 % positifs parmi les EN et ~20.37 % parmi les FR — loguee via `n_val_green` / `n_train_green`
  - [x] Assert automatique avec tolerance 2 pts de pourcentage (2026-04-21) — `_check_fold_stratification()` compare les 5 ratios observes aux cibles `_STRATIFICATION_TARGET_RATIOS` (tolerance `_STRATIFICATION_TOLERANCE_PP = 0.02`). Mode hybride : warning loguru par defaut, `AssertionError` si `strict_stratification=True`. Flag CLI `--strict-stratification` expose par `retrain_pipeline.py`. Ratios observes logges dans MLflow (`fold_X_val_ratio_*`) pour audit post-entrainement.
- [x] Tests unitaires dans `tests/unit/test_stratification.py` (2026-04-21) — 14 tests pytest : coherence des constantes cibles (tolerance 2 pp, EN+FR=1, ordre FR>global>EN), logique `_check_fold_stratification` (conforme, juste en dessous, legerement au-dela, multiples deviations, mode strict, cles manquantes), calcul `_log_fold_split_stats` (5 cles, valeurs manuelles, val vide degenere, propagation `strict_stratification`). Tous verts sans mock IA.
- [x] Logger les stats de chaque fold dans MLflow (params : `fold_X_n_en`, `fold_X_n_fr`, `fold_X_ratio_green_fr`, etc.) — integre dans `train_with_unified_protocol`

#### B3.2 Loss ponderee (class_weight)

- [x] Remplacer `_oversample_minority` dans `src/greentech/ai/models/training.py` par passage du `class_weight` a la CrossEntropy :
  - [x] Calcul au demarrage de chaque fold : `class_weight = torch.tensor([1.0, N_neg_train / N_pos_train])` (~[1.0, 10.46] sur le full train, varie legerement par fold) — `compute_class_weight()` ligne 190
  - [x] Passer via `compute_loss_func` custom ou en sub-classant `Trainer.compute_loss` — `WeightedLossTrainer` ligne 140, override `compute_loss()` avec `torch.nn.CrossEntropyLoss(weight=weight)`
- [x] Supprimer l'oversampling x84 (overfits les 22 memes textes, inflate le train set) — `_oversample_minority` n'est plus appele dans `train_with_unified_protocol`
- [x] Tests unitaires : verifier que la loss converge sur un mini-dataset desequilibre synthetique (2026-04-21) — `tests/unit/ai/test_class_weight.py` (11 tests pytest). Couvre `compute_class_weight` (balanced, imbalanced 1:9, ratio production 1:10.46, degenere sans positif, input numpy, shape/dtype) et `WeightedLossTrainer.compute_loss` (amplification sur batch mixte hard-positifs, formule `sum(w_i*CE_i)/sum(w_i)` verifiee numeriquement, fallback CE standard sans class_weight, restauration `inputs["labels"]`, `return_outputs=True`). Stub Trainer via `__new__` pour eviter l'init HF complet, stub model pour mocker `outputs.logits`.
- [x] Run MLflow : tag explicite `loss_strategy=weighted_ce` pour tracabilite (2026-04-21) — tag ajoute dans l'`ExperimentConfig` de `train_with_unified_protocol` (ligne 1623). Tag complementaire `stratification=multilabel_langue_label` + tag `augmentation=opus-mt-backtranslation|none` pour tracabilite complete du protocole B3 dans MLflow.
- [x] **Focal Loss NON retenue** : les 3 agents convergent — Focal Loss degrade la calibration et n'apporte de gain qu'au-dela de 1:50 (notre ratio 1:10.5 est "modere"). Elle pourra etre testee en phase 2 si le MCC plafonne sous 0.75.

#### B3.3 Back-translation EN<->FR (opus-mt) (EXECUTEE 2026-04-21 04:00-04:13)

> Double le nombre de positifs (1 018 -> ~2 036), ratio effectif 1:10.5 -> ~1:5.25. Gain attendu +0.05 a +0.08 MCC.
>
> **Resultat reel** : 1 018 positifs originaux -> 1 686 positifs totaux (+ 668 variantes back-translation acceptees, ratio effectif ~1:6.3). Le filtre qualite cosine [0.85, 0.99] a rejete ~350 variantes (similarite trop faible ou trop haute).

- [x] Installer `sentence-transformers` (filtre qualite) via `uv add sentence-transformers` — `sentence-transformers>=5.4.1` dans `pyproject.toml:74`
- [x] Telecharger modeles : `Helsinki-NLP/opus-mt-en-fr` et `Helsinki-NLP/opus-mt-fr-en` (2x ~75M params, ~150 Mo chacun) — cache HF local, exploites sans erreur
- [x] Creer `src/greentech/data/processors/back_translator.py` :
  - [x] Classe `BackTranslator` avec methode `augment_positives(articles, target_similarity_range=(0.85, 0.99))` — ligne 139
  - [x] Pipeline : EN positif -> FR (opus-mt-en-fr) -> EN (opus-mt-fr-en) ; idem sens inverse
  - [x] Filtre qualite : similarite cosine `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (original, retraduit) hors [0.85, 0.99] = rejet
  - [x] N'appliquer que sur `resume` (150-220 mots). Titre inchange, reutilise tel quel.
  - [x] Variantes ont meme `id_source`, meme label, mais `uuid` different + flag `augmentation_source='opus-mt-backtranslation-simX.XXX'`
- [x] Creer `scripts/augment_positives.py` (CLI) :
  - [x] Lit `data/golden_dataset.csv`, filtre positifs
  - [x] Genere les variantes via `BackTranslator`
  - [x] Output : `data/golden_dataset_augmented.csv` (union des originaux + variantes acceptees) — 12 332 articles, 1 686 positifs, 140 valeurs distinctes de `augmentation_source` (score de similarite par variante)
  - [x] Logs MLflow : nombre genere, nombre accepte, nombre rejete (similarite hors bornes), temps total — runs `augment-back-translation-1776736802` (2026-04-21 04:00) et `augment-back-translation-1776737626` (2026-04-21 04:13)
- [x] **Regle d'or** : les variantes vont UNIQUEMENT dans le train split de chaque fold. Le val/test split n'est construit qu'a partir des articles originaux (jamais augmentes), sinon on biaise l'evaluation. Cette regle est implementee dans la logique K-fold : stratifier sur les originaux, puis augmenter chaque train split apres splitting — `_build_variant_index` (ligne 1875) + `_collect_variants_for_train` (ligne 1924) + split originaux-only dans `train_with_unified_protocol` (lignes 1587-1657)
- [x] Tests unitaires : verifier que val/test ne contient jamais de `augmentation_source` non-null (2026-04-22) — `tests/unit/ai/test_back_translator.py` (15 tests pytest). Couvre `BackTranslationStats` (acceptance_rate edge cases, to_dict MLflow keys), `BackTranslator._build_result` (filtre similarite rejet/accept, bornes min/max strictes, preservation langues), `augment()` validation (longueurs incoherentes -> ValueError, input vide), routing par langue (langues non supportees loguees, pivot correct EN<->FR, comptage rejets low/high). Mocks de `load`, `_translate_batch`, `_compute_similarities` pour eviter chargement des modeles MarianMT/SentenceTransformer. La regle d'or val/test 100 % originaux est garantie structurellement par `train_with_unified_protocol` qui splitte via `original_indices` (lignes 1587-1657 de `training.py`) avant d'injecter les variantes dans le train uniquement.
- [x] Temps estime sur RX 7900 XTX : ~20-30 min pour 1 018 positifs (batch 32, MarianMT) — reellement ~13 min cumulees (2 runs successifs)

#### B3.4 Calibration post-training (temperature scaling + threshold tuning)

> Integre automatiquement apres chaque fold training. Gain ~+0.02-0.05 MCC indirect via seuil optimal.

- [x] Creer `src/greentech/ai/mlops/calibration.py` :
  - [x] Classe `TemperatureScaler` : 1 parametre T (scalaire), optimise sur val set par LBFGS pour minimiser la NLL. Reference : Guo et al. 2017 (arXiv:1706.04599), validee 2024-2025 pour BERT/DeBERTa fine-tunes. — ligne 79
  - [x] Fonction `find_optimal_threshold(y_true, y_proba, metric='mcc') -> tuple[float, float]` : grille 0.05-0.95 pas 0.01, retourne `(threshold_optimal, mcc_optimal)` — ligne 197
  - [x] Persistence : `temperature.json` (`{"T": 1.87, "optimized_on": "fold_3_val", "nll_before": ..., "nll_after": ...}`) et `optimal_threshold.json` (`{"threshold": 0.42, "metric": "mcc", "value": 0.78}`) dans le dossier du modele — `save_calibration` ligne 268, chargement auto par `inference.py:155-166`
- [x] **Platt scaling et isotonic regression NON retenus** : overfittent avec <1 000 positifs par fold (agent A) — documente dans le docstring de `calibration.py`
- [x] Integration dans le training loop :
  - [x] Apres chaque fold training, sur le val set du fold : optimiser T puis threshold — `training.py:1726-1739`
  - [x] Moyenner les 5 T et 5 thresholds -> un seul `temperature.json` / `optimal_threshold.json` au niveau modele — `training.py:1798-1820`
  - [x] Logger dans MLflow : `T_fold_X`, `threshold_fold_X`, `T_final`, `threshold_final` — `cv_temperature_mean` et `cv_threshold_mean` logues (ligne 1824-1825) + `temperature` / `threshold` par fold dans `fold_metrics_dict`
- [x] Tests unitaires avec dataset synthetique calibre/decalibre (2026-04-22) — `tests/unit/ai/test_calibration.py` (21 tests pytest). Couvre `TemperatureScaler` (init T=1, transform identite a T=1, fit reduit NLL sur logits sur-confiants, preservation argmax, accepte numpy et torch), `find_optimal_threshold` (borne grille, MCC=1 sur dataset separable, metric F1 OK, erreurs shape/grille invalide, grid_values stocke), `save_calibration`/`load_calibration` round-trip (T seul, threshold seul, les deux, dossier vide retourne None/None), `apply_calibration` (retour tuple 2 arrays, equivalence T=1 vs softmax, T eleve aplatit, seuil influence preds pas probas, default 0.5).

#### B3.5 Ensemble K-fold (K=5 seeds x 3) (IMPLEMENTE 2026-04-21)

> Gain MCC +0.03 a +0.07 documente. Moyenne des logits pre-sigmoid.
>
> **Architecture retenue** : assemblage automatique a la fin de `train_with_unified_protocol` (recommandation option (a), utilisateur 2026-04-21). Pour chaque fold, selection de la meilleure seed (max MCC val, F1 en tie-break) via `_select_best_seed_per_fold`. Strategy differente selon l'architecture :

- [x] Apres K=5 folds x 3 seeds = 15 trainings par modele, deux strategies selon l'architecture :
  - [x] **Qwen3-4B LoRA** : fusionner les 5 adapters (un par fold, seeds moyennes par fold) via `PeftModel.merge_and_unload()` puis conserver les poids fusionnes -> 1 seul modele prod, cout inference 1x — implemente dans `_merge_lora_adapters()` + `_average_lora_deltas()` + `_copy_tokenizer_artifacts()` (`training.py`). Moyenne arithmetique des deltas LoRA A/B (equivalent SWA light) avant `merge_and_unload()`. Sauvegarde dans `models/qwen3/merged/`. Rechargeable comme LoRA classique via `LoRAClassifier.load()`.
  - [x] **mDeBERTa** : charger les 5 modeles en parallele a l'inference, moyenner les logits. Cout ~5x latence, ~5.5 Go VRAM (5x 1.1 Go) sur RX 7900 XTX — classe `EnsembleClassifier` dans `inference.py` herite de `BaseClassifier`, charge les K membres via `classifier_factory`, moyenne les `proba_positive` a chaque `predict()`. Active automatiquement par `get_classifier()` quand `strategy=logit_average` detectee dans `ensemble_config.json`.
- [x] Alternative si mDeBERTa trop lent : garder les 3 meilleurs folds (best val MCC) au lieu des 5 — supporte nativement par `EnsembleClassifier` (nombre de membres = taille de `fold_paths` dans `ensemble_config.json`, modifiable a posteriori sans recode).
- [x] Enregistrer tous les modeles folds dans `models/<model>/folds/fold_X/` + un `ensemble_config.json` listant les folds selectionnes — `training.py:1690` sauvegarde dans `folds/fold_X_seed_Y/`. `_build_ensemble()` ecrit `ensemble_config.json` a la racine du modele avec `strategy`, `folds` (fold+seed+metriques+path), `inference_model_path(s)`, `calibration` (T+seuil moyens), `metadata` (date, n_folds, cv_mcc_mean/std). Tags MLflow `ensemble_strategy` + metric `ensemble_n_folds`.

#### B3.6 Validation Deepchecks renforcee

- [ ] Verifier que les tests Deepchecks existants (`tests/ai/`) couvrent :
  - [ ] Data leakage (train/test overlap, aucun article original ET son augmentation ne coexistent entre splits)
  - [ ] Distribution drift entre folds (langue, label, longueur texte)
  - [ ] Robustesse au bruit (typos, casing, ponctuation aleatoire via AEDA)
- [ ] Ajouter si manquants

#### B3.7 Decision finale

- [ ] **Critere** : conserver le modele (Qwen3-4B ou mDeBERTa) qui maximise `MCC_moyen_K-fold` avec `ecart-type_K-fold < 0.10`. Critere secondaire : latence < 200 ms, CO2 CodeCarbon documente.

---

### B4. Benchmark Final & Selection du modele

> **Plan defini par l'utilisateur** :
> 1. Benchmark BRUT (zero-shot, avant entrainement) de DeBERTa adapte + Qwen3-4B sur le nouveau dataset
> 2. Entrainement des 2 modeles avec la methode optimale (issue de B3)
> 3. Benchmark des 2 modeles entraines
> 4. Selection du modele final pour la production

#### B4.1 Selection de la version DeBERTa adaptee (TERMINE 2026-04-21)

- [x] Etudier la repartition linguistique du dataset enrichi (FR/EN/autre) — dataset final EN 74.75 % / FR 25.25 % (cf. B2.9), 1 018 Green IT dont 600 en FR.
- [x] Decision : **`microsoft/mdeberta-v3-base`** (multilingue) retenu. Les 25 % FR sont trop significatifs pour `deberta-v3-base` EN-pur (qui encoderait mal les 600 Green IT FR et fausserait le benchmark). mDeBERTa conserve l'architecture DeBERTa-v3 (278M params, encoder-only, DisentangledSelfAttention) mais pre-entraine sur 100 langues — benchmark equitable encoder-vs-decoder contre Qwen3-4B.
- [ ] Documenter la decision dans `docs/CHOIX_DEBERTA.md` (redaction finale apres benchmark B4.2-B4.3 pour inclure les metriques comparatives)
- [ ] Mettre a jour `settings.huggingface_model_encoder_base` (creer le setting si besoin) — a faire lors de l'implementation du benchmark B4.2

#### B4.2 Benchmark BRUT (zero-shot)

- [ ] Etendre `scripts/benchmark_baseline.py` :
  - [ ] Ajouter un benchmark zero-shot DeBERTa via `pipeline("zero-shot-classification")`
  - [ ] Conserver le benchmark zero-shot Qwen3-4B existant
  - [ ] Run MLflow : un par modele, experiment dedie `baseline-comparison-2026-04`
  - [ ] Metriques : MCC, F1, Recall, Precision, latence moyenne, CO2eq
- [ ] Lancer sur le nouveau dataset complet
- [ ] Generer un tableau comparatif (markdown) dans `docs/BENCHMARK_BRUT_2026-04.md`

#### B4.3 Entrainement des 2 modeles avec le protocole unifie B3

> Hyperparams ci-dessous issus des agents de recherche B (LoRA Qwen3-4B) et C (mDeBERTa). Les deux modeles suivent strictement le meme protocole B3 (stratification langue x label, class_weight, back-translation, calibration, ensemble K=5, 3 seeds par fold) pour un benchmark equitable.

- [ ] **Pour Qwen3-4B + LoRA** (`Qwen3Classifier` actualise dans `src/greentech/ai/models/training.py`) :
  - [ ] `target_modules="all-linear"` (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
  - [ ] `r=32, lora_alpha=64, lora_dropout=0.05`
  - [ ] Head : `AutoModelForSequenceClassification` num_labels=2 (pas prompt + generation : latence / 5-10)
  - [ ] `lr=1e-4`, scheduler `cosine`, `warmup_ratio=0.06`
  - [ ] 3 epochs, early stopping sur val MCC (patience 1), `metric_for_best_model="eval_matthews_correlation"`
  - [ ] `per_device_train_batch_size=2, gradient_accumulation_steps=16` (batch effectif 32)
  - [ ] `max_length=512`, `bf16=True`
  - [ ] **Piege critique** : appeler `model.enable_input_require_grads()` AVANT `get_peft_model()` (sinon gradient_checkpointing n'est pas propage aux adapters, issue HF #42947)
  - [ ] `gradient_checkpointing=True, use_reentrant=False`
  - [ ] Mode : forcer **non-thinking** (Qwen3 a un mode thinking par defaut, desactiver via template ou utiliser `Qwen3-4B-Base`)
  - [ ] K-fold K=5 x 3 seeds stratifie `(langue x label)` + calibration T/threshold post-fold
  - [ ] Run MLflow `qwen3-final-2026-04` (modele de production actuel)
  - [ ] Sauvegarder dans `models/qwen3/folds/fold_X_seed_Y/` + fusion finale dans `models/qwen3/merged/` via `PeftModel.merge_and_unload()`
- [ ] **Pour mDeBERTa-v3-base** (nouvelle classe `MDeBERTaClassifier` dans `training.py`) :
  - [ ] Base : `microsoft/mdeberta-v3-base`, `AutoModelForSequenceClassification` num_labels=2
  - [ ] **Precision** : `bf16=True` si `transformers >= 4.48` (bug #35332 DisentangledSelfAttention corrige en decembre 2024, PR #35336), sinon `fp32`. **fp16 strictement interdit** (NaN garanti sur DeBERTa, non lie a ROCm).
  - [ ] `lr=2e-5`, scheduler `linear`, `warmup_ratio=0.06`
  - [ ] `per_device_train_batch_size=16, gradient_accumulation_steps=2` (batch effectif 32)
  - [ ] 5 epochs, early stopping sur val MCC (patience 2)
  - [ ] `max_length=384` (couvre 98 % des resumes FR+titre, tokenizer SentencePiece FR genere ~1.6 tokens/mot, 200 mots FR ≈ 320 tokens + titre)
  - [ ] `weight_decay=0.01`, dropout 0.1 (default)
  - [ ] `attn_implementation="sdpa"` (Flash-Attention indisponible RDNA3, reference issue ROCm #4391)
  - [ ] `gradient_checkpointing=True`
  - [ ] K-fold K=5 x 3 seeds stratifie `(langue x label)` + calibration T/threshold post-fold
  - [ ] Run MLflow `mdeberta-final-2026-04` (concurrent encoder vs Qwen3 decoder pour le benchmark)
  - [ ] Sauvegarder dans `models/mdeberta/folds/fold_X_seed_Y/` + ensemble logit-average a l'inference (pas de fusion de poids possible car architectures full-fine-tune, pas LoRA)

#### B4.4 Benchmark comparatif des modeles entraines

- [x] Creer `scripts/benchmark_models.py` :
  - [x] Charger les 2 modeles entraines — MODEL_REGISTRY (lignes 73-82) mappe `qwen3` vers `Qwen3Classifier` et `mdeberta` vers `MDeBERTaClassifier`, chargement via `_load_classifier_for_model()` (ligne 119)
  - [ ] Evaluer sur le meme test set (split fige, hold-out non vu pendant l'entrainement) — **RESTE A FAIRE** : execution en attente des modeles entraines (B4.3)
  - [ ] Metriques completes : MCC, F1, Recall, Precision, balanced_accuracy, specificite, matrice de confusion, latence p50/p95, CO2eq — **RESTE A FAIRE** (execution)
  - [ ] Generer un rapport markdown comparatif dans `docs/BENCHMARK_FINAL_2026-04.md` — **RESTE A FAIRE** (execution)
- [ ] Run MLflow `model-selection-final-2026-04` — **RESTE A FAIRE**

#### B4.5 Selection et promotion du modele

- [ ] **Critere de selection** : MCC moyen K-fold le plus eleve, sous condition que la latence reste < 200ms et l'ecart-type MCC < 0.10
- [ ] Documenter la decision dans `docs/SELECTION_CHAMPION_2026-04.md`
- [ ] Promotion du modele vainqueur :
  - [ ] Copier dans `models/production/`
  - [ ] Tag DVC : `uv run dvc add models/production && uv run dvc push`
  - [ ] Mettre a jour `models/production.dvc`
- [ ] Mettre a jour la "Model Card" : `docs/MODEL_CARD.md`

#### B4.6 Validation end-to-end

- [ ] Tests Deepchecks complets sur le modele final
- [ ] Test API : redemarrer FastAPI, lancer 10 analyses via `/analyze`, verifier les resultats
- [ ] Test Frontend : utiliser l'interface React pour analyser des articles divers (3-5 manuels)
- [ ] Verifier les dashboards Grafana : metriques d'inference correctes

#### B4.7 Mise a jour documentation finale

- [ ] Mettre a jour `docs/PLAN_ETAPES.md` section 3.3 avec les nouveaux entrainements
- [ ] Mettre a jour la documentation interne (section "Classifieur fine-tune Qwen3-4B + LoRA") avec la nouvelle famille de modele de production
- [ ] Mettre a jour la documentation Sphinx
- [ ] Tag Git : `vX.Y.Z-prod-2026-04`

---

### B5. (BONUS) Refonte Agentic avec LangGraph

> **A realiser uniquement si B1 a B4 sont entierement valides ET temps disponible.**
> **Framework retenu** : LangGraph (decision utilisateur).
> **Objectif demo** : presenter au jury une architecture moderne et observable, montrant la maturite du projet.

#### B5.1 Conception architecture

- [ ] Creer `docs/ARCHITECTURE_AGENTIC.md` :
  - [ ] Schema des agents et de leurs responsabilites (Mermaid ou diagramme)
  - [ ] Definition de l'etat partage (LangGraph State)
  - [ ] Flux de donnees entre agents (Cleaner -> Summarizer -> Judge -> Classifier)
  - [ ] Strategie de fallback (HF cloud -> local Qwen 2.5-3B -> Qwen 2.5-1.5B)
  - [ ] Strategie d'observabilite (Loguru -> Loki, Prometheus metrics par agent)
- [ ] Validation du design (relecture complete)

#### B5.2 Setup LangGraph

- [ ] Ajouter les dependances : `uv add langgraph langchain-core`
- [ ] Verifier compatibilite avec Pydantic 2.x
- [ ] Creer le package : `src/greentech/ai/agents/__init__.py`
- [ ] Creer le module d'etat partage : `src/greentech/ai/agents/state.py`
  - [ ] TypedDict ou Pydantic model pour l'etat de l'analyse
  - [ ] Champs : `article_url`, `article_text`, `cleaned_text`, `summary`, `is_green_it`, `judge_reasoning`, `classifier_proba`, `errors`

#### B5.3 Implementation des agents

- [ ] **Agent Cleaner** : `src/greentech/ai/agents/cleaner_agent.py`
  - [ ] Tool : nettoyage texte (reuse `spark_cleaner` logic en mode synchrone)
  - [ ] Tool : detection langue (langid ou langdetect)
  - [ ] Tool : traduction si necessaire (Qwen via dispatcher)
- [ ] **Agent Summarizer** : `src/greentech/ai/agents/summarizer_agent.py`
  - [ ] Tool : generation resume classification (`classification_summarizer`, max 450 tokens)
  - [ ] Tool : generation resume Green IT si confirme (`summarizer.summarize_green_aspects`)
- [ ] **Agent Judge Green IT** : `src/greentech/ai/agents/judge_agent.py`
  - [ ] Tool : pre-filtre keywords (etage 1)
  - [ ] Tool : LLM judge (etage 2) via dispatcher HF/local
  - [ ] Sortie structuree : decision booleenne + justification + score de confiance
- [ ] **Agent Classifier** : `src/greentech/ai/agents/classifier_agent.py`
  - [ ] Tool : classification via le modele en production (Qwen3-4B ou mDeBERTa)
  - [ ] Sortie : prediction + probabilite + seuil utilise
- [ ] **Agent Orchestrator** : `src/greentech/ai/agents/orchestrator.py`
  - [ ] Workflow LangGraph : noeuds + edges conditionnels
  - [ ] Gestion erreurs : retry avec backoff sur les agents LLM
  - [ ] Logs structures Loguru -> Loki

#### B5.4 Tests unitaires & integration

- [ ] Tests unitaires par agent : `tests/ai/agents/test_<agent>.py`
- [ ] Tests d'integration de l'orchestrator : flux complet sur articles fictifs
- [ ] Test de fallback : simuler une panne HF, verifier le basculement local
- [ ] Test de reprise : couper en plein milieu, verifier la coherence d'etat

#### B5.5 Integration dans l'API

- [ ] Refactorer `src/greentech/api/routes/analyze.py` :
  - [ ] Remplacer la logique lineaire par un appel a l'orchestrator LangGraph
  - [ ] Garder l'ancienne logique en fallback (feature flag dans settings) ou suppression complete (a discuter)
- [ ] Tests d'integration API : verifier que les reponses sont identiques en sortie

#### B5.6 Observabilite agentic

- [ ] Ajouter des metriques Prometheus par agent :
  - [ ] `agent_duration_seconds{agent="cleaner|summarizer|judge|classifier"}`
  - [ ] `agent_errors_total{agent="..."}`
  - [ ] `agent_fallback_total{agent="..."}`
- [ ] Creer un dashboard Grafana dedie : `config/grafana/dashboards/agentic-pipeline.json`
- [ ] Alertes Prometheus : taux d'erreur agent > 5%, fallback rate > 20%

#### B5.7 Documentation finale

- [ ] Documentation Sphinx complete du package `agents`
- [ ] Demonstration interactive : notebook `notebooks/demo_agentic.ipynb` ou script `scripts/demo_agentic.py`
- [ ] Mettre a jour la documentation interne avec la nouvelle architecture
- [ ] Mettre a jour `docs/PLAN_ETAPES.md` avec une nouvelle section bonus 7
- [ ] Tag Git : `vX.Y.Z-agentic`

---

## Suivi de la Section Bonus

A chaque sous-phase Bx.y terminee :
- [ ] Cocher les cases correspondantes ci-dessus
- [ ] Si la sous-phase touche a une competence C1-C21 deja validee, mettre a jour la competence concernee (ex: C1 si nouveaux collecteurs, C7 si nouveau benchmark, C11 si nouvelles metriques)
- [ ] Mettre a jour `docs/PLAN_ETAPES.md` si l'etape impacte le plan general
- [ ] Commit Git avec message structure (`type(scope): message`) selon les conventions du projet
- [ ] Pousser les nouveaux datasets via DVC apres chaque enrichissement majeur
