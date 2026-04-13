# Specifications Techniques - Extraction & Stockage des Donnees

>
> Document de reference pour la conception du MCD et le developpement
> des modules de collecte, nettoyage et stockage (Etape 2 - Bloc E1).

---

## 1. Vue d'ensemble du flux de donnees

```
                        ┌──────────────────┐
                        │  PostgreSQL       │
                        │  search_config    │
                        │  (mots-cles,      │
                        │   URLs, priorites)│
                        └────────┬─────────┘
                                 │ Lecture config (SQLAlchemy)
                     ┌───────────┼───────────┐
                     ▼           ▼           ▼
              ┌────────────┐ ┌──────────┐ ┌──────────────┐
              │ Module 1   │ │ Module 2 │ │ Module 3     │
              │ API        │ │ Scraping │ │ Fichier      │
              │ NewsData.io│ │ TechCrunc│ │ arXiv/Kaggle │
              │ (httpx)    │ │ (RSS +   │ │ (Python)     │
              │            │ │ Scrapy+  │ │              │
              │            │ │ Playwrigh│ │              │
              └─────┬──────┘ └────┬─────┘ └──────┬───────┘
                    │             │               │
                    ▼             ▼               ▼
              ┌─────────────────────────────────────────┐
              │         MinIO - Bucket [raw-data]       │
              │   JSON brut / HTML brut / JSON dataset  │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │         Apache Spark (PySpark)           │
              │  - Lecture depuis MinIO (S3)             │
              │  - Nettoyage (HTML, doublons, dates)     │
              │  - Agregation en DataFrame unique        │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │        MinIO - Bucket [clean-data]      │
              │         Format Parquet normalise         │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │         PostgreSQL (SQLAlchemy 2.0)      │
              │  Tables : articles, sources, etc.        │
              │  Insertion async + Upsert anti-doublons  │
              └─────────────────────────────────────────┘
```

---

## 2. Sources de donnees externes

### 2.1 Source API : NewsData.io

| Critere                | Detail                                                        |
|------------------------|---------------------------------------------------------------|
| **Type**               | API REST publique (JSON)                                      |
| **URL**                | `https://newsdata.io/api/1/latest`                            |
| **Documentation**      | <https://newsdata.io/documentation>                           |
| **Authentification**   | Cle API obligatoire (parametre `apikey`)                      |
| **Quota**              | 200 credits/jour (plan gratuit), 10 articles/requete          |
| **Latence**            | Fraicheur des articles : 15 min a 12h                         |
| **Format reponse**     | JSON (`status`, `totalResults`, `results[]`)                  |
| **Pagination**         | Curseur `nextPage` dans la reponse                            |
| **Rate limiting**      | Non documente, prevoir delai 1s entre requetes                |
| **Confidentialite**    | Cle API stockee dans `.env` (jamais dans le code)             |

**Parametres de requete utilises** :

| Parametre  | Valeur                                  | Obligatoire |
|------------|-----------------------------------------|-------------|
| `apikey`   | Variable d'environnement                | Oui         |
| `q`        | Mots-cles depuis `search_config`        | Oui         |
| `category` | `technology`                            | Oui         |
| `language` | `en`                                    | Non         |

**Champs JSON recuperes par article** :

| Champ API       | Type     | Description                    | Nullable |
|-----------------|----------|--------------------------------|----------|
| `title`         | string   | Titre de l'article             | Non      |
| `link`          | string   | URL de l'article original      | Non      |
| `description`   | string   | Resume court fourni par l'API  | Oui      |
| `content`       | string   | Contenu complet (si disponible)| Oui      |
| `pubDate`       | string   | Date ISO 8601                  | Oui      |
| `source_name`   | string   | Nom du media d'origine         | Oui      |
| `creator`       | string[] | Noms des auteurs               | Oui      |
| `image_url`     | string   | URL image illustration         | Oui      |

**Contraintes techniques** :

- Requetes HTTP asynchrones via `httpx.AsyncClient`
- Gestion des erreurs HTTP : retry sur 429 (rate limit) et 5xx, abandon sur 4xx
- Timeout de 30 secondes par requete
- Les mots-cles proviennent dynamiquement de la table `search_config`

---

### 2.2 Source Scraping : TechCrunch Climate

| Critere                | Detail                                                        |
|------------------------|---------------------------------------------------------------|
| **Type**               | Scraping HTML hybride (RSS pour decouverte + Scrapy/Playwright pour le HTML) |
| **URLs cibles**        | Flux RSS : `https://techcrunch.com/category/climate/feed/` + pages d'articles scrapees |
| **Authentification**   | Aucune                                                        |
| **Rendu JavaScript**   | Obligatoire sur les pages d'articles (React hydration)        |
| **Outils imposes**     | `scrapy` + `scrapy-playwright` + `playwright` (+ `httpx` et `feedparser` pour l'etape RSS) |
| **robots.txt**         | Respecte imperativement (`ROBOTSTXT_OBEY = True`)             |
| **Delai entre requetes** | Minimum 2 secondes (ethique scraping, `DOWNLOAD_DELAY`)    |
| **User-Agent**         | Identifie : `GreenTech-Bot/1.0`                               |

**Strategie d'extraction en deux etapes** :

**Etape 1 - Decouverte d'URLs via RSS** :
1. Requete HTTP GET sur le flux RSS officiel `/category/climate/feed/`
2. Parsing XML via `feedparser.parse(response.text)`
3. Extraction des URLs d'articles (`entry.link`), filtrage optionnel par mots-cles

**Etape 2 - Scraping HTML (coeur de la source)** :
1. Instanciation d'un `TechCrunchArticleSpider` Scrapy avec la liste d'URLs
2. Pour chaque URL : requete via Playwright (rendu JS complet)
3. Attente du selecteur `h1` (timeout 15s) pour confirmer le chargement
4. Parsing du DOM rendu via selecteurs CSS (voir tableau ci-dessous)
5. Sauvegarde JSON normalisee dans MinIO (`raw-data/scraping/techcrunch/`)

**Selecteurs CSS appliques sur le HTML des articles** :

| Champ              | Selecteur CSS                                         |
|--------------------|-------------------------------------------------------|
| Titre              | `h1.article-hero__title::text`                        |
| Titre (fallback)   | `meta[property="og:title"]::attr(content)`            |
| Date publication   | `time::attr(datetime)` (ISO 8601)                     |
| Auteurs            | `a[href*="/author/"]::text`                           |
| Contenu texte      | `div.entry-content p::text` (concatene)               |
| Contenu HTML brut  | `div.entry-content` (HTML complet)                    |
| Resume court       | `meta[property="og:description"]::attr(content)`      |

**Donnees extraites par article** :

| Champ extrait     | Type   | Source                                 | Nullable |
|-------------------|--------|----------------------------------------|----------|
| Titre             | string | `h1.article-hero__title` du HTML       | Non      |
| URL               | string | URL scraped (apres redirections)       | Non      |
| Date publication  | string | `<time datetime>` (ISO 8601)           | Oui      |
| Auteur            | string | `a[href*="/author/"]`                  | Oui      |
| Contenu           | string | Paragraphes de `div.entry-content`     | Non      |
| Contenu HTML      | string | Bloc `div.entry-content` brut          | Non      |
| Resume court      | string | `meta[property="og:description"]`      | Oui      |

**Contraintes techniques respectees** :

- Max 20 articles par session (`MAX_ARTICLES`)
- Timeout HTTP de 20 secondes (`REQUEST_TIMEOUT`)
- Sauvegarde JSON normalisee dans `raw-data/scraping/techcrunch/`
- Gestion propre des erreurs par URL (une URL cassee n'interrompt pas les autres)

**Pourquoi l'architecture hybride** : TechCrunch a refondu sa structure HTML en
avril 2026 (suppression de la balise `<article>` utilisee comme selecteur sur
la page d'index). Plutot que de scraper le listing HTML (fragile), on delegue
cette decouverte au flux RSS officiel (stable et standardise), puis on scrape
le HTML de **chaque page d'article individuel** (qui garde une structure
predictible : `h1.article-hero__title`, `div.entry-content`, etc.). Le resultat
satisfait le critere C1 ("telechargement HTML + parsing HTML") tout en etant
robuste aux refontes futures de la page d'index.

---

### 2.3 Source Fichier : arXiv Metadata Dataset

| Critere                | Detail                                                        |
|------------------------|---------------------------------------------------------------|
| **Type**               | Dataset statique (fichier JSON volumineux)                    |
| **Source**             | Kaggle / Cornell University                                   |
| **URL**                | <https://www.kaggle.com/datasets/Cornell-University/arxiv>    |
| **Licence**            | CC0 (Domaine Public)                                          |
| **Taille**             | ~3.6 Go (JSON)                                                |
| **Volume**             | 1.7 million+ articles scientifiques                           |
| **Format**             | JSON Lines (un objet JSON par ligne)                          |
| **Telechargement**     | Manuel depuis Kaggle (necessite un compte gratuit)            |

**Champs JSON utilises par article** :

| Champ JSON     | Type     | Description                         | Nullable |
|----------------|----------|-------------------------------------|----------|
| `id`           | string   | Identifiant unique arXiv            | Non      |
| `title`        | string   | Titre de la publication             | Non      |
| `abstract`     | string   | Resume scientifique complet         | Non      |
| `categories`   | string   | Categories arXiv (espace-separe)    | Non      |
| `update_date`  | string   | Date de derniere mise a jour        | Oui      |
| `authors`      | string   | Noms des auteurs (format brut)      | Oui      |

**Contraintes techniques** :

- Le volume (3.6 Go) interdit le chargement en memoire via Python standard
- Traitement obligatoire via Apache Spark (PySpark)
- Filtrage des categories pertinentes : `cs.AI`, `cs.CY`, `cs.SE`, `cs.DC`
  (Intelligence Artificielle, Societe, Genie Logiciel, Calcul Distribue)
- Le fichier est lu ligne par ligne puis uploade vers MinIO `raw-data`

---

## 3. Source interne : Configuration SQL (PostgreSQL)

> Cette source ne fournit pas d'articles mais des **parametres dynamiques**
> utilises par les modules de collecte. Elle valide les competences C1 et C2
> (connexion SGBD + requetes SQL).

| Critere                | Detail                                                        |
|------------------------|---------------------------------------------------------------|
| **Type**               | Base de donnees relationnelle                                 |
| **SGBD**               | PostgreSQL 15                                                 |
| **ORM**                | SQLAlchemy 2.0+ (mode asynchrone via asyncpg)                 |
| **Table**              | `search_config`                                               |
| **Role**               | Fournir les mots-cles et URLs aux modules API et Scraping     |

**Structure de la table `search_config`** :

| Colonne       | Type         | Description                                  |
|---------------|--------------|----------------------------------------------|
| `id`          | SERIAL PK    | Identifiant auto-incremente                  |
| `keyword`     | VARCHAR(100) | Mot-cle de recherche                         |
| `source_url`  | TEXT         | URL specifique (optionnel)                   |
| `source_type` | VARCHAR(20)  | Type cible : `api`, `scraping`, `file`       |
| `priority`    | INTEGER      | Priorite d'execution (1 = haute)             |
| `active`      | BOOLEAN      | Actif ou desactive                           |
| `created_at`  | TIMESTAMPTZ  | Date de creation                             |
| `updated_at`  | TIMESTAMPTZ  | Date de derniere modification                |

**Extraction** :

```sql
SELECT keyword, source_url, source_type
FROM search_config
WHERE active = true
ORDER BY priority ASC;
```

---

## 4. Infrastructure de stockage

### 4.1 MinIO (Stockage Objet / Data Lake)

| Critere        | Detail                                               |
|----------------|------------------------------------------------------|
| **Role**       | Data Lake : stockage brut et nettoye                 |
| **Protocole**  | Compatible S3                                        |
| **Acces**      | `localhost:9000` (API) / `localhost:9001` (Console)  |
| **Credentials**| Variable `.env` (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`) |

**Buckets** :

| Bucket        | Contenu                                      | Ecrit par       | Lu par       |
|---------------|----------------------------------------------|-----------------|--------------|
| `raw-data`    | JSON brut (API), HTML brut (Scraping), JSON (arXiv) | Modules 1, 2, 3 | Spark       |
| `clean-data`  | Parquet normalise (DataFrame agrege)         | Spark            | SQLAlchemy  |
| `models`      | Modeles IA entraines (Etape 3)               | Training script  | API FastAPI |
| `mlflow`      | Artefacts MLflow (Etape 3)                   | MLflow           | MLflow UI   |

### 4.2 PostgreSQL (Base Relationnelle)

| Critere        | Detail                                               |
|----------------|------------------------------------------------------|
| **Role**       | Stockage structure des metadonnees + resultats IA    |
| **Version**    | PostgreSQL 15 (Docker)                               |
| **Driver**     | asyncpg (asynchrone)                                 |
| **ORM**        | SQLAlchemy 2.0+ (mode async)                         |
| **Base**       | `greentech_db`                                       |
| **Acces**      | `localhost:5432`                                     |

---

## 5. Regles de nettoyage (Spark)

Le pipeline Spark applique les transformations suivantes sur les donnees brutes :

| Regle                          | Description                                                  |
|--------------------------------|--------------------------------------------------------------|
| Suppression HTML               | Retrait de toutes les balises HTML residuelles du contenu    |
| Detection doublons             | Deduplication par URL (cle unique)                           |
| Entrees corrompues             | Suppression des lignes sans titre OU sans URL                |
| Normalisation dates            | Conversion de tous les formats de date vers ISO 8601 (UTC)  |
| Normalisation encodage         | Conversion en UTF-8 uniforme                                 |
| Troncature contenu             | Limitation du contenu texte a 50 000 caracteres              |
| Champ `source_type`            | Ajout d'un champ indiquant l'origine (`api`, `scraping`, `file`) |

**Schema du DataFrame agrege (sortie Spark)** :

| Colonne         | Type      | Description                              | Nullable |
|-----------------|-----------|------------------------------------------|----------|
| `title`         | string    | Titre normalise                          | Non      |
| `url`           | string    | URL unique de l'article                  | Non      |
| `content`       | string    | Contenu texte nettoye (sans HTML)        | Oui      |
| `summary`       | string    | Resume (description API ou abstract)     | Oui      |
| `author`        | string    | Auteur (anonymise si nom personnel)      | Oui      |
| `published_at`  | timestamp | Date de publication (ISO 8601 UTC)       | Oui      |
| `source_type`   | string    | Origine : `api`, `scraping`, `file`      | Non      |
| `source_name`   | string    | Nom de la source d'origine               | Non      |
| `raw_data_path` | string    | Chemin MinIO du fichier brut             | Non      |
| `language`      | string    | Langue detectee (defaut: `en`)           | Oui      |

---

## 6. Conformite RGPD

### 6.1 Donnees personnelles identifiees

| Source          | Donnee               | Nature        | Traitement                         |
|-----------------|----------------------|---------------|-------------------------------------|
| NewsData.io     | `creator` (auteurs)  | Nom de personne | Anonymisation : initiales ou suppression |
| TechCrunch      | Auteur article       | Nom de personne | Anonymisation : initiales ou suppression |
| arXiv           | `authors`            | Noms de chercheurs | Anonymisation : initiales ou suppression |

### 6.2 Regles d'anonymisation

- Les noms d'auteurs sont transformes en initiales (`John Doe` → `J.D.`) pendant le nettoyage Spark
- Aucune adresse email n'est stockee en base
- Les donnees brutes dans MinIO `raw-data` peuvent contenir des noms complets
  (acces restreint, non expose par l'API)
- Seules les donnees nettoyees (MinIO `clean-data` et PostgreSQL) sont exposees

### 6.3 Base legale

- **API NewsData.io** : Donnees publiques accessibles via API publique
- **TechCrunch** : Contenu editorial public, scraping a usage academique
- **arXiv** : Licence CC0 (Domaine Public), aucune restriction

---

## 7. Synthese des entites pour le MCD

> Ce tableau recapitule les entites identifiees pour la modelisation
> conceptuelle (MCD) sur Looping.

| Entite            | Description                                          | Origine          |
|-------------------|------------------------------------------------------|------------------|
| **SOURCE**        | Provenance des articles (NewsData.io, TechCrunch, arXiv) | Configuration |
| **ARTICLE**       | Article collecte, nettoye et potentiellement analyse | Collecte + Spark |
| **SEARCH_CONFIG** | Parametres dynamiques de recherche (mots-cles)       | Configuration    |
| **USER**          | Utilisateur de l'application web                     | Etape 4 (Auth)   |
| **ANALYSIS_LOG**  | Trace d'une inference IA sur un article              | Etape 3 (IA)     |
| **DAILY_STATS**   | Statistiques agregees par jour                       | Etape 4 (API)    |

### Associations identifiees

| Association             | Entite 1       | Cardinalite | Entite 2       |
|-------------------------|----------------|-------------|----------------|
| **produit**             | SOURCE         | (1,n)       | ARTICLE        |
| **genere**              | ARTICLE        | (0,n)       | ANALYSIS_LOG   |
| **effectue** *(futur)*  | USER           | (0,n)       | ANALYSIS_LOG   |

> Note : L'association USER ↔ ANALYSIS_LOG sera ajoutee a l'etape 4 (Backend).
> A ce stade du MCD, elle peut etre anticipee ou ajoutee plus tard.

---

## 8. Guide de modelisation MCD / MLD sur Looping

> Guide pas-a-pas pour realiser le Modele Conceptuel de Donnees (MCD) puis
> le Modele Logique de Donnees (MLD) dans le logiciel Looping, a partir
> des specifications decrites dans ce document.

### 8.1 Correspondance des types Looping / PostgreSQL

Dans Looping, le choix du type d'une propriete se fait en **3 niveaux** :

1. **Categorie principale** : Numerique, Texte, Temporel, Logique
2. **Sous-categorie** (selon la categorie)
3. **Parametre** (longueur, precision, etc.)

Voici le detail des choix dans l'interface Looping et leur correspondance SQL :

#### Categorie "Numerique"

Trois sous-categories sont disponibles : **Entier**, **Decimal** et **Reel**.
Chacune demande un parametre de precision supplementaire.

**Entier** — choix de la taille en bits :

| Taille Looping | Type PostgreSQL    | Plage de valeurs                      | Choix pour ce projet |
|----------------|--------------------|---------------------------------------|----------------------|
| 8 bits         | `SMALLINT` (rare)  | -128 a 127                            | Non utilise          |
| 16 bits        | `SMALLINT`         | -32 768 a 32 767                      | Non utilise          |
| **32 bits**    | `INTEGER` / `SERIAL` | -2 147 483 648 a 2 147 483 647      | **Toujours celui-ci** |
| 64 bits        | `BIGINT` / `BIGSERIAL` | ±9.2 × 10^18                      | Non utilise          |

> **Choix retenu : Entier 32 bits** pour tous les champs entiers du projet.
> Les identifiants auto-incrementes (`SERIAL`) et les compteurs (nombre
> d'articles, priorite, temps en ms) restent largement dans la plage 32 bits.
> Le 64 bits serait surdimensionne pour nos besoins.

**Decimal** — choix du nombre total de chiffres et du nombre apres la virgule :

| Parametre Looping          | Type PostgreSQL        | Exemple                  | Choix pour ce projet |
|----------------------------|------------------------|--------------------------|----------------------|
| Chiffres: `n`, Decimales: `d` | `NUMERIC(n,d)` / `DECIMAL(n,d)` | `DECIMAL(10,2)` = 99999999.99 | **Non utilise** |

> Le type Decimal n'est pas utilise dans ce projet. Il serait pertinent
> pour des montants financiers (prix, factures) qui exigent une precision
> exacte. Nos scores de confiance et emissions carbone tolerent
> l'approximation du type Reel.

**Reel** — choix de la precision :

| Precision Looping     | Type PostgreSQL      | Precision                    | Choix pour ce projet |
|-----------------------|----------------------|------------------------------|----------------------|
| simple 32 bits        | `REAL`               | ~7 chiffres significatifs    | Non utilise          |
| **double 64 bits**    | `DOUBLE PRECISION`   | ~15 chiffres significatifs   | **Toujours celui-ci** |

> **Choix retenu : Reel double 64 bits** pour tous les champs reels du projet.
> En PostgreSQL, le type `FLOAT` (utilise dans `init.sql`) correspond a
> `DOUBLE PRECISION` (64 bits) par defaut. Choisir "double 64 bits" dans
> Looping garantit la coherence avec le schema SQL existant.

#### Categorie "Texte"

On choisit d'abord le jeu de caracteres (`caracteres ASCII` / `caracteres unicode` /
`binaire`), puis la sous-categorie de taille :

| Jeu de caracteres | Sous-categorie | Longueur  | Type PostgreSQL | Usage                                       |
|-------------------|----------------|-----------|-----------------|----------------------------------------------|
| unicode           | **variable**   | `n`       | `VARCHAR(n)`    | Chaines courtes a longueur limitee (noms, emails, codes) |
| unicode           | **volumineux** | *(aucune)*| `TEXT`           | Chaines longues **sans limite** (contenu, URL, description) |
| unicode           | fixe           | `n`       | `CHAR(n)`       | Non utilise dans ce projet                   |

> **Precision importante sur "volumineux"** : c'est le choix qui correspond au
> type SQL `TEXT`, soit une chaine de caracteres de **longueur illimitee**.
> Contrairement a "variable" (= `VARCHAR(n)`) qui impose un maximum de *n*
> caracteres, "volumineux" n'a aucune borne de taille. On l'utilise pour tous
> les champs potentiellement tres longs : `contenu`, `resume`, `url`,
> `description`, `chemin_donnees_brutes`, etc.
>
> **Toujours choisir "caracteres unicode"** (pas ASCII) pour garantir le
> support des accents, caracteres speciaux et alphabets non latins.

#### Categorie "Temporel"

| Sous-categorie Looping | Type PostgreSQL              | Usage                                    |
|-------------------------|------------------------------|------------------------------------------|
| **Date**                | `DATE`                       | Dates sans heure (stats quotidiennes)    |
| **DateHeure**           | `TIMESTAMP WITH TIME ZONE`   | Horodatages complets avec fuseau horaire |

#### Categorie "Logique"

| Sous-categorie Looping | Type PostgreSQL | Usage                                      |
|-------------------------|-----------------|--------------------------------------------|
| **Booleen**             | `BOOLEAN`       | Drapeaux vrai/faux (actif, est_green_it)   |

#### Recapitulatif rapide (notation abregee pour les tableaux suivants)

Pour alleger les tableaux des entites, la notation abregee suivante est utilisee :

| Notation abregee dans ce guide | Chemin complet dans Looping                                    |
|--------------------------------|----------------------------------------------------------------|
| `Entier`                       | Numerique → Entier → **32 bits**                               |
| `Reel`                         | Numerique → Reel → **double 64 bits**                          |
| `Variable(n)`                  | Texte → caracteres unicode → variable → longueur `n`           |
| `Volumineux`                   | Texte → caracteres unicode → volumineux                        |
| `Date`                         | Temporel → Date                                                |
| `DateHeure`                    | Temporel → DateHeure                                           |
| `Booleen`                      | Logique → Booleen                                              |

> **Precision sur les UUID** : Looping ne gere pas nativement le type `UUID`
> de PostgreSQL. Utiliser `Variable(36)` pour representer un UUID
> (format standard : 32 caracteres hexadecimaux + 4 tirets = 36 caracteres).
> La generation effective (`uuid_generate_v4()`) sera geree cote SQL.

---

### 8.2 Preparation du projet Looping

1. Lancer **Looping** (double-clic sur l'executable)
2. Menu **Fichier → Nouveau**
3. Sauvegarder immediatement : **Fichier → Enregistrer sous** →
   `greentech_intelligence.loo` (dans le dossier `docs/`)
4. S'assurer que l'on est bien en vue **MCD** (onglet ou menu **Affichage → MCD**)

---

### 8.3 Creation des entites

Pour chaque entite : **clic droit** sur la zone de travail → **Ajouter une entite**,
ou utiliser l'icone entite (rectangle) dans la barre d'outils.

Ensuite, **double-cliquer** sur l'entite pour ouvrir la fenetre de proprietes
et saisir le nom, puis ajouter les proprietes une par une.

> Convention de nommage : dans le MCD, les noms d'entites sont en
> **MAJUSCULES** et les proprietes en **minuscules_snake_case**.

> **Ce que l'on configure dans Looping pour chaque propriete :**
>
> Dans la fenetre de parametrage d'une propriete, on peut cocher **3 cases** :
>
> | Case Looping    | Signification                                          |
> |-----------------|--------------------------------------------------------|
> | **Identifiant** | Cle primaire (la propriete sera soulignee sur le MCD)  |
> | **NOT NULL**    | Valeur obligatoire (interdit les champs vides)         |
> | **UNIQUE**      | Pas de doublon autorise dans cette colonne             |
>
> **Ce qui n'est PAS configurable dans Looping** (gere uniquement dans `init.sql`) :
>
> | Contrainte SQL       | Exemple                                        |
> |----------------------|------------------------------------------------|
> | `DEFAULT`            | `DEFAULT NOW()`, `DEFAULT true`, `DEFAULT 0`   |
> | `CHECK`              | `CHECK (type IN ('api','scraping','file'))`     |
> | `ON DELETE`          | `ON DELETE CASCADE`, `ON DELETE SET NULL`        |
> | Triggers             | `update_updated_at_column()`                    |
> | Extensions           | `uuid_generate_v4()`                            |
>
> Ces contraintes sont deja presentes dans le fichier `scripts/sql/init.sql`.
> Elles n'apparaissent dans les tableaux ci-dessous qu'a titre de reference,
> dans la colonne "SQL uniquement".

---

#### Entite 1 : SOURCE

> Represente les provenances des articles (NewsData.io, TechCrunch, arXiv).
> Correspond a la table `sources` dans `init.sql`.

| Propriete           | Type Looping  | Identifiant | NOT NULL | UNIQUE  | SQL uniquement (hors Looping) |
|---------------------|---------------|-------------|----------|---------|-------------------------------|
| `id_source`         | Entier        | **Oui**     |          |         |                               |
| `nom`               | Variable(100) |             | **Oui**  | **Oui** |                               |
| `type`              | Variable(20)  |             | **Oui**  |         | `CHECK IN ('api','scraping','file')` |
| `url_base`          | Volumineux    |             |          |         |                               |
| `description`       | Volumineux    |             |          |         |                               |
| `est_active`        | Booleen       |             |          |         | `DEFAULT true`                |
| `derniere_collecte` | DateHeure     |             |          |         |                               |
| `date_creation`     | DateHeure     |             |          |         | `DEFAULT NOW()`               |

> Note : quand **Identifiant** est coche, Looping considere automatiquement
> la propriete comme NOT NULL et UNIQUE — inutile de cocher ces cases en plus.

**Saisie dans Looping :**

1. Creer l'entite, la nommer `SOURCE`
2. `id_source` : Numerique → Entier → **32 bits**, cocher **Identifiant**
3. `nom` : Texte → caracteres unicode → variable → longueur `100`, cocher **NOT NULL** + **UNIQUE**
4. `type` : Texte → caracteres unicode → variable → longueur `20`, cocher **NOT NULL**
5. `url_base` et `description` : Texte → caracteres unicode → **volumineux**
6. `est_active` : Logique → Booleen
7. `derniere_collecte` et `date_creation` : Temporel → DateHeure

---

#### Entite 2 : ARTICLE

> Entite centrale du projet — article collecte, nettoye et potentiellement
> analyse par l'IA. Correspond a la table `articles` dans `init.sql`.

| Propriete                | Type Looping   | Identifiant | NOT NULL | UNIQUE  | SQL uniquement (hors Looping) |
|--------------------------|----------------|-------------|----------|---------|-------------------------------|
| `id_article`             | Entier         | **Oui**     |          |         |                               |
| `uuid`                   | Variable(36)   |             |          | **Oui** | `DEFAULT uuid_generate_v4()`  |
| `titre`                  | Variable(500)  |             | **Oui**  |         |                               |
| `url`                    | Volumineux     |             | **Oui**  | **Oui** |                               |
| `contenu`                | Volumineux     |             |          |         |                               |
| `resume`                 | Volumineux     |             |          |         |                               |
| `auteur`                 | Variable(200)  |             |          |         |                               |
| `date_publication`       | DateHeure      |             |          |         |                               |
| `langue`                 | Variable(10)   |             |          |         | `DEFAULT 'en'`                |
| `est_green_it`           | Booleen        |             |          |         |                               |
| `score_confiance`        | Reel           |             |          |         | `CHECK (>= 0 ET <= 1)`       |
| `modele_classification`  | Variable(100)  |             |          |         |                               |
| `chemin_donnees_brutes`  | Volumineux     |             |          |         |                               |
| `date_analyse`           | DateHeure      |             |          |         |                               |
| `date_creation`          | DateHeure      |             |          |         | `DEFAULT NOW()`               |
| `date_modification`      | DateHeure      |             |          |         | `DEFAULT NOW()` + trigger     |

**Saisie dans Looping :**

1. Creer l'entite, la nommer `ARTICLE`
2. `id_article` → Numerique → Entier → **32 bits**, cocher **Identifiant**
3. `uuid` → Texte → caracteres unicode → variable → longueur `36`, cocher **UNIQUE**
4. `titre` → Texte → caracteres unicode → variable → longueur `500`, cocher **NOT NULL**
5. `url` → Texte → caracteres unicode → **volumineux**, cocher **NOT NULL** + **UNIQUE**
6. `contenu`, `resume`, `chemin_donnees_brutes` → Texte → caracteres unicode → **volumineux**
7. `auteur` → variable → longueur `200` ; `langue` → variable → longueur `10` ; `modele_classification` → variable → longueur `100`
8. `est_green_it` → Logique → Booleen
9. `score_confiance` → Numerique → Reel → **double 64 bits**
10. Proprietes temporelles → Temporel → DateHeure

---

#### Entite 3 : SEARCH_CONFIG

> Parametres dynamiques de recherche utilises par les modules de collecte.
> Correspond a la table `search_config` dans `init.sql`.

| Propriete           | Type Looping   | Identifiant | NOT NULL | UNIQUE | SQL uniquement (hors Looping) |
|---------------------|----------------|-------------|----------|--------|-------------------------------|
| `id_config`         | Entier         | **Oui**     |          |        |                               |
| `mot_cle`           | Variable(100)  |             | **Oui**  |        |                               |
| `url_source`        | Volumineux     |             |          |        |                               |
| `type_source`       | Variable(20)   |             |          |        | `CHECK IN ('api','scraping','file')` |
| `priorite`          | Entier         |             |          |        | `DEFAULT 1`                   |
| `actif`             | Booleen        |             |          |        | `DEFAULT true`                |
| `date_creation`     | DateHeure      |             |          |        | `DEFAULT NOW()`               |
| `date_modification` | DateHeure      |             |          |        | `DEFAULT NOW()` + trigger     |

**Saisie dans Looping :**

1. Creer l'entite, la nommer `SEARCH_CONFIG`
2. `id_config` → Numerique → Entier → **32 bits**, cocher **Identifiant**
3. `mot_cle` → Texte → caracteres unicode → variable → longueur `100`, cocher **NOT NULL**
4. `url_source` → Texte → caracteres unicode → **volumineux**
5. `type_source` → Texte → caracteres unicode → variable → longueur `20`
6. `priorite` → Numerique → Entier → **32 bits**
7. `actif` → Logique → Booleen
8. `date_creation`, `date_modification` → Temporel → DateHeure

---

#### Entite 4 : UTILISATEUR

> Utilisateur de l'application web (authentification JWT).
> Correspond a la table `users` dans `init.sql`.

| Propriete            | Type Looping    | Identifiant | NOT NULL | UNIQUE  | SQL uniquement (hors Looping) |
|----------------------|-----------------|-------------|----------|---------|-------------------------------|
| `id_utilisateur`     | Variable(36)    | **Oui**     |          |         | `DEFAULT uuid_generate_v4()`  |
| `email`              | Variable(320)   |             | **Oui**  | **Oui** |                               |
| `mot_de_passe_hash`  | Variable(1024)  |             | **Oui**  |         |                               |
| `est_actif`          | Booleen         |             |          |         | `DEFAULT true`                |
| `est_superuser`      | Booleen         |             |          |         | `DEFAULT false`               |
| `est_verifie`        | Booleen         |             |          |         | `DEFAULT false`               |
| `date_creation`      | DateHeure       |             |          |         | `DEFAULT NOW()`               |
| `date_modification`  | DateHeure       |             |          |         | `DEFAULT NOW()` + trigger     |

**Saisie dans Looping :**

1. Creer l'entite, la nommer `UTILISATEUR`
2. `id_utilisateur` → Texte → caracteres unicode → variable → longueur `36`, cocher **Identifiant**
3. `email` → Texte → caracteres unicode → variable → longueur `320`, cocher **NOT NULL** + **UNIQUE**
4. `mot_de_passe_hash` → Texte → caracteres unicode → variable → longueur `1024`, cocher **NOT NULL**
5. `est_actif`, `est_superuser`, `est_verifie` → Logique → Booleen
6. `date_creation`, `date_modification` → Temporel → DateHeure

---

#### Entite 5 : LOG_ANALYSE

> Trace d'une inference IA sur un article (modele, duree, emissions CO2).
> Correspond a la table `analysis_logs` dans `init.sql`.

| Propriete              | Type Looping   | Identifiant | NOT NULL | UNIQUE | SQL uniquement (hors Looping) |
|------------------------|----------------|-------------|----------|--------|-------------------------------|
| `id_log`               | Entier         | **Oui**     |          |        |                               |
| `nom_modele`           | Variable(100)  |             | **Oui**  |        |                               |
| `version_modele`       | Variable(50)   |             |          |        |                               |
| `temps_inference_ms`   | Entier         |             |          |        |                               |
| `emissions_carbone_kg` | Reel           |             |          |        |                               |
| `prediction`           | Booleen        |             |          |        |                               |
| `confiance`            | Reel           |             |          |        |                               |
| `date_creation`        | DateHeure      |             |          |        | `DEFAULT NOW()`               |

**Saisie dans Looping :**

1. Creer l'entite, la nommer `LOG_ANALYSE`
2. `id_log` → Numerique → Entier → **32 bits**, cocher **Identifiant**
3. `nom_modele` → Texte → caracteres unicode → variable → longueur `100`, cocher **NOT NULL**
4. `version_modele` → Texte → caracteres unicode → variable → longueur `50`
5. `temps_inference_ms` → Numerique → Entier → **32 bits**
6. `emissions_carbone_kg`, `confiance` → Numerique → Reel → **double 64 bits**
7. `prediction` → Logique → Booleen
8. `date_creation` → Temporel → DateHeure

---

#### Entite 6 : STATS_QUOTIDIENNES

> Statistiques agregees par jour pour le dashboard.
> Correspond a la table `daily_stats` dans `init.sql`.

| Propriete                | Type Looping   | Identifiant | NOT NULL | UNIQUE  | SQL uniquement (hors Looping) |
|--------------------------|----------------|-------------|----------|---------|-------------------------------|
| `id_stats`               | Entier         | **Oui**     |          |         |                               |
| `date_stat`              | Date           |             | **Oui**  | **Oui** |                               |
| `total_articles`         | Entier         |             |          |         | `DEFAULT 0`                   |
| `articles_green_it`      | Entier         |             |          |         | `DEFAULT 0`                   |
| `articles_non_green_it`  | Entier         |             |          |         | `DEFAULT 0`                   |
| `score_confiance_moyen`  | Reel           |             |          |         |                               |
| `articles_par_source`    | Volumineux     |             |          |         | `JSONB` en PostgreSQL         |
| `date_creation`          | DateHeure      |             |          |         | `DEFAULT NOW()`               |

**Saisie dans Looping :**

1. Creer l'entite, la nommer `STATS_QUOTIDIENNES`
2. `id_stats` → Numerique → Entier → **32 bits**, cocher **Identifiant**
3. `date_stat` → Temporel → **Date** (pas DateHeure), cocher **NOT NULL** + **UNIQUE**
4. `total_articles`, `articles_green_it`, `articles_non_green_it` → Numerique → Entier → **32 bits**
5. `score_confiance_moyen` → Numerique → Reel → **double 64 bits**
6. `articles_par_source` → Texte → caracteres unicode → **volumineux**
   (le type reel en PostgreSQL est `JSONB`, Looping ne le gere pas)
7. `date_creation` → Temporel → DateHeure

---

### 8.4 Creation des associations

Pour creer une association dans Looping :

1. Selectionner l'outil **Association** (icone losange) dans la barre d'outils
2. Cliquer sur la premiere entite
3. Cliquer sur la seconde entite
4. Un losange apparait entre les deux : double-cliquer dessus pour le nommer
5. Regler les cardinalites en double-cliquant sur chaque extremite du lien

---

#### Association 1 : produit (SOURCE → ARTICLE)

> Une source produit des articles. Un article provient d'une seule source
> (ou d'aucune si la source est supprimee : `ON DELETE SET NULL`).

| Propriete      | Valeur                                                       |
|----------------|--------------------------------------------------------------|
| **Nom**        | `produit`                                                    |
| **Entite 1**   | `SOURCE`                                                     |
| **Entite 2**   | `ARTICLE`                                                    |
| **Cardinalite SOURCE**  | **(1,n)** — une source produit au minimum 1 article |
| **Cardinalite ARTICLE** | **(0,1)** — un article provient de 0 ou 1 source   |

**Saisie dans Looping :**

1. Outil association → cliquer `SOURCE` puis `ARTICLE`
2. Nommer le losange `produit`
3. Cote `SOURCE` : saisir **1,n**
4. Cote `ARTICLE` : saisir **0,1**

> Justification du (0,1) cote ARTICLE : dans le SQL, `source_id` est nullable
> avec `ON DELETE SET NULL`. Si une source est supprimee, l'article reste en
> base avec `source_id = NULL`, soit 0 source associee.

---

#### Association 2 : genere (ARTICLE → LOG_ANALYSE)

> Un article peut etre analyse plusieurs fois (differents modeles, versions).
> Chaque log d'analyse concerne exactement un article.

| Propriete      | Valeur                                                       |
|----------------|--------------------------------------------------------------|
| **Nom**        | `genere`                                                     |
| **Entite 1**   | `ARTICLE`                                                    |
| **Entite 2**   | `LOG_ANALYSE`                                                |
| **Cardinalite ARTICLE**     | **(0,n)** — un article peut avoir 0 ou N analyses |
| **Cardinalite LOG_ANALYSE** | **(1,1)** — un log concerne exactement 1 article  |

**Saisie dans Looping :**

1. Outil association → cliquer `ARTICLE` puis `LOG_ANALYSE`
2. Nommer le losange `genere`
3. Cote `ARTICLE` : saisir **0,n**
4. Cote `LOG_ANALYSE` : saisir **1,1**

> Justification du (1,1) cote LOG_ANALYSE : dans le SQL, `article_id` est
> `NOT NULL` avec `ON DELETE CASCADE`. Un log ne peut pas exister sans
> article, et la suppression de l'article entraine la suppression du log.

---

#### Association 3 : effectue (UTILISATEUR → LOG_ANALYSE)

> Un utilisateur peut declencher des analyses IA. Un log peut etre associe
> a un utilisateur (analyse manuelle) ou non (analyse batch automatique).

| Propriete      | Valeur                                                       |
|----------------|--------------------------------------------------------------|
| **Nom**        | `effectue`                                                   |
| **Entite 1**   | `UTILISATEUR`                                                |
| **Entite 2**   | `LOG_ANALYSE`                                                |
| **Cardinalite UTILISATEUR** | **(0,n)** — un utilisateur peut effectuer 0 ou N analyses |
| **Cardinalite LOG_ANALYSE** | **(0,1)** — un log est lie a 0 ou 1 utilisateur          |

**Saisie dans Looping :**

1. Outil association → cliquer `UTILISATEUR` puis `LOG_ANALYSE`
2. Nommer le losange `effectue`
3. Cote `UTILISATEUR` : saisir **0,n**
4. Cote `LOG_ANALYSE` : saisir **0,1**

> Justification du (0,1) cote LOG_ANALYSE : un log peut exister sans
> utilisateur si l'analyse est declenchee automatiquement (batch/cron).
> Cette FK (`user_id`) n'existe pas encore dans `init.sql` et sera ajoutee
> lors du developpement de l'authentification (Etape 4).

---

### 8.5 Entites isolees (sans association)

Deux entites n'ont **aucune association directe** avec les autres :

| Entite                | Raison de l'isolement                                        |
|-----------------------|--------------------------------------------------------------|
| **SEARCH_CONFIG**     | Table de configuration autonome. Lue par les scripts Python de collecte mais sans cle etrangere vers/depuis les autres tables. |
| **STATS_QUOTIDIENNES** | Table d'agregation autonome. Les donnees sont calculees a partir des articles, mais aucune FK directe n'est definie (les statistiques sont pre-calculees). |

**Dans Looping** : laisser ces entites positionnees a l'ecart du groupe
principal. C'est tout a fait normal en Merise — ce sont des entites
independantes. On peut les regrouper visuellement dans une zone "Configuration"
et "Analytics".

---

### 8.6 Disposition recommandee du MCD

```
  ┌──────────────────┐                              ┌──────────────────────┐
  │  SEARCH_CONFIG   │                              │  STATS_QUOTIDIENNES  │
  │  (config)        │                              │  (analytics)         │
  └──────────────────┘                              └──────────────────────┘



       ┌──────────┐         produit          ┌────────────┐
       │  SOURCE  │─────────────────────────│  ARTICLE   │
       │          │  (1,n)         (0,1)     │            │
       └──────────┘                          └─────┬──────┘
                                                   │
                                              genere│(0,n)
                                                   │
                                            ┌──────┴───────┐
                                            │ LOG_ANALYSE  │
                                            │              │
                                            └──────┬───────┘
                                                   │(0,1)
                                           effectue│
                                                   │(0,n)
                                           ┌───────┴────────┐
                                           │  UTILISATEUR   │
                                           │                │
                                           └────────────────┘
```

> Conseil : dans Looping, disposer les entites de haut en bas selon le flux
> de donnees (source → article → analyse → utilisateur). Placer les entites
> isolees dans les coins superieurs.

---

### 8.7 Generation du MLD (passage automatique)

#### Procedure dans Looping

1. S'assurer que le MCD est complet et sauvegarde
2. Cliquer sur l'onglet **MLD** (ou menu **Affichage → Modele Logique**)
3. Looping applique automatiquement les regles de passage Merise

#### Regles de passage appliquees

Looping transforme les associations selon les cardinalites :

| Cardinalite cote entite | Regle de passage                                             |
|-------------------------|--------------------------------------------------------------|
| **(x,1)**               | La cle primaire de l'autre entite **migre** en tant que cle etrangere dans cette entite |
| **(x,n) — (x,n)**      | Creation d'une **table de jonction** (pas le cas ici)        |

Pour notre modele :

- `produit` : cardinalite (0,1) cote ARTICLE → `id_source` migre dans ARTICLE comme FK
- `genere` : cardinalite (1,1) cote LOG_ANALYSE → `id_article` migre dans LOG_ANALYSE comme FK
- `effectue` : cardinalite (0,1) cote LOG_ANALYSE → `id_utilisateur` migre dans LOG_ANALYSE comme FK

---

### 8.8 Resultat attendu du MLD

Apres la generation automatique, Looping doit produire les 6 tables suivantes.
Les cles etrangeres sont prefixees par `#` dans Looping.

#### Table SOURCE

```
SOURCE (id_source PK, nom, type, url_base, description, est_active,
        derniere_collecte, date_creation)
```

Aucune cle etrangere.

#### Table ARTICLE

```
ARTICLE (id_article PK, uuid, titre, url, contenu, resume, auteur,
         date_publication, langue, est_green_it, score_confiance,
         modele_classification, chemin_donnees_brutes, date_analyse,
         date_creation, date_modification,
         #id_source FK → SOURCE)
```

`#id_source` migre depuis l'association `produit` (cardinalite 0,1 cote ARTICLE).

#### Table LOG_ANALYSE

```
LOG_ANALYSE (id_log PK, nom_modele, version_modele, temps_inference_ms,
             emissions_carbone_kg, prediction, confiance, date_creation,
             #id_article FK → ARTICLE,
             #id_utilisateur FK → UTILISATEUR)
```

Deux cles etrangeres migrees :
- `#id_article` depuis l'association `genere` (cardinalite 1,1)
- `#id_utilisateur` depuis l'association `effectue` (cardinalite 0,1)

#### Table UTILISATEUR

```
UTILISATEUR (id_utilisateur PK, email, mot_de_passe_hash, est_actif,
             est_superuser, est_verifie, date_creation, date_modification)
```

Aucune cle etrangere.

#### Table SEARCH_CONFIG

```
SEARCH_CONFIG (id_config PK, mot_cle, url_source, type_source,
               priorite, actif, date_creation, date_modification)
```

Aucune cle etrangere (entite isolee).

#### Table STATS_QUOTIDIENNES

```
STATS_QUOTIDIENNES (id_stats PK, date_stat, total_articles,
                    articles_green_it, articles_non_green_it,
                    score_confiance_moyen, articles_par_source,
                    date_creation)
```

Aucune cle etrangere (entite isolee).

---

### 8.9 Verification du MLD

Apres generation, verifier dans Looping :

- [ ] `ARTICLE` contient `#id_source` (prefixe `#` = cle etrangere)
- [ ] `LOG_ANALYSE` contient `#id_article` ET `#id_utilisateur`
- [ ] Les cles etrangeres pointent vers les bonnes tables
- [ ] Les tables isolees (`SEARCH_CONFIG`, `STATS_QUOTIDIENNES`) n'ont aucune FK
- [ ] Chaque table conserve sa cle primaire d'origine

---

### 8.10 Correspondance MLD ↔ init.sql

| Table MLD (Looping)     | Table SQL (`init.sql`)  | Cle(s) etrangere(s) SQL                              |
|-------------------------|-------------------------|-------------------------------------------------------|
| `SOURCE`                | `sources`               | —                                                     |
| `ARTICLE`               | `articles`              | `source_id → sources(id) ON DELETE SET NULL`          |
| `LOG_ANALYSE`           | `analysis_logs`         | `article_id → articles(id) ON DELETE CASCADE`         |
| `UTILISATEUR`           | `users`                 | —                                                     |
| `SEARCH_CONFIG`         | `search_config`         | —                                                     |
| `STATS_QUOTIDIENNES`    | `daily_stats`           | —                                                     |

> **Note** : la FK `id_utilisateur` dans `LOG_ANALYSE` n'existe pas encore
> dans le `init.sql` actuel. Elle sera ajoutee lors du developpement de
> l'authentification (Etape 4). Le MCD l'anticipe correctement.

---

### 8.11 Generation du script SQL depuis Looping

1. Menu **Fichier → Generer le script SQL** (ou **Outils → SQL**)
2. Choisir le SGBD cible : **PostgreSQL**
3. Looping genere un script SQL de creation des tables

**Important** : le script genere par Looping est basique. Il faudra ajouter
manuellement dans `init.sql` les elements suivants que Looping ne gere pas :

| Element manquant                  | Exemple                                                |
|-----------------------------------|--------------------------------------------------------|
| Extension UUID                    | `CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`          |
| Contraintes CHECK                 | `CHECK (type IN ('api', 'scraping', 'file'))`          |
| Valeurs par defaut                | `DEFAULT NOW()`, `DEFAULT uuid_generate_v4()`          |
| Index de performance              | `CREATE INDEX idx_articles_is_green_it ON articles(...)` |
| Triggers `updated_at`             | Fonction `update_updated_at_column()` + triggers       |
| Vues SQL                          | `v_global_stats`, `v_recent_articles`                  |
| Type JSONB                        | `articles_by_source JSONB` (= `Volumineux` dans Looping) |
| Donnees de test (INSERT)          | Sources initiales et configuration de recherche        |

Le fichier `scripts/sql/init.sql` existant contient deja tous ces elements.
Le MCD/MLD sert de **modelisation visuelle** pour la documentation et la
soutenance, pas de remplacement du script SQL.

---

### 8.12 Export et sauvegarde

1. **Sauvegarder** le fichier Looping : **Fichier → Enregistrer** (`docs/greentech_intelligence.loo`)
2. **Exporter le MCD en image** : **Fichier → Exporter** → format PNG
   → sauvegarder sous `docs/images/mcd_greentech.png`
3. **Exporter le MLD en image** : basculer en vue MLD, puis exporter
   → sauvegarder sous `docs/images/mld_greentech.png`
4. Ces images pourront etre integrees dans la documentation Sphinx

---

### 8.13 Checklist de validation finale

| #  | Verification                                                    | Fait |
|----|-----------------------------------------------------------------|------|
| 1  | 6 entites creees avec toutes leurs proprietes                   | ☐    |
| 2  | Chaque entite a un identifiant souligne (cle primaire)          | ☐    |
| 3  | Les types Looping correspondent au tableau 8.1                  | ☐    |
| 4  | Association `produit` : SOURCE (1,n) — ARTICLE (0,1)           | ☐    |
| 5  | Association `genere` : ARTICLE (0,n) — LOG_ANALYSE (1,1)       | ☐    |
| 6  | Association `effectue` : UTILISATEUR (0,n) — LOG_ANALYSE (0,1) | ☐    |
| 7  | SEARCH_CONFIG et STATS_QUOTIDIENNES sont isolees (pas de lien) | ☐    |
| 8  | MLD genere avec les FK correctement migrees                     | ☐    |
| 9  | MLD coherent avec `scripts/sql/init.sql`                        | ☐    |
| 10 | MCD et MLD exportes en images PNG                               | ☐    |
| 11 | Fichier `.loo` sauvegarde dans `docs/`                          | ☐    |

---

## 9. Architecture & Workflow de stockage

### 9.1 Distinction : init.sql vs SQLAlchemy ORM

Le projet GreenTech Intelligence utilise **deux couches complémentaires** pour la gestion
des données. Cette architecture est une pratique standard dans les projets professionnels
Python/PostgreSQL.

---

#### 9.1.1 Couche 1 : `scripts/sql/init.sql` (SQL pur)

**Rôle** : Initialisation de la structure de la base de données

**Nature** : Script SQL classique exécuté directement par PostgreSQL

**Utilisation** : Exécuté **une seule fois** au déploiement initial ou lors du reset de la base

**Contenu** :
- Création des tables avec `CREATE TABLE`
- Définition des contraintes (`CHECK`, `UNIQUE`, `FOREIGN KEY`)
- Création des index de performance
- Création des triggers et fonctions PL/pgSQL
- Création des vues SQL
- Insertion des données de référence (sources, configuration initiale)
- Extension UUID (`uuid-ossp`)

**Avantages** :
- ✅ Crée toute la structure d'un coup (rapide, atomique)
- ✅ Permet des contraintes SQL avancées difficiles à exprimer en ORM
- ✅ Documentation complète et lisible du schéma
- ✅ Facilite le déploiement Docker (un seul fichier à monter)
- ✅ Indépendant du framework applicatif

**Commande d'exécution** :
```bash
# Via Docker
docker exec -i greentech-postgres psql -U greentech -d greentech_db < scripts/sql/init.sql

# Via docker-compose (au premier lancement)
# Le fichier est monté dans /docker-entrypoint-initdb.d/
```

---

#### 9.1.2 Couche 2 : `src/greentech/data/storage/models.py` (SQLAlchemy ORM 2.0)

**Rôle** : Interaction avec la base de données depuis le code Python applicatif

**Nature** : Classes Python mappées aux tables SQL via l'ORM SQLAlchemy

**Utilisation** : Quotidienne dans tout le code applicatif (collecte, API, IA)

**Contenu** :
- Classes Python représentant les tables (`Article`, `Source`, `User`, etc.)
- Relations entre objets (clés étrangères traduites en attributs Python)
- Validation automatique via type hints (`Mapped[str]`, `Mapped[int | None]`)
- Configuration de la connexion asynchrone (`AsyncSession`)

**Avantages** :
- ✅ Requêtes pythoniques et lisibles (pas de SQL brut dans le code)
- ✅ Requêtes asynchrones natives (`async`/`await`)
- ✅ Type safety avec mypy/pyright (auto-complétion IDE)
- ✅ Protection automatique contre les injections SQL
- ✅ Relations automatiques entre objets (ex: `article.source.nom`)
- ✅ Compatible avec Pydantic pour la validation

**Exemple de code** :

```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Boolean, Float, ForeignKey, Text
from datetime import datetime
from uuid import UUID

class Base(DeclarativeBase):
    pass

class Source(Base):
    __tablename__ = "sources"

    id_source: Mapped[int] = mapped_column(primary_key=True)
    nom: Mapped[str] = mapped_column(String(100), unique=True)
    type: Mapped[str] = mapped_column(String(20))
    url_base: Mapped[str | None] = mapped_column(Text, nullable=True)
    est_active: Mapped[bool] = mapped_column(default=True)

    # Relation : une source a plusieurs articles
    articles: Mapped[list["Article"]] = relationship(back_populates="source")

class Article(Base):
    __tablename__ = "articles"

    id_article: Mapped[int] = mapped_column(primary_key=True)
    uuid: Mapped[UUID] = mapped_column(unique=True)
    id_source: Mapped[int | None] = mapped_column(ForeignKey("sources.id_source"))
    titre: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(Text, unique=True)
    est_green_it: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score_confiance: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relation inverse
    source: Mapped["Source"] = relationship(back_populates="articles")
```

**Utilisation dans le code** :

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

async def get_green_articles(db: AsyncSession, limit: int = 10):
    """Récupère les articles Green IT depuis la base."""
    result = await db.execute(
        select(Article)
        .where(Article.est_green_it == True)
        .order_by(Article.date_publication.desc())
        .limit(limit)
    )
    return result.scalars().all()

# Utilisation
articles = await get_green_articles(db_session)
for article in articles:
    print(f"{article.titre} - Source: {article.source.nom}")
```

---

#### 9.1.3 Workflow complet

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 : INITIALISATION (une fois)                            │
│  ─────────────────────────────────────────────────────────────  │
│  $ docker-compose up -d postgres                                │
│  → init.sql est exécuté automatiquement                         │
│  → Tables, index, triggers, vues créés                          │
│  → Données de référence insérées                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 : DÉVELOPPEMENT (quotidien)                            │
│  ─────────────────────────────────────────────────────────────  │
│  Le code Python utilise SQLAlchemy ORM                           │
│  → models.py définit les classes Article, Source, User, etc.    │
│  → Le code interagit via l'ORM (jamais de SQL brut)             │
│  → Les requêtes sont asynchrones et type-safe                   │
└─────────────────────────────────────────────────────────────────┘
```

---

#### 9.1.4 Quand utiliser l'un ou l'autre ?

| Cas d'usage                              | Outil recommandé       |
|------------------------------------------|------------------------|
| Créer la structure initiale de la BDD    | `init.sql`             |
| Ajouter une contrainte CHECK complexe    | `init.sql`             |
| Créer un trigger ou une fonction PL/pgSQL| `init.sql`             |
| Créer une vue SQL                        | `init.sql`             |
| Insérer les données de référence         | `init.sql`             |
| Faire une requête SELECT dans le code    | SQLAlchemy ORM         |
| Insérer/modifier des données depuis l'API| SQLAlchemy ORM         |
| Définir les relations entre entités      | SQLAlchemy ORM         |
| Faire évoluer le schéma (migrations)     | Alembic + SQLAlchemy   |

---

#### 9.1.5 Évolution future : Alembic (optionnel)

Si le schéma de la base évolue après le déploiement initial (ajout de colonnes,
modification de contraintes), il est recommandé d'utiliser **Alembic** (outil de
migrations SQLAlchemy) pour gérer les changements de manière versionnée.

**Exemple de migration Alembic** :
```bash
# Générer automatiquement une migration
alembic revision --autogenerate -m "Ajout colonne score_carbone"

# Appliquer la migration
alembic upgrade head
```

Alembic détecte les différences entre `models.py` et le schéma actuel de la base,
puis génère automatiquement le script SQL de migration.

---

### 9.2 Localisation des fichiers

| Fichier                                  | Rôle                                      |
|------------------------------------------|-------------------------------------------|
| `scripts/sql/init.sql`                   | Initialisation de la structure SQL        |
| `src/greentech/data/storage/models.py`   | Modèles SQLAlchemy ORM                    |
| `src/greentech/data/storage/database.py` | Configuration de la connexion async       |
| `src/greentech/config.py`                | Settings (URL de connexion via `.env`)    |

---

### 9.3 Conformité du schéma SQL avec le MCD/MLD

Le fichier `scripts/sql/init.sql` **doit être strictement conforme** au MLD généré
dans Looping (section 8). Les noms de tables et de colonnes utilisent la nomenclature
**française** définie dans le MCD :

- `id_article`, `id_source`, `id_utilisateur` (et non `id`, `user_id`)
- `titre`, `contenu`, `resume` (et non `title`, `content`, `summary`)
- `est_green_it`, `date_publication` (et non `is_green_it`, `published_at`)

Les modèles SQLAlchemy (`models.py`) doivent **refléter exactement** ces noms pour
garantir la cohérence entre le schéma SQL et le code Python.

**Note importante** : Cette nomenclature française est un choix de conception assumé
pour ce projet francophone académique. Dans un contexte international, l'anglais
serait privilégié.

---

## 10. Documentation technique du pipeline Data (Implementation)

>
> Cette section documente la logique algorithmique de chaque script,
> les choix d'implementation et les commandes pour executer le pipeline complet.

---

### 10.1 Vue d'ensemble des scripts

Le pipeline Data (Bloc E1) est compose de 6 modules Python executables
independamment ou en sequence :

```
 COLLECTE                    NETTOYAGE               INGESTION SQL
 ────────                    ─────────               ─────────────
 Module 0                    Module 4                Module 5
 search_config (SQL)         spark_cleaner.py        sql_ingester.py
       │                          │                       │
       ▼                          ▼                       ▼
 Module 1  ──► raw-data    raw-data ──► clean-data   clean-data ──► PostgreSQL
 api_collector.py                (Parquet)               (articles, sources)
       │
 Module 2  ──► raw-data
 scraper.py
       │
 Module 3  ──► raw-data
 file_ingester.py
```

| Module | Fichier | Dependance | Entree | Sortie |
|--------|---------|-----------|--------|--------|
| 0 | `data/collectors/base.py` | PostgreSQL | Table `search_config` | Liste de mots-cles |
| 1 | `data/collectors/api_collector.py` | Module 0, MinIO | API NewsData.io | MinIO `raw-data/api/` |
| 2 | `data/collectors/scraper.py` | MinIO | TechCrunch HTML | MinIO `raw-data/scraping/` |
| 3 | `data/collectors/file_ingester.py` | MinIO | Fichier arXiv local | MinIO `raw-data/file/` |
| 4 | `data/processors/spark_cleaner.py` | MinIO (S3A) | MinIO `raw-data/` | MinIO `clean-data/articles/` (Parquet) |
| 5 | `data/storage/sql_ingester.py` | MinIO, PostgreSQL | MinIO `clean-data/articles/` | Table `articles` PostgreSQL |

---

### 10.2 Module 0 : Configuration dynamique SQL

**Fichier** : `src/greentech/data/collectors/base.py`

**Fonction cle** : `get_config_from_db(session, type_source, actif_seulement)`

**Algorithme** :

1. Connexion async a PostgreSQL via SQLAlchemy
2. Requete `SELECT` sur la table `search_config` avec filtres optionnels
3. Tri par priorite ascendante (priorite 1 = plus important)
4. Retour d'une liste d'objets `SearchConfig` (ORM)

**Requete SQL generee** :

```sql
SELECT search_config.id_config, search_config.mot_cle,
       search_config.url_source, search_config.type_source,
       search_config.priorite, search_config.actif
FROM search_config
WHERE search_config.type_source = :type_source
  AND search_config.actif = true
ORDER BY search_config.priorite
```

**Choix de conception** : La configuration est stockee en base (et non dans un
fichier YAML) pour permettre la modification a chaud sans redemarrage. Cela
valide la competence "Extraction depuis un SGBD via requete SQL".

---

### 10.3 Module 1 : Collecte API (httpx)

**Fichier** : `src/greentech/data/collectors/api_collector.py`

**Classe** : `ApiCollector(BaseCollector)`

**Algorithme** :

1. Charger les mots-cles depuis `search_config` (Module 0, filtre `type_source='api'`)
2. Pour chaque mot-cle :
   a. Construire la requete HTTP GET vers `newsdata.io/api/1/latest`
   b. Parametres : `apikey`, `q=<mot_cle>`, `category=technology`, `language=en`, `size=10`
   c. Envoyer via httpx (client async, timeout 30s)
   d. Verifier `status="success"` dans la reponse JSON
   e. Parser les articles : extraire `title`, `link`, `content`, `pubDate`, `creator`, `source_id`
   f. Filtrer les articles incomplets (sans titre ou URL)
   g. Sauvegarder le lot JSON dans MinIO `raw-data/api/newsdata/<date>/<timestamp>.json`
3. Retourner le `CollectResult` (nombre d'articles, chemins MinIO, erreurs)

**Gestion des erreurs** :
- Codes HTTP 4xx/5xx : capture et log, passage au mot-cle suivant
- Cle API manquante : avertissement, arret immediat sans erreur fatale
- Timeout : httpx leve une exception capturee par le try/except

**Choix de conception** :
- httpx plutot que requests : support natif de `async/await`
- 10 articles par requete : limite du plan gratuit NewsData.io
- Sauvegarde brute dans MinIO avant nettoyage : architecture Data Lakehouse

---

### 10.4 Module 2 : Scraping hybride RSS + Scrapy/Playwright

**Fichier** : `src/greentech/data/collectors/scraper.py`

**Classes** : `TechCrunchArticleSpider(Spider)` + `ScrapingCollector(BaseCollector)`

**Algorithme en deux etapes** :

**Etape 1 - Decouverte d'URLs via RSS** (`_discover_urls_via_rss`) :
1. Requete HTTP GET asynchrone vers `https://techcrunch.com/category/climate/feed/`
2. Parser la reponse XML via `feedparser.parse()`
3. Extraire `entry.link` pour les 20 premieres entrees (filtrage mots-cles optionnel)
4. Retourner la liste d'URLs a scraper

**Etape 2 - Scraping HTML via Scrapy + Playwright** (`_scrape_html_pages`) :
1. Instancier `TechCrunchArticleSpider(urls=[...])` avec la liste d'URLs du RSS
2. Configurer Scrapy avec `scrapy-playwright` (handler + reactor asyncio)
3. Pour chaque URL :
   a. Requete Scrapy avec `playwright=True`
   b. Playwright charge la page et attend le selecteur `h1` (timeout 15s)
   c. Callback `parse_article` applique les selecteurs CSS sur le DOM rendu
   d. Extraction : titre, date, auteurs, contenu texte, HTML brut, og:description
   e. Validation : on ignore les articles sans titre ou sans contenu
4. Collecter les articles dans `spider.collected_articles`
5. Sauvegarder le lot JSON dans MinIO `raw-data/scraping/techcrunch/<date>/<timestamp>.json`

**Selecteurs CSS appliques** :
- Titre : `h1.article-hero__title::text` (fallback `og:title`)
- Date : `time::attr(datetime)` (ISO 8601)
- Auteurs : `a[href*="/author/"]::text` (concatenes)
- Contenu : `div.entry-content p::text` (paragraphes fusionnes)
- HTML brut : `div.entry-content` (bloc complet)
- Resume : `meta[property="og:description"]::attr(content)`

**Contraintes techniques respectees** :
- `ROBOTSTXT_OBEY = True` : respect du fichier robots.txt
- `DOWNLOAD_DELAY = 2.0` : delai minimum 2s entre chaque page
- `CONCURRENT_REQUESTS = 1` : une seule requete simultanee
- `User-Agent` : `GreenTech-Bot/1.0` (identifie et transparent)
- Gestion propre des erreurs par URL (`errback`) : une page cassee n'interrompt pas les autres

**Pourquoi Playwright est necessaire** : TechCrunch utilise du chargement
dynamique JavaScript (React hydration) sur les pages d'articles. Un scraper
HTTP classique recupererait uniquement le squelette sans le contenu rendu.

**Pourquoi l'architecture hybride RSS + HTML** : TechCrunch a refondu sa page
d'index en avril 2026 (suppression de la balise `<article>` utilisee comme
selecteur sur la page de listing). Utiliser le flux RSS officiel comme source
d'URLs rend la collecte robuste aux refontes futures de l'index, tandis que
le scraping HTML de **chaque page d'article individuel** (qui garde une
structure stable : `h1.article-hero__title`, `div.entry-content`) reste le
coeur de la collecte et satisfait le critere de certification C1.

---

### 10.5 Module 3 : Ingestion fichier (arXiv Dataset)

**Fichier** : `src/greentech/data/collectors/file_ingester.py`

**Classe** : `FileIngester(BaseCollector)`

**Algorithme** :

1. Ouvrir le fichier arXiv (~3.6 Go, format JSON Lines)
2. Lire ligne par ligne (lecture incrementale pour ne pas saturer la RAM) :
   a. Parser chaque ligne en JSON
   b. Filtrer par categories arXiv pertinentes : `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.SE`
   c. Si des mots-cles sont fournis, filtrer par presence dans titre + abstract
   d. Normaliser : titre (retrait `\n`), URL (`https://arxiv.org/abs/{id}`), auteur, date
   e. Fixer `source_nom = "arXiv Dataset"`
3. Arreter apres `MAX_ARTICLES = 5000` articles retenus
4. Upload par batchs de 500 vers MinIO `raw-data/file/arxiv_batch_NNN/<date>/<timestamp>.json`

**Choix de conception** :
- Lecture ligne par ligne : le fichier de 3.6 Go ne tient pas en memoire
- Filtrage en 2 etapes (categories puis mots-cles) : reduit le volume rapidement
- Batchs de 500 : equilibre entre nombre de requetes MinIO et taille par fichier

---

### 10.6 Module 4 : Nettoyage Big Data (PySpark)

**Fichier** : `src/greentech/data/processors/spark_cleaner.py`

**Fonction principale** : `run_spark_cleaning()`

**Algorithme** :

1. **Initialisation Spark** :
   - Session locale (`local[*]`) avec connecteur S3A vers MinIO
   - JARs Hadoop-AWS 3.4.2 + AWS SDK 1.12.367 charges automatiquement
   - Configuration : `path.style.access=true`, SSL desactive (dev local)

2. **Lecture des 3 sources depuis MinIO** (lecture recursive JSON via S3A) :
   - `s3a://raw-data/api/newsdata/` → `clean_api_data()`
   - `s3a://raw-data/scraping/techcrunch/` → `clean_scraping_data()`
   - `s3a://raw-data/file/arxiv_dataset/` → `clean_file_data()`
   - Chaque fonction extrait les champs du JSON imbrique (`articles[]`)
     vers un schema unifie : `titre, url, contenu, auteur, date_publication, source_nom, langue`

3. **Agregation** : `unionByName(allowMissingColumns=True)` des 3 DataFrames

4. **Pipeline de nettoyage** (`apply_cleaning_pipeline()`) :

   | Etape | Transformation | Implementation |
   |-------|---------------|----------------|
   | 1 | Suppression balises HTML | `regexp_replace(col, "<[^>]+>", " ")` + entites HTML |
   | 2 | Anonymisation auteurs (RGPD) | Split par virgule → initiales de chaque mot → `"J.D., M.L."` |
   | 3 | Normalisation dates ISO 8601 | Detection du format (`rlike`) puis conversion en `YYYY-MM-DDTHH:MM:SSZ` |
   | 4 | Suppression entrees corrompues | `filter(titre IS NOT NULL AND url IS NOT NULL)` |
   | 5 | Suppression doublons | `dropDuplicates(["url"])` |

   **Note** : Toutes les transformations utilisent des **fonctions Spark SQL natives**
   (pas de UDFs Python), ce qui est plus performant et compatible Windows.

5. **Sauvegarde** : Ecriture en Parquet dans `s3a://clean-data/articles/`
   (mode `overwrite`)

**Schema Parquet de sortie** :

| Colonne | Type | Nullable | Description |
|---------|------|----------|-------------|
| `titre` | string | Non | Titre nettoye (sans HTML) |
| `url` | string | Non | URL unique de l'article |
| `contenu` | string | Oui | Contenu texte nettoye |
| `auteur` | string | Oui | Initiales RGPD (ex: "J.D.") ou "Auteur anonyme" |
| `date_publication` | string | Oui | Date ISO 8601 (ex: "2026-03-08T14:30:00Z") |
| `source_nom` | string | Oui | Nom de la source d'origine |
| `langue` | string | Oui | Code langue (defaut: "en") |

---

### 10.7 Module 5 : Ingestion SQL (SQLAlchemy async)

**Fichier** : `src/greentech/data/storage/sql_ingester.py`

**Fonction principale** : `run_sql_ingestion()`

**Algorithme** :

1. **Lecture des Parquet depuis MinIO** (`read_parquet_from_minio()`) :
   - Lister les objets dans `clean-data/articles/` (lecture recursive)
   - Filtrer les fichiers `.parquet` (Spark genere aussi des fichiers `_SUCCESS`, `_metadata`)
   - Telecharger chaque partition via le client MinIO
   - Lire avec PyArrow (`pq.read_table()`)
   - Convertir en liste de dictionnaires Python

2. **Resolution des sources** (`_resolve_source_name()`) :

   Le champ `source_nom` dans le Parquet peut contenir :
   - Des identifiants de medias depuis NewsData.io (ex: `"bbc_news"`, `"techradar"`)
   - `"TechCrunch Climate"` (fixe depuis le scraper)
   - `"arXiv Dataset"` (fixe depuis l'ingestion fichier)

   Logique de resolution vers les sources PostgreSQL :

   ```
   source_nom exact dans le mapping   → nom canonique
   source_nom contient "techcrunch"   → "TechCrunch Climate"
   source_nom contient "arxiv"        → "arXiv Dataset"
   sinon (identifiant media inconnu)  → "NewsData.io" (defaut)
   ```

   Le cache `{nom: id_source}` est charge une seule fois au debut.
   Si une source n'existe pas en base, elle est creee automatiquement.

3. **Insertion par batchs** (`ingest_to_postgresql()`) :
   - Batchs de 50 articles avec commit apres chaque batch
   - Pour chaque article :
     a. Resoudre `source_nom` → `id_source`
     b. Parser la date ISO 8601 en `datetime` Python (timezone-aware)
     c. Tronquer titre a 500 caracteres, auteur a 200 caracteres
     d. Executer `INSERT ... ON CONFLICT (url) DO NOTHING`
   - Comptabiliser : inseres, ignores (doublons), erreurs

   **Requete SQL generee** :

   ```text
   INSERT INTO articles (uuid, id_source, titre, url, contenu, auteur,
                          date_publication, langue, chemin_donnees_brutes)
   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
   ON CONFLICT (url) DO NOTHING
   RETURNING articles.id_article
   ```

   **Pourquoi ON CONFLICT DO NOTHING** : L'URL est une contrainte UNIQUE dans
   la table `articles`. Si un article avec la meme URL existe deja, l'insertion
   est silencieusement ignoree. Cela garantit l'**idempotence** du script :
   relancer l'ingestion ne cree jamais de doublons.

4. **Mise a jour des sources** :
   - Apres l'insertion, `derniere_collecte = NOW()` pour chaque source utilisee

5. **Verification post-ingestion** (`verify_ingestion()`) :

   ```sql
   -- Total articles
   SELECT COUNT(*) FROM articles;

   -- Articles par source
   SELECT sources.nom, sources.type, COUNT(articles.id_article)
   FROM sources LEFT OUTER JOIN articles ON articles.id_source = sources.id_source
   GROUP BY sources.nom, sources.type
   ORDER BY COUNT(articles.id_article) DESC;

   -- En attente d'analyse IA
   SELECT COUNT(*) FROM articles WHERE est_green_it IS NULL;
   ```

---

### 10.8 Commandes d'execution

**Prerequis** : Docker (PostgreSQL + MinIO) doit etre lance.

```bash
# 1. Lancer l'infrastructure
docker-compose up -d postgres minio

# 2. Verifier l'infrastructure
uv run python scripts/verify_infrastructure.py

# 3. Collecte des donnees brutes (3 modules independants)
# Module 1 : API (necessite API_NEWS_KEY dans .env)
uv run python -m greentech.data.collectors.api_collector

# Module 2 : Scraping hybride TechCrunch (RSS + Scrapy/Playwright)
# Necessite playwright install chromium la premiere fois
uv run python -m greentech.data.collectors.scraper

# Module 3 : Fichier (necessite le fichier arXiv telecharge)
uv run python -m greentech.data.collectors.file_ingester <chemin_fichier_arxiv>

# 4. Nettoyage Big Data (Spark)
uv run python -m greentech.data.processors.spark_cleaner

# 5. Ingestion dans PostgreSQL
uv run python -m greentech.data.storage.sql_ingester
```

**Pipeline complet en une commande** (si toutes les sources sont configurees) :

```bash
uv run python -m greentech.data.collectors.api_collector && \
uv run python -m greentech.data.processors.spark_cleaner && \
uv run python -m greentech.data.storage.sql_ingester
```

**Donnees de test** (sans API ni scraping) :

```bash
# Generer des donnees de test dans MinIO clean-data
uv run python scripts/seed_test_data.py

# Ingerer les donnees de test dans PostgreSQL
uv run python -m greentech.data.storage.sql_ingester
```

---

### 10.9 Dependances du pipeline

| Module | Dependances Python | Services externes |
|--------|-------------------|-------------------|
| Module 0 | sqlalchemy, asyncpg | PostgreSQL |
| Module 1 | httpx | PostgreSQL, MinIO, API NewsData.io |
| Module 2 | httpx, feedparser, scrapy, scrapy-playwright, playwright | MinIO |
| Module 3 | (stdlib uniquement) | MinIO |
| Module 4 | pyspark, hadoop-aws | MinIO (via S3A) |
| Module 5 | pyarrow, sqlalchemy, asyncpg | MinIO, PostgreSQL |

**Variables d'environnement requises** (fichier `.env`) :

| Variable | Module(s) | Description |
|----------|-----------|-------------|
| `POSTGRES_HOST` | 0, 1, 5 | Hote PostgreSQL (defaut: `localhost`) |
| `POSTGRES_PORT` | 0, 1, 5 | Port PostgreSQL (defaut: `5432`) |
| `POSTGRES_APP_USER` | 0, 1, 5 | Utilisateur applicatif |
| `POSTGRES_APP_PASSWORD` | 0, 1, 5 | Mot de passe applicatif |
| `MINIO_ENDPOINT` | 1-5 | Endpoint MinIO (defaut: `localhost:9000`) |
| `MINIO_ACCESS_KEY` | 1-5 | Cle d'acces MinIO |
| `MINIO_SECRET_KEY` | 1-5 | Cle secrete MinIO |
| `API_NEWS_KEY` | 1 | Cle API NewsData.io |

---

### 10.10 Choix techniques et justifications

| Choix | Justification |
|-------|---------------|
| **Parquet** (et non CSV/JSON) | Format colonnaire compresse, lecture rapide, schema type, standard Big Data |
| **ON CONFLICT DO NOTHING** (et non DO UPDATE) | Idempotence sans ecrasement des analyses IA deja effectuees |
| **Batchs de 50** | Compromis entre nombre de commits (performance) et taille de transaction (memoire) |
| **PyArrow** (et non PySpark pour la lecture) | Plus leger qu'une session Spark pour simplement lire des Parquet |
| **Mapping source_nom dynamique** | Les source_id de NewsData.io sont imprevisibles, le mapping gere les cas inconnus |
| **Fonctions Spark SQL natives** (et non UDFs Python) | Pas de serialisation Python-JVM, compatible Windows, plus performant |
| **Async SQLAlchemy** | Coherent avec le reste de la stack (FastAPI, httpx), non-bloquant |

---

## 11. Pipeline de classification hybride Green IT (Implementation)

> Cette section documente l'architecture de classification hybride en deux
> etages utilisee pour etiqueter chaque article comme Green IT ou non, ainsi
> que le mecanisme de fallback local du LLM judge lorsque le quota mensuel
> Hugging Face Inference Providers est epuise.

---

### 11.1 Vue d'ensemble

La classification d'un corpus d'articles heterogenes combine deux modeles
complementaires afin d'obtenir un bon equilibre recall / precision sans
saturer le quota cloud :

```
 ETAGE 1 (rapide, gratuit)          ETAGE 2 (precis, LLM)
 ────────────────────────           ──────────────────────
 auto_annotate_dataset.py     ──►   classify_candidates.py
 scoring mots-cles ponderes         LLM judge Qwen2.5-7B
       │                                   │
       ▼                                   ▼
 NON_GREEN direct ou CANDIDATE        est_green_it = True / False
 (6022 articles -> 95% NON_GREEN)     (verdict + score + raison)
```

| Etage | Module | Entree | Sortie |
|-------|--------|--------|--------|
| 1 | `scripts/auto_annotate_dataset.py` | Articles `modele_classification IS NULL` | DB : `est_green_it=false` (direct) ou `NULL` (candidat LLM) |
| 2 | `scripts/classify_candidates.py` + `src/greentech/ai/services/classifier_llm.py` | Articles `modele_classification='keyword_filter'` et `est_green_it IS NULL` | DB : `est_green_it=true/false`, `modele_classification='keyword_filter+qwen_llm_judge'` |

---

### 11.2 Etage 1 : pre-filtre mots-cles permissif

**Fichier** : `scripts/auto_annotate_dataset.py`

**Fonction cle** : `prefilter_article(titre, contenu, source_nom) -> PrefilterResult`

**Algorithme** :

1. Normalisation du titre et du contenu (minuscules, tirets → espaces).
2. Calcul de deux scores :
   - `green_score` = somme ponderee des matches sur ~120 indicateurs Green IT (poids 1.0 a 5.0) ;
   - `non_green_score` = somme ponderee des matches sur ~60 indicateurs negatifs.
3. Ajustements contextuels par source (ex : bonus energie/carbone pour arXiv).
4. Penalite sur les rapports boursiers et cryptomonnaies (signal anti-Green tres fort).
5. Decision binaire :

   - `CANDIDATE` si `green_score >= CANDIDATE_MIN_GREEN_SCORE` (0.1) **ET** `non_green_score < green_score + NON_GREEN_SCORE_THRESHOLD` (12.0) → verification LLM requise.
   - `NON_GREEN` sinon (rejet direct, economise un appel LLM).

**Bulk update** : les decisions sont ecrites en base en deux UPDATE
massifs (un pour NON_GREEN, un pour CANDIDATE), ce qui evite l'ouverture
de milliers de transactions individuelles.

**Metriques observees** (corpus 6022 articles) :
- ~15% marques CANDIDATE (envoyes au LLM).
- ~85% marques NON_GREEN directement (pas de cout LLM).

---

### 11.3 Etage 2 : LLM judge Qwen2.5-7B avec fallback local

**Fichiers** :
- `src/greentech/ai/services/classifier_llm.py` : interface metier (prompt, parsing)
- `src/greentech/ai/services/llm_dispatcher.py` : routage HF / local
- `src/greentech/ai/services/llm_local.py` : client d'inference locale ROCm
- `scripts/classify_candidates.py` : orchestrateur batch

**Prompt system** : definition permissive du Green IT avec regles de
decision explicites ("en cas de doute raisonnable, classer Green IT").
La reponse est contrainte a un objet JSON
`{"est_green_it": bool, "confiance": 0-1, "raison": str}`.

**Parser robuste** en trois passes pour encaisser les derives de
formatage des LLM :

1. `json.loads` direct sur la premiere region `{...}`.
2. Nettoyage des backslashes non conformes a la spec JSON (`\\escape` -> `\\\\escape`).
3. Extraction regex des trois champs individuellement si le JSON reste invalide.

**Retry avec backoff exponentiel** : chaque appel est retente jusqu'a
3 fois avec un delai de 2^n secondes en cas d'erreur transitoire.

---

### 11.4 Dispatcher de secours HF → Qwen local

**Fichier** : `src/greentech/ai/services/llm_dispatcher.py`

**Comportement** :

1. Tant que l'etat de session `_hf_quota_exhausted` est `False`, chaque appel
   `chat_completion` passe par `huggingface_hub.AsyncInferenceClient` contre
   l'API Hugging Face Serverless Inference.
2. A la premiere exception HTTP 402 (`Payment Required`, quota mensuel
   epuise), `mark_hf_quota_exhausted()` passe le flag a `True` et **la requete
   en cours est reessayee immediatement** sur le modele local.
3. Tous les appels suivants (classification + resumes) basculent directement
   sur le client local sans retenter HF, jusqu'a la fin du processus Python.
4. Au demarrage d'un nouveau processus, le flag est reset : on retente HF
   (utile apres un debut de mois qui recharge le quota).

**Fichier** : `src/greentech/ai/services/llm_local.py`

**Modele** : `Qwen/Qwen2.5-7B-Instruct` (identique au modele HF, pour une
continuite qualitative totale entre les deux backends).

**Device auto-detecte** :

1. `cuda` (RX 7900 XTX via ROCm 7.2) si `torch.cuda.is_available()`.
2. `torch_directml.device()` si le package DirectML est installe (PC portable).
3. `cpu` en dernier ressort.

**Chargement** : paresseux et proteges par verrou thread-safe. Le modele
(~14 Go FP16) n'est charge qu'au premier appel local reellement necessaire.
Un singleton garantit qu'un seul chargement est effectue pour tout le processus.

**Interface** : la methode `chat_completion(messages, max_tokens, temperature)`
renvoie un objet dont la forme (`response.choices[0].message.content`) est
strictement compatible avec l'API `AsyncInferenceClient`, ce qui permet de
remplacer le backend sans modifier les appelants.

---

### 11.5 Application aux resumes (summarizer.py)

Le meme dispatcher est utilise par `src/greentech/ai/services/summarizer.py`
pour les deux resumes :

- `summarize_text(text)` : resume general en francais.
- `summarize_green_aspects(text)` : resume centre sur les aspects ecologiques.

Les appels `chat_completion` passent tous par le dispatcher, donc beneficient
automatiquement du fallback local en cas de quota HF epuise. Les resumes en
mode batch (`summarize_green_it_articles`) ne sont generes que pour les
articles confirmes Green IT (`est_green_it=true`) afin de limiter le volume
de requetes.

---

### 11.6 Tracabilite en base

La colonne `articles.modele_classification` enregistre l'etage qui a rendu
la decision finale :

| Valeur | Signification |
|--------|---------------|
| `NULL` | Article jamais classifie |
| `keyword_filter` | Decision du pre-filtre (NON_GREEN direct) OU candidat en attente (si `est_green_it IS NULL`) |
| `keyword_filter+qwen_llm_judge` | Decision du LLM judge (etage 2) |
| `meta-llama/Llama-3.2-3B+LoRA` | Decision du modele de production en inference temps reel via `/analyze` |

Le champ `articles.score_confiance` est renseigne par l'etage 2 et le
modele de production (valeur nulle pour les NON_GREEN directs).

---

### 11.7 Commandes d'execution

```bash
# Etage 1 : pre-filtre mots-cles (binaire CANDIDATE/NON_GREEN)
uv run python scripts/auto_annotate_dataset.py

# Etage 2 : LLM judge sur les candidats (bascule locale si quota HF epuise)
uv run python scripts/classify_candidates.py

# Resumes Green IT confirmes (general + ecologique, meme fallback)
uv run python -m greentech.ai.services.summarizer

# Export golden_dataset.csv depuis l'etat final de la DB
uv run python scripts/export_golden_dataset.py

# Enchaine tout via le pipeline de re-entrainement
uv run python scripts/retrain_pipeline.py annotate classify summarize export-golden
```

---

### 11.8 Variables d'environnement

| Variable | Description |
|----------|-------------|
| `HUGGINGFACE_TOKEN` | Token HF (scope `read`). Sert a l'API Serverless et au telechargement du modele local en fallback. |
| `HUGGINGFACE_MODEL_CLASSIFIER_LLM` | Modele LLM judge (defaut : `Qwen/Qwen2.5-7B-Instruct`). |
| `HUGGINGFACE_MODEL_LOCAL_FALLBACK` | Modele local utilise en fallback (defaut : `Qwen/Qwen2.5-7B-Instruct`, meme que HF pour continuite). |

---

### 11.9 Choix techniques et justifications

| Choix | Justification |
|-------|---------------|
| **Pipeline en deux etages** (mots-cles + LLM) | Le pre-filtre elimine ~85% du corpus sans cout LLM. Le LLM judge ne traite que les ~15% ambigus, ce qui divise par ~7 la consommation du quota HF pour une qualite similaire. |
| **Pre-filtre volontairement permissif** (`CANDIDATE_MIN_GREEN_SCORE=0.1`) | Maximise le recall a l'etage 1 : tout signal Green meme faible declenche une verification LLM. Les faux positifs sont corriges en etage 2, mais les vrais Green IT ne peuvent pas etre rates. |
| **Fallback Qwen local (meme modele que HF)** | Continuite qualitative totale entre les deux backends : un article donne obtient le meme type de verdict qu'il soit traite par HF ou localement. Pas de divergence de comportement. |
| **Detection 402 et bascule automatique** | Evite toute interruption de service lors du franchissement du quota mensuel HF. Le traitement redemarre automatiquement sur GPU local sans intervention manuelle. |
| **Singleton `LocalQwenClient` + chargement paresseux** | Le modele (~14 Go FP16) n'est charge qu'au premier appel local. Les processus qui ne depassent jamais le quota HF ne paient donc jamais le cout de chargement. |
| **Parser JSON en trois passes** | Les LLM produisent parfois du JSON avec des backslashes non standards. Le parser tolerant recupere la majorite des reponses "presque valides" au lieu d'echouer. |
