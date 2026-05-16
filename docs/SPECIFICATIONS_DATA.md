# Spécifications des Sources de Données - GreenTech Intelligence

> **Auteur** : KaRn1zC
> **Date de création** : 2026-05-16
> **Version** : 1.0
> **Document complémentaire** : `docs/SPECIFICATIONS_TECHNIQUES.md` (architecture technique du pipeline)
> **Référence RGPD** : `docs/REGISTRE_RGPD.md` (traitement des données personnelles par source)

---

## 1. Vue d'ensemble

Ce document décrit l'inventaire exhaustif des **10 sources de données actives** alimentant le pipeline de collecte de GreenTech Intelligence au 2026-05-16. Il valide la couverture des **5 catégories techniques C1** exigées par le référentiel diplômant (REST/JSON, scraping, fichier, BDD, Big Data) et documente les contraintes opérationnelles de chaque collecteur.

### 1.1 Synthèse

| Indicateur | Valeur |
|------------|--------|
| Sources actives | 10 |
| Sources désactivées (historique) | 1 (NewsData.io, retirée le 2026-04-19) |
| Articles collectés (BDD `articles`) | 11 664 |
| Articles Green IT confirmés | 1 018 (8.73 %) |
| Répartition linguistique | EN 74.75 % (8 719) / FR 25.25 % (2 945) |
| Catégories techniques C1 couvertes | 5 / 5 |

### 1.2 Couverture des catégories C1

| Catégorie C1 | Sources concernées | Validation |
|--------------|--------------------|------------|
| **API REST/JSON** | The Guardian, Dev.to, arXiv API, Crossref | ✅ 4 sources |
| **Scraping HTML** (statique + dynamique) | TechCrunch Climate (Playwright), GreenIT.fr, GSF, SWD, CAT (Scrapy) | ✅ 5 sources |
| **Fichier (dataset)** | arXiv Dataset Kaggle (JSONL) | ✅ 1 source |
| **Base de données relationnelle** | PostgreSQL `search_config` (mots-clés et URLs dynamiques) | ✅ Module 0 |
| **Big Data** | Apache Spark + MinIO (lecture/écriture S3, traitement DataFrame) | ✅ Étape 2.4 |

---

## 2. Inventaire détaillé des sources

### 2.1 Source 1 : The Guardian (Open Platform)

| Critère | Valeur |
|---------|--------|
| Type technique | API REST/JSON |
| URL de base | `https://content.guardianapis.com/search` |
| Authentification | Clé API obligatoire (`GUARDIAN_API_KEY`) |
| Quota | 5 000 requêtes/jour (tier Developer, non-commercial) |
| Localisation fournisseur | Royaume-Uni (Guardian News & Media) |
| Sections collectées | `environment` (5 932 articles 2024+), `technology` (2 674 articles 2024+) |
| Format de retour | JSON enrichi (`fields=bodyText,trailText,byline,...`), 8 000+ caractères/article |
| Pagination | `page` + `pageSize` (max 200/page) |
| Délai inter-requête | 1 s (recommandation Guardian) |
| Délai entre runs | Idempotent via pre-check URL (skip articles déjà en BDD) |
| Collecteur Python | `src/greentech/data/collectors/guardian_collector.py` |
| Commande CLI | `uv run python -m greentech.data.collectors.guardian_collector` |
| Volume en BDD | 1 252 articles |

**Contraintes techniques** :
- Sub-sections (ex: `technology/green-computing`) **inexistantes côté API** (vérifié via endpoint `/sections`). Utilisation du filtre `section=` direct.
- Dédoublonnage URL au niveau collecteur (mode 2 passes : plein-texte + N sections).

### 2.2 Source 2 : Dev.to (API Forem)

| Critère | Valeur |
|---------|--------|
| Type technique | API REST/JSON |
| URL de base | `https://dev.to/api/articles` |
| Authentification | **Aucune clé requise** en lecture publique (variable `DEVTO_API_KEY` laissée vide) |
| Quota | Non documenté, prévoir délai 1 s entre requêtes |
| Localisation fournisseur | États-Unis (Forem Inc.) |
| Tags collectés | `greenit`, `sustainability`, `greentech`, `sustainable`, `climate-tech`, `carbon`, etc. (8 tags) |
| Format de retour | JSON (`title`, `body_markdown`, `user.name`, `tags`) |
| Pagination | `page` + `per_page` (max 1 000/page) |
| Collecteur Python | `src/greentech/data/collectors/devto_collector.py` |
| Commande CLI | `uv run python -m greentech.data.collectors.devto_collector` |
| Volume en BDD | 135 articles |

### 2.3 Source 3 : arXiv API

| Critère | Valeur |
|---------|--------|
| Type technique | API REST + Atom XML |
| URL de base | `http://export.arxiv.org/api/query` |
| Authentification | Aucune (API publique académique) |
| Quota | Non documenté, **délai obligatoire 3 s** entre requêtes (bonnes pratiques arXiv) |
| Localisation fournisseur | États-Unis (Cornell University) |
| Catégories filtrées (post-fetch) | `cs.*`, `eess.*`, `stat.ML` (filtrage souple) |
| Queries Green IT (9) | `green computing`, `sustainable AI`, `green AI`, `carbon-aware computing`, `energy-efficient ML`, `green software engineering`, `low-power neural network`, `data center sustainability`, `sustainable computing` |
| Format de retour | Atom XML (parsing via `feedparser`), `summary` = abstract 150-300 mots |
| Pagination | `start` + `max_results` (PAGE_SIZE=100, MAX_RESULTS_PER_KEYWORD=500) |
| Identifiant unique | `arxiv_id` sans version (`arxiv.org/abs/ID`) pour dédup naturelle |
| Collecteur Python | `src/greentech/data/collectors/arxiv_collector.py` |
| Commande CLI | `uv run python -m greentech.data.collectors.arxiv_collector` |
| Volume en BDD | 382 articles |

### 2.4 Source 4 : arXiv Dataset (Kaggle snapshot)

| Critère | Valeur |
|---------|--------|
| Type technique | **Fichier local** (JSONL) - catégorie C1 "fichier" |
| URL d'origine | <https://www.kaggle.com/datasets/Cornell-University/arxiv> |
| Fichier local | `data/arxiv-metadata-oai-snapshot.json` |
| Format | JSON Lines (1 article/ligne), ~2.5 millions d'entrées totales (filtrage post-fetch) |
| Localisation fournisseur | États-Unis (Cornell University) |
| Filtrage appliqué | Catégories `cs.*` + filtrage par mots-clés Green IT |
| Collecteur Python | `src/greentech/data/collectors/file_ingester.py` |
| Commande CLI | Intégrée au pipeline `retrain_pipeline.py collect` |
| Volume en BDD | 4 957 articles |
| Note historique | Source d'entraînement initiale, conservée pour rétro-compatibilité et démonstration de la catégorie C1 "fichier" |

### 2.5 Source 5 : Crossref (Polite Pool)

| Critère | Valeur |
|---------|--------|
| Type technique | API REST/JSON |
| URL de base | `https://api.crossref.org/works` |
| Authentification | Optionnelle (Polite Pool via `mailto:` dans User-Agent) |
| Quota | Tier prioritaire si Polite Pool, non documenté précisément |
| Localisation fournisseur | Royaume-Uni (Crossref / PILA) |
| Filtres serveur | `has-abstract:true,from-pub-date:2020,type:journal-article` + tri `relevance:desc` |
| Queries (8) | `green computing`, `sustainable computing`, `green AI`, `green software`, `energy-efficient computing`, `carbon-aware`, `sustainable software engineering`, `green IT` |
| Format de retour | JSON avec abstract en JATS XML (strip via regex `<jats:p>`, `<jats:sec>`) |
| Types acceptés | `journal-article`, `proceedings-article` (livres et chapitres rejetés) |
| Identifiant unique | DOI (`https://doi.org/<DOI>`) |
| Variable d'environnement | `CROSSREF_MAILTO` (vide = pool public sans transmission e-mail) |
| Collecteur Python | `src/greentech/data/collectors/crossref_collector.py` |
| Commande CLI | `uv run python -m greentech.data.collectors.crossref_collector` |
| Volume en BDD | 1 499 articles |

**Précision RGPD** : la transmission de l'e-mail du responsable de traitement à Crossref est documentée dans `docs/REGISTRE_RGPD.md` section 6.1.

### 2.6 Source 6 : TechCrunch Climate (Scraping HTML dynamique)

| Critère | Valeur |
|---------|--------|
| Type technique | **Scraping HTML dynamique** (Scrapy + Playwright) |
| URL de base | `https://techcrunch.com/category/climate/feed/` (RSS) puis pages détail |
| Authentification | Aucune |
| Quota | Limité par `robots.txt` et politesse (DOWNLOAD_DELAY=2s) |
| Localisation fournisseur | États-Unis (Yahoo Inc. depuis 2022) |
| Format de retour | HTML rendu par Playwright (JavaScript exécuté) |
| Filtrage pré-fetch | Pre-check URL en BDD via `url_precheck.py` (évite fetch Playwright coûteux ~5-10s/page) |
| Délai inter-requête | 2 s minimum |
| Respect `robots.txt` | `ROBOTSTXT_OBEY = True` |
| Collecteur Python | `src/greentech/data/collectors/scraper.py` |
| Commande CLI | `uv run python -m greentech.data.collectors.scraper` |
| Volume en BDD | 105 articles |

### 2.7 Source 7 : GreenIT.fr (Scraping HTML statique)

| Critère | Valeur |
|---------|--------|
| Type technique | **Scraping HTML statique** (Scrapy) |
| URL de base | `https://www.greenit.fr/` |
| `robots.txt` | Aucun fichier (convention = allow all) |
| Découverte | 3 sitemaps Yoast (`post-sitemap.xml`, `post-sitemap2.xml`, `post-sitemap3.xml`), 1 001 posts WordPress |
| URL pattern | `/YYYY/MM/DD/slug/` |
| Langue | Français |
| Pertinence Green IT | **100 %** (site spécialisé écoconception numérique) |
| Format article | HTML statique, 5 000+ caractères/article |
| Délai inter-requête | 2 s |
| Spider Python | `src/greentech/data/collectors/spiders/greenit_fr_spider.py` |
| Orchestrateur | `src/greentech/data/collectors/static_scraping_collector.py` |
| Volume en BDD | 2 945 articles |

### 2.8 Source 8 : Green Software Foundation (Scraping HTML statique)

| Critère | Valeur |
|---------|--------|
| Type technique | **Scraping HTML statique** (Scrapy) |
| URL de base | `https://greensoftware.foundation/articles/` |
| `robots.txt` | Permissif (Cloudflare content-signal sans règle bloquante) |
| Découverte | Pagination HTML `/articles/N` (17 pages × 10 articles) |
| Langue | Anglais |
| Pertinence Green IT | **100 %** (foundation spécialisée Green Software) |
| Format article | HTML statique, 8 500 caractères/article |
| Délai inter-requête | 2 s |
| Spider Python | `src/greentech/data/collectors/spiders/greensoftware_spider.py` |
| Volume en BDD | 193 articles |

### 2.9 Source 9 : Sustainable Web Design (Scraping HTML statique)

| Critère | Valeur |
|---------|--------|
| Type technique | **Scraping HTML statique** (Scrapy) |
| URL de base | `https://sustainablewebdesign.org/` |
| `robots.txt` | `User-agent: * / Disallow:` (= all allowed) |
| Découverte | 2 sitemaps Yoast (`post-sitemap.xml` 50 posts + `guidelines-sitemap.xml` 81 guidelines) |
| Langue | Anglais |
| Pertinence Green IT | **100 %** (organisation Mightybytes, projet Sustainable Web Manifesto) |
| Format article | HTML statique, 8 500+ caractères |
| Délai inter-requête | 2 s |
| Spider Python | `src/greentech/data/collectors/spiders/sustainable_web_spider.py` |
| Volume en BDD | 130 articles |

### 2.10 Source 10 : Climate Action Tech (Scraping HTML statique)

| Critère | Valeur |
|---------|--------|
| Type technique | **Scraping HTML statique** (Scrapy) |
| URL de base | `https://climateaction.tech/blog/` |
| `robots.txt` | `Disallow: /wp-admin/` (le reste autorisé) |
| Découverte | `sitemap_index.xml` → `post-sitemap.xml` (71 posts WordPress) |
| URL pattern | `/blog/slug/` |
| Langue | Anglais |
| Pertinence Green IT | **100 %** (communauté tech+climat) |
| Format article | HTML statique, 4 500 caractères/article |
| Délai inter-requête | 2 s |
| Spider Python | `src/greentech/data/collectors/spiders/climate_action_tech_spider.py` |
| Volume en BDD | 66 articles |

---

## 3. Sources désactivées (historique)

### 3.1 NewsData.io (désactivée 2026-04-19)

| Critère | Valeur |
|---------|--------|
| Motif de désactivation | Contenu tronqué en free tier (extraits seulement, ~200 caractères/article), exploitabilité insuffisante pour le LLM judge et l'entraînement du classifieur |
| Volume historique | 1 316 articles (supprimés de la BDD et de MinIO le 2026-04-19) |
| Statut RGPD | Toutes les données personnelles issues de NewsData.io ont été purgées (voir `docs/REGISTRE_RGPD.md` section 2.1.1) |
| Collecteur Python conservé | `src/greentech/data/collectors/api_collector.py` (désactivé via flag, code conservé pour documentation historique) |

### 3.2 Hugging Face dataset `climatebert/climate_detection` (écartée 2026-04-19)

Évaluée puis **rejetée** avant intégration :
- Format incompatible (ni titre, ni URL, paragraphes corporate hors scope tech)
- Labels "climat général" qui auraient corrompu le ground truth Green IT
- Décision documentée dans la phase B2.1 du plan (`docs/PLAN_ETAPES.md` section 7.2)

Pourrait éventuellement servir comme corpus d'évaluation de robustesse dans une phase ultérieure.

---

## 4. Stratégie de déduplication

### 4.1 Module commun `url_precheck.py`

Toutes les sources REST/JSON et tous les spiders Scrapy utilisent le module partagé `src/greentech/data/collectors/url_precheck.py` :

```python
load_known_urls(source_name) -> set[str]  # asyncpg direct, < 500 ms pour 50k URLs
url_is_known(url, known_set) -> bool      # comparaison normalisée (O(1))
normalize_url(url) -> str                  # http→https, host lowercase, trailing slash retiré
```

**Garanties** :
- **Aucun faux positif** : seules les URLs effectivement présentes en BDD déclenchent un skip
- **Faux négatifs éliminés** sur les cas réels (arXiv Kaggle https vs API http, casse du host, trailing slash)
- **Couverture** : 42 tests unitaires dans `tests/unit/data/collectors/test_url_precheck.py`

### 4.2 Flag `skip_existing`

Tous les collecteurs exposent `skip_existing: bool = True` par défaut. Override possible :

```bash
uv run python -m greentech.data.collectors.guardian_collector --skip-existing false
# (pour les spiders)
scrapy crawl greenit_fr -a skip_existing=false
```

### 4.3 Gain de performance mesuré

| Source | Gain (re-run) | Mécanisme |
|--------|---------------|-----------|
| TechCrunch | **Massif** | Évite fetch Playwright 5-10s/page |
| Dev.to | **Massif** | Évite fetch détail par article |
| arXiv API / Crossref / Guardian | Modeste | Évite pollution MinIO `raw-data` |
| File ingester | Faible | Évite push doublons vers MinIO |

---

## 5. Pipeline de traitement après collecte

Une fois les fichiers bruts déposés dans `s3://raw-data/<source>/<date>/`, le pipeline poursuit :

1. **Spark cleaning** (`src/greentech/data/processors/spark_cleaner.py`) :
   - Lecture du préfixe MinIO `scraping/` global (TechCrunch + 4 spiders B2.3)
   - Suppression des balises HTML résiduelles via trafilatura
   - Filtrage des articles < 50 caractères (`step_clean`)
   - Homogénéisation des dates en ISO 8601
   - Anonymisation des noms d'auteurs (initiales)
   - Détection de la langue (langdetect)
   - Écriture en Parquet vers `s3://clean-data/`

2. **Ingestion SQL** (`src/greentech/data/storage/sql_ingester.py`) :
   - Lecture des Parquet depuis MinIO
   - Mapping vers le modèle SQLAlchemy `Article`
   - Upsert sur URL normalisée (anti-doublons)

3. **Génération résumé classification** (`scripts/generate_classification_summaries.py`) :
   - Pour chaque article sans abstract natif, génération via `classification_summarizer.py` (max 450 tokens)
   - Persistance dans `articles.resume`

4. **Annotation Green IT en 2 étages** :
   - Étage 1 : `scripts/auto_annotate_dataset.py` (pré-filtre mots-clés)
   - Étage 2 : `scripts/classify_candidates.py` (LLM judge Qwen)

5. **Export du dataset d'entraînement** (`scripts/export_golden_dataset.py`) :
   - Régénère `data/golden_dataset.csv` depuis la BDD
   - 11 664 articles annotés, 1 018 Green IT (8.73 %)

---

## 6. Métriques de collecte (cibles et réalisé)

| Indicateur | Cible initiale (B2) | Réalisé (2026-04-21) |
|------------|---------------------|----------------------|
| Volume total | 5 000-8 000 nouveaux articles | 11 664 articles totaux |
| Articles Green IT confirmés | 200-500 | **1 018 (8.73 %)** |
| Sources actives | 8 nouvelles + 2 historiques | 10 actives (objectif dépassé : 8 nouvelles sur 9 prévues, dataset HF écarté) |
| Couverture C1 | 5 catégories | 5 / 5 ✅ |

---

## 7. Variables d'environnement requises

```bash
# Sources REST/JSON
GUARDIAN_API_KEY=<clé Open Platform Developer tier>
DEVTO_API_KEY=                    # vide = lecture publique
CROSSREF_MAILTO=                  # vide = pool public sans transmission e-mail

# Stockage et BDD
DATABASE_URL=postgresql+asyncpg://greentech:password@localhost:5432/greentech_db
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123
```

Toutes les variables sont également documentées dans `.env.example` à la racine du projet.

---

## 8. Documents connexes

- `docs/SPECIFICATIONS_TECHNIQUES.md` : architecture technique détaillée du pipeline (flux Spark, schéma BDD, contraintes d'extraction historiques)
- `docs/REGISTRE_RGPD.md` : traitement des données personnelles par source et règles d'anonymisation
- `docs/PLAN_ETAPES.md` section 2 et section 7.2 : feuille de route de la collecte et de l'enrichissement
- `documentation interne` section "Data (E1)" et "Commandes" : référence opérationnelle pour les commandes CLI

---

**Date de dernière mise à jour** : 2026-05-16
**Version** : 1.0
