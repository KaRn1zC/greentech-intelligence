# Sources de Donnees - Projet GreenTech Intelligence

Ce document recense les sources de donnees externes selectionnees pour alimenter
l'architecture Data Lakehouse du projet. Elles ont ete choisies pour leur pertinence
thematique ("Green IT", "Sustainable AI") et pour valider les contraintes techniques
du diplome (API REST/JSON, Scraping hybride, Big Data).

Le critere **E1 / C1** du referentiel impose **trois types de sources differents**,
respectivement couverts par : une API REST/JSON (The Guardian + Dev.to en
complement), un scraping hybride (TechCrunch) et un dataset volumineux traite
avec Spark (arXiv Kaggle).

> **Historique** : la premiere version du projet utilisait `NewsData.io` comme
> source API. Le free tier tronquant systematiquement le contenu au placeholder
> "ONLY AVAILABLE IN PAID PLANS" (1316 articles pourris sur le dataset, soit
> 21% inexploitables), cette source a ete desactivee le 16 avril 2026 et
> remplacee par **The Guardian Open Platform**. Le collecteur
> `api_collector.py` reste disponible pour un usage futur si l'abonnement est
> paye, mais n'est plus alimente par defaut.

---

## 1. Source API REST/JSON principale : The Guardian (Veille Journalistique)

**Nom** : The Guardian Open Platform

**Type** : API REST (JSON)

**URL d'acces** : <https://open-platform.theguardian.com/>

**Documentation technique** : <https://open-platform.theguardian.com/documentation/>

**Endpoint principal** :

```
GET https://content.guardianapis.com/search?
    api-key=YOUR_KEY
    &q=green+IT
    &show-fields=bodyText,headline,trailText,byline,publication,lastModified,lang
    &page-size=50
    &order-by=newest
```

### Pourquoi cette source ?

- **Contenu integral garanti** : avec `show-fields=bodyText`, l'API renvoie
  l'integralite du corps de l'article (plusieurs milliers de caracteres en
  moyenne), pas un teaser ni un resume tronque. C'est precisement le probleme
  qui disqualifiait NewsData et la plupart des APIs news "gratuites".
- **Quota genereux** : 5000 requetes/jour, 12 requetes/seconde en tier
  **Developer** gratuit. A 50 articles par requete, cela represente jusqu'a
  250 000 articles/jour theoriques, largement au-dela de nos besoins.
- **Couverture Green IT native** : sections `environment`, `technology`,
  `sustainable-business` et `green-living` fournissent des articles
  directement pertinents pour le domaine, avec des redacteurs specialises.
- **Qualite journalistique** : articles longs (1500 a 5000+ caracteres),
  bien structures, avec un vocabulaire technique soutenu qui enrichit
  le dataset d'entrainement.
- **Stabilite** : API en production depuis 2009, maintenue par The Guardian
  News & Media. Aucun risque d'arret ou de bascule en tier payant a court
  terme.

### Conditions d'usage

- **Usage non-commercial uniquement** (tier Developer). Pour un usage
  commercial, passer au tier Commercial (payant).
- **Images et videos non inclus** dans le tier Developer (texte seulement),
  ce qui correspond exactement a nos besoins.
- **Suppression automatique de la cle** si aucune requete pendant 90 jours
  (penser a renouveler si mise en pause prolongee).

### Donnees recuperees (pour le MCD)

| Champ API               | Description                                  |
|-------------------------|----------------------------------------------|
| `webTitle`              | Titre de l'article                           |
| `webUrl`                | URL de l'article                             |
| `fields.bodyText`       | **Contenu integral en texte brut**           |
| `fields.trailText`      | Resume court (description affichee)          |
| `fields.byline`         | Auteur(s)                                    |
| `fields.lang`           | Langue (en, fr, ...)                         |
| `webPublicationDate`    | Date de publication (ISO 8601)               |
| `sectionName`           | Section editoriale (Environment, Tech, ...)  |
| `id`                    | Identifiant Guardian unique                  |

---

## 1bis. Source API REST/JSON complementaire : Dev.to / Forem

**Nom** : Dev.to (Forem API)

**Type** : API REST (JSON), aucune cle requise

**URL d'acces** : <https://developers.forem.com/api>

**Endpoints utilises** :

```
# 1. Lister les articles par tag (renvoie uniquement description + metadonnees)
GET https://dev.to/api/articles?tag=greenit&per_page=30&page=1

# 2. Recuperer le corps integral (body_markdown) - 1 appel par article
GET https://dev.to/api/articles/{id}
```

### Pourquoi cette source complementaire ?

- **Diversification du registre** : Guardian apporte une prose journalistique ;
  Dev.to apporte un registre technique ecrit par des developpeurs (retours
  d'experience, tutoriels, benchmarks) avec un vocabulaire plus pratique
  (outils, frameworks, code). Le classifieur gagne en robustesse avec cette
  double exposition.
- **Tags Green IT natifs** : `greenit`, `sustainability`, `climatechange`,
  `webperf`, `cleanenergy`, `environment`, `sustainabletech`, `greensoftware`.
- **Aucune friction** : pas de cle API a gerer, pas de quota dur documente,
  simple a integrer dans la CI/CD.
- **Contenu integral** : `body_markdown` est complet (typiquement 1500-10000
  caracteres par article) apres le second appel.

### Particularite d'implementation

Pattern **2 appels par article** obligatoire :
1. `GET /articles?tag=X` ne renvoie qu'une `description` tronquee (piege
   courant avec Dev.to).
2. `GET /articles/{id}` renvoie le `body_markdown` integral.

Le collecteur `devto_collector.py` orchestre ce pattern et nettoie le markdown
(retrait des blocs de code, images, simplification des liens) avant stockage
pour produire un texte compatible avec les etapes de classification et de
resume.

### Donnees recuperees (pour le MCD)

| Champ API         | Description                                    |
|-------------------|------------------------------------------------|
| `title`           | Titre de l'article                             |
| `url`             | URL dev.to/user/article-slug                   |
| `body_markdown`   | **Contenu integral en markdown, puis nettoye** |
| `description`     | Resume court                                   |
| `user.name`       | Auteur                                         |
| `published_at`    | Date de publication ISO 8601                   |
| `tag_list`        | Tags (greenit, sustainability, ...)            |
| `id`              | Identifiant Dev.to unique                      |

---

## 2. Source Scraping hybride : Blog Tech (Market Intelligence)

**Nom** : TechCrunch - Section Climate

**Type** : Scraping hybride en deux etapes
1. **Decouverte d'URLs** via le flux RSS officiel (listing stable)
2. **Scraping HTML** des articles individuels avec Scrapy + Playwright

**URLs cibles** :
- Flux RSS : <https://techcrunch.com/category/climate/feed/>
- Pages d'articles : `https://techcrunch.com/YYYY/MM/DD/<slug>/`

**Outils imposes (critere C1 du referentiel)** :
- `scrapy` : framework de crawl/scraping (HTTP, parsing, pipeline)
- `playwright` : rendu JS pour pages dynamiques
- `scrapy-playwright` : integration navigateur dans Scrapy
- `httpx` + `feedparser` : pour l'etape RSS initiale

### Pourquoi cette architecture hybride ?

**Robustesse** : le flux RSS sert de liste d'URLs stable, insensible aux refontes
de la page d'index. TechCrunch a par exemple supprime la balise `<article>` en
avril 2026 sur la page de listing, ce qui aurait casse un scraper purement HTML
sur l'index.

**Conformite au referentiel C1** : le scraping HTML reste le coeur de la collecte.
Chaque URL fournie par le RSS est telechargee avec Scrapy + Playwright, puis
parsee via des selecteurs CSS sur le DOM rendu. Cela coche explicitement les
criteres de certification :
- "telechargement de l'HTML d'une ou plusieurs pages web visees par une action de scraping"
- "filtrage/parsing des donnees utiles dans les resultats obtenus depuis l'HTML
  collecte d'un site web (scraping)"

**Qualite des donnees** : le RSS ne fournit qu'un resume partiel (quelques
paragraphes). Le scraping HTML recupere l'integralite du contenu de l'article,
ce qui nourrit mieux le modele de classification Green IT et le summarizer Qwen.

### Selecteurs CSS utilises (pages d'articles TechCrunch)

| Champ              | Selecteur CSS                                         |
|--------------------|-------------------------------------------------------|
| Titre              | `h1.article-hero__title::text`                        |
| Titre (fallback)   | `meta[property="og:title"]::attr(content)`            |
| Date publication   | `time::attr(datetime)` (ISO 8601)                     |
| Auteurs            | `a[href*="/author/"]::text`                           |
| Contenu texte      | `div.entry-content p::text` (concatene)               |
| Contenu HTML brut  | `div.entry-content` (HTML complet)                    |
| Resume court       | `meta[property="og:description"]::attr(content)`      |

### Donnees recuperees (pour le MCD)

| Champ extrait      | Description                                  |
|--------------------|----------------------------------------------|
| Titre article      | Titre principal extrait du DOM HTML          |
| URL                | URL finale de l'article (apres redirections) |
| Date de publication| Date ISO 8601 de parution                    |
| Auteur             | Noms des auteurs concatenes                  |
| Contenu texte      | Article complet (paragraphes fusionnes)      |
| Contenu HTML       | Bloc `entry-content` brut                    |
| Resume court       | Description OpenGraph                        |

### Contraintes ethiques respectees

- `ROBOTSTXT_OBEY = True` : respect du fichier robots.txt
- `DOWNLOAD_DELAY = 2.0` : delai de 2 secondes entre chaque page
- `CONCURRENT_REQUESTS = 1` : une seule requete simultanee
- User-Agent identifie : `GreenTech-Bot/1.0`
- Limite a 20 articles par session (`MAX_ARTICLES`)

---

## 3. Source Big Data : Archives Scientifiques (Recherche de Fond)

**Nom** : arXiv Metadata Dataset (Kaggle / Cornell University)

**Type** : Dataset statique (JSON volumineux)

**URL de telechargement** : <https://www.kaggle.com/datasets/Cornell-University/arxiv>

**Licence** : CC0 (Domaine Public)

**Contrainte technique validee** : Utilisation obligatoire d'Apache Spark et MinIO.

### Pourquoi cette source ?

- **Volume "Big Data"** : Le fichier pese environ 3.6 Go (format JSON) et contient
  plus de 1.7 million d'articles scientifiques.
- **Justification Spark** : Le volume est trop important pour etre traite en memoire
  simple. Il necessite un traitement distribue pour filtrer les categories IA (`cs.AI`)
  et analyser les abstracts.
- **Richesse textuelle** : Contient des resumes (abstracts) techniques ideaux
  pour l'analyse NLP et l'entrainement du modele de classification Green IT.

### Donnees recuperees (pour le MCD)

| Champ JSON     | Description                          |
|----------------|--------------------------------------|
| `id`           | Identifiant unique arXiv             |
| `title`        | Titre de la publication              |
| `abstract`     | Resume scientifique                  |
| `categories`   | Categories arXiv (ex: `cs.AI`)       |
| `update_date`  | Date de derniere mise a jour         |
