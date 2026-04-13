# Sources de Donnees - Projet GreenTech Intelligence


Ce document recense les trois sources de donnees externes selectionnees pour alimenter
l'architecture Data Lakehouse du projet. Elles ont ete choisies pour leur pertinence
thematique ("Green IT", "Sustainable AI") et pour valider les contraintes techniques
du diplome (API, Scraping dynamique, Big Data).

---

## 1. Source API : Flux d'Actualites (Veille Chaude)

**Nom** : NewsData.io

**Type** : API REST (JSON)

**URL d'acces** : <https://newsdata.io/>

**Documentation technique** : <https://newsdata.io/documentation>

**Endpoint principal** :

```
GET https://newsdata.io/api/1/latest?apikey=YOUR_API_KEY&q=green+IT&category=technology&language=en
```

### Pourquoi cette source ?

- **Quota genereux** : 200 credits par jour dans le plan gratuit (contre 100 habituellement),
  offrant une marge de securite pour le developpement.
- **Filtrage semantique** : Permet de combiner des mots-cles complexes
  (`"Sustainable AI" OR "Green IT"`) avec un filtre de categorie `technology`.
- **Fraicheur** : Delai de mise a jour raisonnable (15 min a 12h) pour de la veille.

### Donnees recuperees (pour le MCD)

| Champ API     | Description                  |
|---------------|------------------------------|
| `title`       | Titre de l'article           |
| `link`        | URL de l'article             |
| `description` | Resume court                 |
| `content`     | Contenu complet              |
| `pubDate`     | Date de publication          |
| `source_id`   | Identifiant de la source     |

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
