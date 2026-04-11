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

## 2. Source Scraping : Blog Tech (Market Intelligence)

**Nom** : TechCrunch - Section Climate

**Type** : Web Scraping (Site Dynamique / SPA)

**URL cible** : <https://techcrunch.com/category/climate/>

**Contrainte technique validee** : Utilisation obligatoire de Playwright.

### Pourquoi cette source ?

- **Architecture dynamique** : Le site utilise un chargement en "Infinite Scroll"
  (defilement infini).
- **Justification Playwright** : Un scraper simple ne verrait que les premiers articles.
  Playwright est necessaire pour simuler le scroll utilisateur et attendre le chargement
  reseau (hydratation React).
- **Pertinence** : Couvre les levees de fonds et l'innovation materielle (Hardware) durable.

### Donnees recuperees (pour le MCD)

| Champ extrait      | Description                  |
|--------------------|------------------------------|
| Titre article      | Titre de l'article           |
| URL                | Lien vers l'article complet  |
| Date de publication| Date de parution             |
| Contenu HTML       | Corps de l'article           |
| Auteur             | Nom de l'auteur              |

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
