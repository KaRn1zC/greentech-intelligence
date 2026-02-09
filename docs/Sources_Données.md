📂 Sources de Données - Projet GreenTech Intelligence
Ce document recense les trois sources de données officielles sélectionnées pour alimenter l'architecture Data Lakehouse du projet. Elles ont été choisies pour leur pertinence thématique ("Green IT", "Sustainable AI") et pour valider les contraintes techniques du diplôme (API, Scraping dynamique, Big Data).

1. Source API : Flux d'Actualités (Veille Chaude)
Nom : NewsData.io

Type : API REST (JSON)

🔗 URL d'accès : https://newsdata.io/

Documentation technique : Documentation Officielle

Pourquoi cette source ?


Quota généreux : Offre 200 crédits par jour dans le plan gratuit (contre 100 habituellement), offrant une marge de sécurité pour le développement.


Filtrage sémantique : Permet de combiner des mots-clés complexes ("Sustainable AI" OR "Green IT") avec un filtre de catégorie technology.
+1


Fraîcheur : Délai de mise à jour raisonnable (15 min à 12h) pour de la veille.

Exemple de Données à récupérer (pour le MCD) :

title, link, description, content, pubDate, source_id.

2. Source Scraping : Blog Tech (Market Intelligence)
Nom : TechCrunch - Section Climate

Type : Web Scraping (Site Dynamique / SPA)


🔗 URL Cible : https://techcrunch.com/category/climate/ 

Contrainte Technique Validée : Utilisation obligatoire de Playwright.

Pourquoi cette source ?


Architecture Dynamique : Le site utilise un chargement en "Infinite Scroll" (défilement infini).

Justification Playwright : Un scraper simple ne verrait que les premiers articles. Playwright est nécessaire pour simuler le scroll utilisateur et attendre le chargement réseau (hydratation React).
+1


Pertinence : Couvre les levées de fonds et l'innovation matérielle (Hardware) durable.

Exemple de Données à récupérer (pour le MCD) :

Titre Article, URL, Date Publication, Contenu HTML, Auteur.

3. Source Big Data : Archives Scientifiques (Recherche de Fond)
Nom : arXiv Metadata Dataset (Kaggle)

Type : Dataset Statique (JSON volumineux)


🔗 URL de Téléchargement : Kaggle - arXiv Dataset (ou miroir direct : Lien Dataset )

Contrainte Technique Validée : Utilisation obligatoire d'Apache Spark et MinIO.

Pourquoi cette source ?


Volume "Big Data" : Le fichier pèse environ 3.6 Go (format JSON) et contient plus de 1.7 million d'articles.
+1

Justification Spark : Le volume est trop important pour être traité en mémoire simple. Il nécessite un traitement distribué pour filtrer les catégories IA (cs.AI) et analyser les abstracts.
+1


Richesse Textuelle : Contient des résumés (abstracts) techniques idéaux pour l'analyse NLP.

Exemple de Données à récupérer (pour le MCD) :

id, title, abstract, categories, update_date.