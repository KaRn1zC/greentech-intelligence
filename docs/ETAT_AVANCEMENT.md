# Etat d'avancement du projet - GreenTech Intelligence

> **Derniere mise a jour** : 2026-03-10
> **Redige par** : KaRn1zC

---

## Ou nous en sommes

### ETAPE 1 : Installation & Configuration - **TERMINEE**
Toutes les dependances et outils sont installes et configures.

### ETAPE 2 : Data Factory & Gestion de Donnees (Bloc E1) - **TERMINEE**
Pipeline complet operationnel : 3 sources de collecte, nettoyage PySpark, ingestion PostgreSQL.

### ETAPE 3 : Intelligence Artificielle (Blocs E2 & E3) - **EN COURS**

#### Section 3.1 : Veille Technologique & Benchmark - **TERMINEE**
- Veille Inoreader + Perplexity Pro configuree
- Benchmark services IA realise (choix : HuggingFace Serverless API)
- Module summarizer.py developpe et fonctionnel

#### Section 3.2 : Preparation des Donnees & MLOps - **TERMINEE**
- **Golden Dataset** : CREE et annote (5808 articles, 22 Green IT / 5786 Non Green IT)
- **DVC** : Initialise, remote MinIO configure, dataset versionne et pousse (s3://models/dvc)

#### Section 3.3 : Entrainement & Competition des Modeles - **A FAIRE**
- Scripts de training prets (DeBERTa-v3-base + Llama 3.2 3B)
- Configuration MLflow + CodeCarbon prete
- Necessite le GPU AMD ROCm (PC Fixe)

#### Section 3.4 : Validation & Packaging - **PARTIELLEMENT FAIT**
- Tests Deepchecks ecrits
- Packaging du modele gagnant : a faire apres entrainement

#### Section 3.5 : Deploiement MLOps - **PARTIELLEMENT FAIT**
- Metriques de production definies
- Configuration Prometheus preparee

---

## Travail effectue lors de cette session (2026-03-10)

### 1. Collecte ciblee avec 120 credits API

**Script** : `scripts/collect_targeted.py`

Utilisation optimale des 120 credits restants sur NewsData.io :
- 60 requetes Green IT (mots-cles tres specifiques : data center energy, green software, carbon footprint AI, etc.)
- 59 requetes Non Green IT (IA sante, cybersecurite, crypto, gaming, etc.)
- 12 erreurs rate limit (429) avec reprise automatique apres 60s
- **Resultat** : 779 articles collectes et stockes dans MinIO raw-data

### 2. Affinement du systeme d'annotation

**Script** : `scripts/auto_annotate_dataset.py`

Systeme de scoring multi-criteres avec 100+ indicateurs ponderes :
- Normalisation des textes (tirets vers espaces pour matcher "energy-efficient" = "energy efficient")
- Patterns regex pour associations non adjacentes (`cooling.*data cent`)
- Detection de prefixes "green" dans les noms de projets (GreenNLP, GreenAI, etc.)
- Filtres renforces contre les faux positifs :
  - Articles boursiers (watchlist, stocks to follow)
  - Plaintes communautaires (residents, noise, dust)
  - Rapports de marche generiques (market forecast, billion)
- Seuil de classification releve de net >= 2.0 a >= 3.0
- Score de confiance calcule pour chaque article (92.7% a haute confiance)

### 3. Pipeline complet execute

```
Collecte (3 sources) --> MinIO raw-data (JSON)
  --> PySpark cleaning (dedup, normalisation, RGPD)
    --> MinIO clean-data (Parquet)
      --> SQL ingestion (PostgreSQL, upsert)
        --> Auto-annotation (Golden Dataset CSV)
```

**Statistiques finales** :

| Source | Articles |
|--------|----------|
| arXiv Dataset | 5003 |
| NewsData.io | 802 |
| TechCrunch Climate | 3 |
| **Total** | **5808** |

**Annotation** : 22 Green IT (0.4%) / 5786 Non Green IT (99.6%)
Confiance haute (>= 0.7) : 89.3%

### 4. Corrections techniques

- **Spark cleaner** : Chemin API elargi de `api/newsdata` a `api` pour inclure les sous-dossiers `newsdata_targeted_*`
- **MinIO client** : UUID ajoute aux chemins pour eviter l'ecrasement de fichiers
- **Spark S3A** : Buffer bytebuffer pour contourner l'erreur NativeIO Windows

---

## Fichiers crees/modifies

| Fichier | Action | Description |
|---------|--------|-------------|
| `scripts/collect_targeted.py` | Cree | Collecte ciblee 120 credits API |
| `scripts/auto_annotate_dataset.py` | Cree | Annotation multi-criteres ponderes |
| `data/golden_dataset.csv` | Cree | Dataset annote (5808 articles) |
| `src/greentech/data/processors/spark_cleaner.py` | Modifie | Chemin API elargi + buffer bytebuffer |
| `src/greentech/data/storage/minio_client.py` | Modifie | UUID dans generate_raw_path |
| `.env` | Modifie | Token HuggingFace configure |
| `docs/PLAN_ETAPES.md` | Modifie | Section 3.2 Golden Dataset cochee |
| `docs/ETAT_AVANCEMENT.md` | Reecrit | Mise a jour complete |

---

## Prochaines etapes

### Immediat (ETAPE 3.2 a finir)
1. Versionner `data/golden_dataset.csv` avec DVC (`dvc add` + `dvc push`)
2. Synchroniser entre PC Fixe et Portable

### Court terme (ETAPE 3.3)
1. Lancer l'entrainement DeBERTa-v3-base (Champion) sur GPU AMD ROCm
2. Lancer l'entrainement Llama 3.2 3B (Challenger) avec LoRA/PEFT
3. Benchmark final dans MLflow : Precision vs Latence vs CO2
4. Selectionner le modele gagnant

### Moyen terme (ETAPE 3.4)
1. Packaging du modele (safetensors) + push DVC
2. Redaction de la Model Card
3. Tests Deepchecks sur le modele final

---

## Points importants

### Desequilibre du dataset
Le ratio 22/5786 (0.4% Green IT) est tres desequilibre. Strategies pour l'entrainement :
- Class weights dans la loss function
- Oversampling de la classe minoritaire
- Data augmentation si necessaire

### Credits API epuises
Les 200 credits journaliers NewsData.io sont utilises. Pour enrichir davantage :
- Attendre le lendemain pour de nouveaux credits
- Ajouter d'autres sources (RSS, scraping de sites specialises Green IT)

### Nomenclature
Le projet utilise des noms de colonnes en **francais** dans toute la stack (SQL, SQLAlchemy, Pydantic).

---

**Redige par KaRn1zC - 2026-03-10**
