# Guide d'annotation manuelle — audit B2.10 (avril 2026)

Ce guide accompagne l'annotation manuelle des articles **borderline**
identifies par le LLM judge (Qwen) lors de l'etape 2 du pipeline hybride
de classification. Son objectif est d'ameliorer la qualite du ground
truth avant l'entrainement final des modeles (Qwen3-4B et
mDeBERTa-v3-base, B3/B4).

---

## 1. Pourquoi une annotation manuelle ?

Le LLM judge Qwen retourne, pour chaque article, un triplet
`(est_green_it, confiance, raison)`. La **confiance** est un score dans
[0 ; 1] que le LLM auto-attribue a sa propre decision.

Sur les 5 574 articles classifies par le LLM, la distribution des
scores montre une zone d'incertitude substantielle :

| Score de confiance | Volume | Green IT | Non Green IT |
|---|---|---|---|
| [0.0 ; 0.3[ | 224 | 0 | 224 |
| **[0.3 ; 0.7]** | **1 325** | **0** | **1 325** |
| ]0.7 ; 0.8[ | 423 | 0 | 423 |
| [0.8 ; 1.0] | 4 025 | 1 018 | 3 007 |

**Observation cle** : aucun article classe Green IT n'a un score
inferieur a 0.8. Les 1 325 articles borderline sont **tous classes Non
Green IT par le LLM**, mais avec une confiance faible a moyenne. C'est
dans cette zone que se cachent les **faux negatifs** potentiels : des
articles que le LLM a hesite a classer Green IT mais qui en sont
reellement. Les corriger enrichit le dataset minoritaire et ameliore le
MCC attendu de l'entrainement final.

---

## 2. Pourquoi la fenetre [0.3 ; 0.7] et pas [0.3 ; 0.8] ?

Le seuil est calibre sur deux criteres :

1. **Distance a la decision de pile a face (0.5)**. Un score de 0.7
   correspond a une decision "plutot confiante" (70 % / 30 %). Au-dela,
   on considere que le LLM a tranche. En deca de 0.3 egalement — mais
   symetrisement, les scores < 0.3 sont des Non Green IT que le LLM
   rejette avec conviction, donc moins prioritaires a auditer.

2. **Volume annotable**. Etendre a [0.3 ; 0.8] ajoute 423 articles pour
   passer a 1 748 au total (+32 %). C'est jouable mais :
   - Ces 423 articles sont dans une zone plus confiante (scores 0.7-0.8).
     Probabilite d'erreur plus faible, ratio temps / gain moins bon.
   - Mieux vaut d'abord epuiser les 969 borderline GreenIT.fr (cf. ci-dessous)
     et mesurer le taux de correction avant d'elargir.

**Si apres l'audit [0.3 ; 0.7] le ratio de corrections est eleve
(> 20 %)**, elargir a [0.3 ; 0.8] est rentable — la commande est
```bash
uv run python scripts/manual_annotation_helper.py --score-max 0.8
```

---

## 3. Priorite : commencer par GreenIT.fr

Repartition des 1 325 borderline par source :

| Source | Borderline | Priorite |
|---|---|---|
| **GreenIT.fr** | **969** | **1** — site 100 % Green IT selon B2.1. Tout article non classe Green IT y est suspect. |
| Crossref | 176 | 2 — peer-reviewed journal articles, abstracts scientifiques precis |
| arXiv Dataset | 51 | 3 — preprints historiques |
| arXiv API | 45 | 3 — preprints recents |
| Green Software Foundation | 29 | 2 — site 100 % Green IT egalement |
| Sustainable Web Design | 19 | 2 — site 100 % Green IT |
| The Guardian | 12 | 4 — journalisme generaliste, decisions plus faciles |
| Dev.to | 9 | 4 |
| Climate Action Tech | 8 | 2 — 100 % tech + climat |
| TechCrunch Climate | 7 | 4 |

Commande recommandee pour l'audit prioritaire :

```bash
uv run python scripts/manual_annotation_helper.py --source "GreenIT.fr"
```

---

## 4. Definition du Green IT (criteres de decision)

Un article est classe **Green IT** s'il traite, de facon substantielle
(pas seulement en mention accessoire), d'au moins l'un des themes
suivants. Cette definition est rigoureusement identique a celle du
prompt systeme du LLM judge (`classifier_llm.py::CLASSIFIER_SYSTEM_PROMPT`)
pour eviter toute divergence d'interpretation entre annotations LLM et
annotations humaines.

### 4.1 Themes qui qualifient Green IT (inclusifs)

- **Infrastructures numeriques bas carbone** : reduction de la
  consommation energetique ou de l'empreinte carbone des data centers,
  cloud, reseaux.
- **Efficacite energetique du materiel IT** : serveurs, GPU,
  accelerateurs, puces basse consommation, refroidissement vert.
- **Sobriete numerique et eco-conception logicielle** : optimisation
  energetique de modeles IA/ML (quantization, pruning, distillation,
  compression visant l'energie), IA frugale.
- **Mesure, suivi, reporting** de l'empreinte carbone du numerique.
- **E-waste, economie circulaire** des equipements electroniques,
  refurbishing, durabilite du materiel, sustainable hardware.
- **Energies renouvelables DANS un contexte numerique** : data center
  solaire, cloud bas carbone, hydrogene vert pour data centers.
- **Usage du numerique pour la transition ecologique** — uniquement
  quand l'angle IT est clairement present, pas seulement une mention
  accessoire.

### 4.2 Themes exclus (Non Green IT)

- **Recherche IA/ML purement theorique** portant uniquement sur la
  precision ou la complexite algorithmique, sans consideration
  energetique.
- **Cryptomonnaies, rapports boursiers, previsions de marche**.
- **Cybersecurite pure, gaming, metaverse, reseaux sociaux,
  smartphones grand public** (meme si tech).
- **Energies renouvelables ou vehicules electriques SANS lien avec
  le numerique**.
- **Sujets sante / sciences appliquees** qui utilisent simplement de
  l'IA sans aborder son impact environnemental.

### 4.3 Regle du doute raisonnable

En cas de doute raisonnable, **classer Green IT** (`g`). C'est le meme
biais que le prompt LLM : mieux vaut un faux positif qu'un faux negatif
(la classe positive est rare, chaque Green IT manque est un gros manque).

**Exemples de cas limites** :

| Article | Decision | Justification |
|---|---|---|
| Paper `Efficient Transformer Training via Dynamic Quantization` (Crossref, score 0.55) | **Green IT** | Optimisation energetique implicite des modeles ML via quantization. |
| Article `Top 10 Cryptocurrencies to Watch in 2025` (Guardian, score 0.42) | **Non Green IT** | Crypto pur sans angle energetique. |
| Post `Comment j'ai reduit l'empreinte carbone de mon site WordPress de 40 %` (GreenIT.fr, score 0.62) | **Green IT** | Eco-conception logicielle concrete. |
| Paper `Deep Learning for Early Diagnosis of Cardiac Arrhythmias` (arXiv, score 0.35) | **Non Green IT** | IA en sante, pas d'impact environnemental. |
| Post `Smart Grids and the Future of Renewable Energy` (TechCrunch, score 0.48) | **Non Green IT** si pas d'angle IT. **Green IT** si parle d'optimisation du reseau energetique via data/AI. | Relire le contenu pour trancher. |
| Article `Apple Silicon M4 vs Intel Core Ultra : which is more energy-efficient ?` (Dev.to, score 0.60) | **Green IT** | Efficacite energetique du materiel IT. |

---

## 5. Utiliser l'outil CLI

L'outil `scripts/manual_annotation_helper.py` fournit une interface
interactive Rich qui affiche pour chaque article : titre, URL, source,
langue, decision LLM + score, raison LLM (si peuplee), resume de
classification, et extrait du contenu (1500 premiers caracteres).

### 5.1 Commandes usuelles

```bash
# Audit complet (toutes sources, ~1325 articles)
uv run python scripts/manual_annotation_helper.py

# Prioriser GreenIT.fr (recommande en premier)
uv run python scripts/manual_annotation_helper.py --source "GreenIT.fr"

# Session courte de 50 articles
uv run python scripts/manual_annotation_helper.py --limit 50

# Elargir la fenetre a 0.3-0.8 (si ratio de corrections eleve sur la fenetre 0.7)
uv run python scripts/manual_annotation_helper.py --score-max 0.8

# Prioriser les plus bas scores (les plus incertains)
# (deja le cas par defaut : tri par score ascendant au sein de chaque source)
uv run python scripts/manual_annotation_helper.py --source "GreenIT.fr" --limit 100

# Changer d'annotateur (pour sessions a plusieurs)
uv run python scripts/manual_annotation_helper.py --by "Daisy"
```

### 5.2 Raccourcis clavier par article

| Touche | Action |
|---|---|
| `g` | Classer **Green IT** (ou corriger Non Green IT → Green IT) |
| `n` | Classer **Non Green IT** (ou confirmer la decision LLM) |
| `s` | **Skip** cet article (pas d'ecriture BDD, sera a nouveau propose au prochain run) |
| `o` | **Ouvrir** l'URL dans le navigateur pour contexte, puis redemande la decision |
| `q` | **Quitter** proprement (les decisions deja prises sont persistees) |

### 5.3 Reprise de session

Les articles annotes sont marques `annotation_source='manual'` en base,
avec `annotated_at` et `annotated_by` renseignes. Le script les exclut
automatiquement au prochain lancement — on peut donc faire l'audit en
plusieurs sessions etalees sur plusieurs jours sans re-traiter les
memes articles.

Pour **annuler une annotation manuelle** (cas rare, ex: erreur de
saisie), il faut manuellement reinitialiser les colonnes en SQL :

```sql
UPDATE articles
SET annotation_source = 'llm_judge',
    annotated_at = NULL,
    annotated_by = NULL,
    est_green_it = <ancienne valeur LLM>
WHERE id_article = <id a corriger>;
```

La colonne `score_confiance` reste intacte meme apres une annotation
manuelle, donc on peut toujours recuperer l'etat LLM original.

---

## 6. Apres l'audit : etapes suivantes

Une fois les annotations manuelles terminees, lancer dans l'ordre :

### 6.1 Regenerer les resumes Green IT des faux negatifs corriges

Les articles passes de Non Green IT → Green IT n'ont pas de
`resume_ecologique`. Il faut lancer `generate_green_summaries.py` qui
detecte automatiquement les articles Green IT sans resume ecologique et
les produit via le LLM.

```bash
uv run python scripts/generate_green_summaries.py
```

### 6.2 Supprimer les resumes Green IT des faux positifs corriges

Symetriquement, les articles passes de Green IT → Non Green IT gardent
un `resume_ecologique` qui n'a plus lieu d'etre. **Optionnel** mais
propre :

```sql
UPDATE articles
SET resume_ecologique = NULL
WHERE annotation_source = 'manual'
  AND est_green_it = false
  AND resume_ecologique IS NOT NULL;
```

### 6.3 Re-exporter le golden dataset

```bash
uv run python scripts/export_golden_dataset.py
```

Puis versionner la nouvelle version via DVC :

```bash
uv run dvc add data/golden_dataset.csv
uv run dvc push
```

### 6.4 Regenerer l'augmentation back-translation (facultatif)

Si beaucoup de nouveaux positifs ont ete identifies, re-lancer la
back-translation pour etendre le set augmente :

```bash
uv run python scripts/augment_positives.py
```

### 6.5 Lancer le benchmark brut puis l'entrainement

Apres l'audit, le chemin vers le modele final est :

```bash
# Benchmark zero-shot des deux modeles cibles (mdeberta + qwen3)
uv run python scripts/retrain_pipeline.py baseline-both

# Entrainement K-fold protocole unifie B3 (~6-8h cumulees)
uv run python scripts/retrain_pipeline.py train-cv-both

# Benchmark comparatif final
uv run python scripts/retrain_pipeline.py benchmark-models
```

---

## 7. Conseils pour l'annotation efficace

- **Commencer par GreenIT.fr** (969 articles, priorite 1). Au sein
  d'une meme source, les articles sont tries par score croissant : les
  plus incertains en premier. C'est la zone ou les erreurs sont les
  plus probables.
- **Cibler des sessions de 30-60 min** (20-30 articles). Au-dela, la
  fatigue degrade la qualite des decisions. Le flag `--limit 30`
  discipline la duree.
- **Ouvrir l'URL (`o`) en cas de doute** : le contenu BDD est tronque
  a 1500 caracteres, le site original a souvent plus de contexte
  (images, schemas, commentaires).
- **Relire la definition Green IT (section 4)** toutes les 10
  annotations pour garder un ancrage mental stable. Le biais de derive
  est reel sur des sessions longues.
- **Rester coherent avec la regle du doute raisonnable** : en cas
  d'incertitude, classer Green IT (`g`). Un modele legerement
  sur-inclusif est plus facile a recalibrer via le temperature scaling
  (B3.4) qu'un modele sous-inclusif.

---

## 8. Tracabilite et RGPD

Chaque annotation manuelle laisse une trace complete en base :

- `annotation_source = 'manual'`
- `annotated_at` = timestamp UTC
- `annotated_by` = identifiant de l'annotateur (`KaRn1zC` par defaut)
- `est_green_it` = decision binaire (peut differer de la decision LLM
  originale, qu'on conserve via `score_confiance` et `modele_classification`)

Aucune donnee personnelle supplementaire n'est collectee. Le registre
RGPD reste inchange — seul le nom de l'annotateur (pseudonyme) est
persiste, pas d'email ni d'autre PII.

Si plusieurs annotateurs participent a l'audit, utiliser l'option
`--by "pseudo"` pour differencier les contributions. La mesure du taux
d'accord inter-annotateurs est possible a posteriori en requetant la
table.

---

*Guide redige pour l'etape B2.10 du projet GreenTech Intelligence.*
*Auteur : KaRn1zC.*
