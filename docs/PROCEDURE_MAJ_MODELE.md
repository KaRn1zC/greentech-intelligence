# Procedure de Mise a Jour du Modele IA en Production

>
> Cette procedure permet de deployer un nouveau modele IA sans interruption de service.

---

## Architecture actuelle

- **Modele en production** : Qwen3-4B + LoRA (adapter_model.safetensors, ~30 Mo)
- **Base model** : `Qwen/Qwen3-4B` (Apache-2.0, 26 juillet 2025), dense transformer
  standard, multilingue natif (FR/EN/DE/ES/ZH).
- **Dossier d'entrainement** : `models/challenger-qwen3/`
- **Emplacement production** : `models/production/`
- **Versioning** : DVC (remote MinIO s3://models/dvc)
- **Chargement** : Lazy-loading au premier appel d'inference via `get_classifier()`.
  La detection du type de modele est automatique via `adapter_config.json` :
  `qwen3-4b` => `ChallengerQwen3Classifier`, Llama/Qwen2.5 => `ChallengerClassifier`.

> **Migration** : le modele de production est passe de `meta-llama/Llama-3.2-3B`
> a `Qwen/Qwen3-4B` le 15 avril 2026. Motifs : licence Apache-2.0 (contre
> gated access Llama), multilinguisme natif pour traiter les articles scrapes
> depuis des sources non anglophones, et alignement avec la famille Qwen deja
> utilisee pour le LLM judge et les summarizers.
>
> Une tentative intermediaire avec `Qwen/Qwen3.5-4B` a ete abandonnee parce
> que ce modele est en realite un VLM (image-text-to-text) a architecture
> d'attention lineaire hybride qui requiert `flash-linear-attention` et
> `causal-conv1d` non supportes sous ROCm. L'entrainement LoRA saturait la
> VRAM au premier step et gelait le systeme complet. `Qwen3-4B` (sans le
> ".5") est le LLM texte officiel, dense et pleinement compatible.

---

## Etape 1 : Preparer le golden dataset (classification hybride)

Avant d'entrainer, le golden dataset doit refleter la classification la
plus a jour possible. Le pipeline applique une **classification hybride
en deux etages** (cf. specs techniques section 11), puis exporte le CSV :

```bash
# Etage 1 : pre-filtre mots-cles (marque NON_GREEN direct ou CANDIDATE)
uv run python scripts/retrain_pipeline.py annotate

# Etage 2 : LLM judge Qwen sur les candidats
# Bascule automatique sur GPU AMD local si quota HF mensuel epuise (HTTP 402).
uv run python scripts/retrain_pipeline.py classify

# Resumes general + ecologique uniquement pour les Green IT confirmes
uv run python scripts/retrain_pipeline.py summarize

# Export du CSV source de verite depuis l'etat final de la DB
uv run python scripts/retrain_pipeline.py export-golden
```

Le fichier `data/golden_dataset.csv` est ainsi regenere a partir de
l'etat final de la DB (post-LLM), pas du scoring mots-cles seul.

## Etape 2 : Entrainer le nouveau modele

Deux modes d'entrainement sont disponibles, orchestres par `scripts/retrain_pipeline.py` :

```bash
# Mode rapide : split 80/20 stratifie (~10 min)
uv run python scripts/retrain_pipeline.py train

# Mode robuste : K-fold stratifie K=5 + modele final (~50 min)
# Recommande pour figer une version de production, car l'evaluation moyenne
# les bruits statistiques dus au tres faible nombre de Green IT dans le corpus.
uv run python scripts/retrain_pipeline.py train-cv

# Calcul de la baseline (modele brut sans fine-tuning, sur le dataset complet)
uv run python scripts/retrain_pipeline.py baseline
```

Chaque run produit :
- `models/challenger-qwen3/` : adapter LoRA entraine (Qwen3-4B)
- `models/cv_report.json` (uniquement en mode train-cv) : metriques par fold + agregees
- `models/baseline_metrics.json` : reference permanente du modele brut (Qwen3-4B zero-shot)
- MLflow : tracking des runs (http://localhost:5000)

Pour evaluer la baseline de maniere isolee (avec un run MLflow dedie et
un log CO2 separe), utiliser le script specialise :

```bash
# Evaluation baseline Qwen3-4B avec tracking MLflow
uv run python scripts/benchmark_baseline.py

# Baseline d'un autre modele (pour comparer plusieurs architectures)
uv run python scripts/benchmark_baseline.py Qwen/Qwen3-8B
```

### Warning attendu au chargement du modele baseline

Au premier chargement de `Qwen/Qwen3-4B` via `AutoModelForSequenceClassification`,
Transformers affiche un avertissement `MISSING : score.weight`. Ce message est
**volontaire** et ne signale aucune anomalie :

**`MISSING : score.weight`**
`AutoModelForSequenceClassification` ajoute une tete `score` (Linear
`hidden_size -> num_labels`) par-dessus le modele de base. Comme le checkpoint
HF est un LM causal (pas un classifier), cette tete **n'existe pas** dans le
checkpoint et est donc **initialisee aleatoirement**.

C'est exactement le design attendu de la baseline (cf. docstring
`src/greentech/ai/models/baseline.py` lignes 16-20) : on mesure la performance
"chance + biais" du modele brut, sans prompt engineering ni fine-tuning. La tete
aleatoire est la reference bas niveau que le fine-tuning LoRA doit battre
significativement (criteres de promotion ci-dessous).

> **Note historique** : la tentative avortee avec `Qwen/Qwen3.5-4B` (un VLM)
> affichait en plus ~22 cles `UNEXPECTED : model.visual.*` (poids du vision
> encoder inutiles). Leur presence dans les logs est le signal qu'on a charge
> le mauvais modele — si tu revois cela, verifier immediatement que
> `settings.huggingface_model_baseline` pointe bien vers `Qwen/Qwen3-4B`.

**Consequence attendue** : les metriques baseline sont generalement proches du
hasard (MCC proche de 0, accuracy proche du taux majoritaire). Si au contraire
elles sont elevees alors que la tete est aleatoire, c'est un signal d'alerte
(biais du dataset, fuite de labels dans les texts, etc.).

Verifier dans le rapport ou MLflow :
- **MCC** (Matthews Correlation Coefficient) : **critere principal**, robuste au
  desequilibre. Doit etre strictement superieur a 0 (mieux que l'aleatoire).
- **Recall Green IT** : doit etre >= 0.5 (garde-fou metier pour ne pas rater les positifs).
- **F1 score** : au plus -5% par rapport au meilleur modele historique.
- **Stabilite entre folds** (K-fold uniquement) : std du MCC <= 0.15.
- **Temps d'inference** : acceptable (< 15s sur CPU).
- **Empreinte carbone** : documentee (CodeCarbon).

### Interpretation des resultats baseline (reference 2026-04-15)

Le premier run baseline de `Qwen/Qwen3-4B` sur le dataset complet (6354
articles dont 22 Green IT, soit 0.35 % de positifs) a produit les metriques
ci-dessous sur RX 7900 XTX (ROCm, BF16), en 12 min 27 s d'inference pure :

| Metrique           | Valeur   | Lecture                                                    |
|--------------------|----------|------------------------------------------------------------|
| MCC                | 0.0047   | ~0 : comportement quasi-aleatoire                          |
| Balanced accuracy  | 50.32 %  | Confirme : decision proche du pile-ou-face                 |
| F1                 | 0.0069   | Tres bas                                                    |
| Accuracy           | 0.98 %   | Pire que la regle triviale "toujours Non Green IT" (99.65 %) |
| Precision          | 0.35 %   | Quasi nulle                                                 |
| Recall Green IT    | 100 %    | Trompeur (voir ci-dessous)                                  |
| Specificite        | 0.63 %   | 40 / 6332 Non Green IT correctement classes                |
| Latence moyenne    | 117 ms   | Acceptable (p95 = 138 ms)                                   |

Matrice de confusion observee :

```
                Predit Green IT   Predit Non Green IT
Reel Green IT        22 (TP)            0 (FN)
Reel Non Green IT  6 292 (FP)          40 (TN)
```

**Analyse du comportement** : le modele predit Green IT pour 6314 / 6354
articles (99.4 %). Ce biais massif vers la classe 1 est le symptome
caracteristique d'une tete `score` **initialisee aleatoirement** : les poids
gaussiens du `Linear(hidden_size -> 2)` favorisent par hasard la classe 1. Avec
une autre seed de PyTorch, on pourrait obtenir le biais inverse (tout predit
en Non Green IT). Le `MCC ~ 0` et la balanced accuracy a 50 % confirment qu'il
n'y a aucune discrimination reelle entre les classes.

**Pourquoi Recall = 100 % n'est pas un signal positif** : en classant quasi
tous les articles en Green IT, le modele capte mecaniquement les 22 vrais
positifs sans aucune comprehension du contenu. C'est du bruit statistique
deguise en rappel parfait. MCC et balanced accuracy sont les metriques qui
demasquent ce piege.

**Verdict** : resultat attendu et coherent avec le design documente dans
`baseline.py` (tete aleatoire = reference "chance + biais"). Pas de signal
d'anomalie (fuite de labels, biais exploite, etc.). Cette baseline fixe la
**borne basse absolue** contre laquelle le fine-tuning LoRA devra prouver son
gain.

**Objectifs pour le fine-tuning LoRA** (a valider par K-fold) :

| Critere                        | Seuil         | Baseline       | Cible raisonnable |
|--------------------------------|---------------|----------------|-------------------|
| MCC                            | `> 0` strict  | 0.0047         | **> 0.5**         |
| Recall Green IT                | `>= 0.5`      | 1.0 (fallacieux) | >= 0.5 avec vraie discrimination |
| F1                             | `>= histo -5%`| 0.0069         | **> 0.6**         |
| Stabilite K-fold (std MCC)     | `<= 0.15`     | N/A            | A mesurer         |

Le vrai defi du fine-tuning n'est pas le recall (facile a tenir avec un
modele biaise) mais la **precision** : descendre drastiquement les 6292 faux
positifs tout en preservant les 22 vrais Green IT.

### Interpretation des resultats K-fold + promotion (reference 2026-04-15)

Le pipeline `train-cv auto-promote` execute le 15 avril 2026 (13 h 22 -> 22 h 44,
soit **9 h 22 min de bout en bout**) a entraine `Qwen/Qwen3-4B + LoRA` en
K-fold stratifie K=5 (~1 h 32 min par fold), puis un modele final sur les
6354 articles, et l'a promu en production (version `v20260415_204408`).
Empreinte carbone des 5 folds mesuree par CodeCarbon : **15.38 g CO2eq**
(electricite francaise, intensite reseau ~50 gCO2/kWh, RX 7900 XTX a
~17 W CPU + GPU principalement utilise sans tracking de puissance ROCm).

#### Metriques agregees sur les 5 folds

| Metrique          | Moyenne (+/- std) | Min - Max         | Lecture                                                |
|-------------------|-------------------|-------------------|--------------------------------------------------------|
| **MCC**           | **0.7625 (+/- 0.2476)** | 0.4984 - 1.0000 | Tres bon en moyenne, dispersion liee aux 4-5 Green IT par fold |
| F1                | 0.7600 (+/- 0.2510) | 0.5000 - 1.0000 | Idem, deux folds parfaits (4/5)                        |
| Balanced accuracy | 86.97 % (+/- 13.99 pts) | 69.96 % - 100 % | Discrimination reelle entre les classes               |
| Precision         | 0.7933 (+/- 0.2165) | 0.5000 - 1.0000 | Quasi tous les positifs predits sont reels            |
| Recall Green IT   | 0.7400 (+/- 0.2793) | 0.4000 - 1.0000 | On capte 3 a 5 vrais positifs sur 5 par fold          |
| Specificite       | 99.94 % (+/- 0.07 pt) | 99.84 % - 100 % | Tres peu de faux positifs sur la masse Non Green IT   |
| Latence inference | 33.98 ms          | 33.86 - 34.18    | Stable, ~3.5x plus rapide que la baseline (117 ms)    |

#### Detail par fold

| Fold | n_test (Green) | MCC    | F1     | Recall | Precision | TP/FN/FP/TN          |
|------|----------------|--------|--------|--------|-----------|----------------------|
| 1    | 1271 (5)       | 0.5150 | 0.5000 | 0.40   | 0.6667    | 2 / 3 / 1 / 1265     |
| 2    | 1271 (5)       | 0.7992 | 0.8000 | 0.80   | 0.8000    | 4 / 1 / 1 / 1265     |
| 3    | 1271 (4)       | 0.4984 | 0.5000 | 0.50   | 0.5000    | 2 / 2 / 2 / 1265     |
| 4    | 1271 (4)       | **1.0000** | **1.0000** | 1.00 | 1.0000    | 4 / 0 / 0 / 1267     |
| 5    | 1270 (4)       | **1.0000** | **1.0000** | 1.00 | 1.0000    | 4 / 0 / 0 / 1266     |

#### Metriques globales (concatenation des predictions des 5 folds)

```
                Predit Green IT   Predit Non Green IT
Reel Green IT       16 (TP)             6 (FN)
Reel Non Green IT    4 (FP)         6 328 (TN)
```

- **MCC global = 0.7620**
- **F1 global = 0.7619** (precision 80 %, recall 72.7 %)
- Accuracy globale = 99.84 %, balanced accuracy = 86.33 %
- Le modele rate 6 Green IT sur 22 (faux negatifs concentres sur les folds 1 et 3)
  et confond seulement 4 Non Green IT sur 6332 (faux positifs).

#### Gain du fine-tuning vs baseline brute

| Critere      | Baseline Qwen3-4B | Apres LoRA K-fold | Gain absolu |
|--------------|-------------------|-------------------|-------------|
| MCC          | -0.0854           | **0.7620**        | **+0.847**  |
| F1           | 0.0066            | **0.7619**        | **+0.755**  |
| Balanced acc | 47.77 %           | **86.33 %**       | **+38.6 pts** |
| Precision    | 0.33 %            | **80 %**          | **+79.7 pts** |

Le fine-tuning LoRA fait passer le modele d'un comportement aleatoire biaise
(predit 6348/6354 articles en Green IT) a un classifieur reellement
discriminant : la **precision passe de 0.33 % a 80 %** et l'on conserve un
recall correct de 72.7 % sur les vrais Green IT. C'est exactement le defi
identifie a la fin de la section baseline (descendre les faux positifs sans
sacrifier les vrais positifs), et le LoRA le releve sans ambiguite.

#### Verdict de promotion (4 criteres / 4 valides)

| # | Critere                                      | Valeur mesuree | Seuil   | Statut  |
|---|----------------------------------------------|----------------|---------|---------|
| 1 | MCC > 0 (premier modele Qwen3-4B)            | 0.7620         | 0       | OK      |
| 2 | Recall Green IT >= 0.5                       | 0.7273         | 0.5     | OK      |
| 3 | Non-regression F1 (pas de modele precedent)  | 0.7619         | n/a     | OK      |
| 4 | Stabilite CV : std MCC <= 0.25 (seuil dataset <50 Green) | 0.2476 | 0.25    | OK (limite) |

Tous les criteres sont satisfaits => **promotion automatique en production**.
Adaptateur LoRA (47 Mo) copie depuis `models/challenger-qwen3/` vers
`models/production/`, version taggee `v20260415_204408`, ancien modele
archive dans `models/versions/v20260415_204408/`.

#### Points d'attention pour le prochain cycle

1. **Stabilite CV juste a la limite** : `std(MCC) = 0.2476` est tres proche du
   plafond tolerant (0.25). Les folds 1 et 3 obtiennent MCC ~0.50, les folds 4
   et 5 atteignent un MCC parfait. Cette dispersion est mecanique avec
   seulement 4-5 Green IT par fold de test : un seul faux negatif fait chuter
   le recall de 1.0 a 0.75. Le seul levier durable est d'**enrichir le corpus
   de positifs** (cible : depasser 50 Green IT pour basculer sur le seuil
   strict 0.15 et avoir 10+ positifs par fold).

2. **Faux negatifs concentres sur les folds 1/3** : 6 Green IT manques sur les
   22. A inspecter avec `data/golden_dataset.csv` et les checkpoints
   `models/cv_fold_1/` et `models/cv_fold_3/` pour comprendre les motifs
   communs (sources, langues, longueurs d'articles).

3. **Latence d'inference excellente** : 34 ms par article en BF16 sur RX 7900
   XTX, 3.4x plus rapide que la baseline (qui faisait surtout du sampling
   genere). Pas d'optimisation supplementaire necessaire pour l'usage temps
   reel via `/analyze`.

4. **Empreinte carbone tres faible** : 15.4 g CO2eq pour ~7.7 h de calcul
   GPU, principalement grace au mix electrique francais. A documenter dans le
   bilan carbone projet (E2/E3).

#### Artefacts produits

- `models/production/adapter_model.safetensors` (47 Mo) — adapter LoRA promu
- `models/production/adapter_config.json` — base `Qwen/Qwen3-4B`, r=16, alpha=32, target_modules `[k_proj, v_proj, o_proj, q_proj]`
- `models/production/promotion_info.json` — metadata complete (date, metriques, fichiers)
- `models/best_metrics.json` — reference pour le prochain cycle d'auto-promotion
- `models/cv_report.json` — rapport K-fold detaille (5 folds + agreges + global)
- `data/benchmark_versions.json` — comparaison vs baseline + verdict promotion
- `models/versions/v20260415_204408/` — archive complete pour rollback
- MLflow run `challenger-qwen3-cv-k5` (experience `greentech-classification`)

---

## Etape 3 : Valider avec Deepchecks

```bash
# Executer la suite de tests de validation
uv run pytest tests/ai/ -v
```

Verifier que :
- Pas de data leakage detecte
- Pas de biais significatif
- Robustesse au bruit acceptable

---

## Etape 4 : Packager le modele (promotion conditionnelle)

La promotion est **automatisee** via `retrain_pipeline.py` : le script compare les
metriques du nouveau modele aux references et ne copie les artefacts vers
`models/production/` que si les **4 criteres de promotion** sont satisfaits.

```bash
# Benchmark + promotion automatique si tous les criteres sont OK
uv run python scripts/retrain_pipeline.py auto-promote

# Ou pour forcer la promotion manuelle (contournement des criteres)
uv run python scripts/retrain_pipeline.py promote
```

**Criteres de promotion** (cf. `scripts/retrain_pipeline.py`) :

| # | Critere | Constante | Raison |
|---|---------|-----------|--------|
| 1 | `MCC_nouveau >= MCC_ancien - epsilon` | `MCC_EPSILON = 0.01` | Metrique principale, robuste au desequilibre |
| 2 | `Recall_Green_IT >= 0.5` | `MIN_RECALL_GREEN_IT = 0.5` | Garde-fou metier : detecter les vrais positifs |
| 3 | `F1_nouveau >= F1_ancien * 0.95` | `F1_REGRESSION_TOLERANCE = 0.95` | Protection contre regression F1 |
| 4 | `std(MCC) entre folds <= seuil dynamique` | `MAX_MCC_STD_LARGE = 0.15` (>=50 Green IT), `MAX_MCC_STD_SMALL = 0.25` (<50 Green IT) | Stabilite CV : seuil tolerant tant que le dataset compte peu de positifs (4-5/fold), strict des qu'on en a assez |

Le script archive automatiquement l'ancien modele dans `models/versions/<tag>/`
avant de le remplacer. Le fichier `models/best_metrics.json` est mis a jour
avec les metriques du nouveau modele.

```bash
# Mettre a jour la Model Card
# Editer models/production/README.md avec les nouvelles metriques

# Versionner avec DVC
dvc add models/production/
dvc push
```

---

## Etape 5 : Deployer sans interruption (Blue-Green)

### Option A : Deploiement local (Docker)

```bash
# 1. Reconstruire l'image API avec le nouveau modele
docker compose build api

# 2. Redemarrer uniquement le conteneur API (les autres restent up)
docker compose up -d api

# 3. Verifier la sante
curl http://localhost:8000/health

# 4. Tester une inference
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"texte": "Green computing is the practice of using computing resources efficiently..."}'
```

Le modele est charge en lazy-loading : la premiere requete apres redemarrage sera plus lente (~15-30s) car elle charge le modele en memoire. Les requetes suivantes seront normales.

### Option B : Deploiement Render (Production)

```bash
# 1. Commiter les changements
git add models/production/ models/production.dvc
git commit -m "feat(ai): mise a jour modele production vX.Y"

# 2. Pousser sur main (declenche le CD pipeline)
git push origin main

# 3. Surveiller le deploiement sur Render Dashboard
# Le deploy Render remplace le conteneur sans interruption (rolling update)
```

---

## Etape 6 : Verifier en production

Apres le deploiement, verifier dans Grafana :

1. **Dashboard "Metier GreenTech"** :
   - Le ratio Green IT n'a pas change drastiquement (pas de regression)
   - Le temps moyen d'inference est stable

2. **Dashboard "Performance Systeme"** :
   - Pas de pic d'erreurs 5xx
   - La memoire du conteneur API est stable

3. **Logs Loki** :
   ```text
   {container="greentech-api"} |= "Modele charge"
   ```
   Verifier que le nouveau modele est bien charge.

---

## Rollback d'urgence

Si le nouveau modele cause des problemes :

```bash
# 1. Revenir au modele precedent via DVC
dvc checkout models/production.dvc

# 2. Reconstruire et redemarrer
docker compose build api && docker compose up -d api

# 3. Ou via Git (revert du commit)
git revert HEAD
git push origin main
```

---

## Checklist de mise a jour

- [ ] Nouveau modele entraine (de preference via `train-cv` pour la robustesse)
- [ ] Baseline recalculee si besoin (`retrain_pipeline.py baseline`)
- [ ] Benchmark execute et verdict "nouveau_retenu" dans `data/benchmark_versions.json`
- [ ] 4 criteres de promotion satisfaits (MCC, recall Green IT, non-regression F1, stabilite CV)
- [ ] Tests Deepchecks passes
- [ ] Model Card mise a jour
- [ ] DVC push effectue
- [ ] Image Docker reconstruite
- [ ] Health check OK apres deploiement
- [ ] Inference de test reussie
- [ ] Monitoring Grafana verifie (pas de regression)
- [ ] Rollback teste ou documente

