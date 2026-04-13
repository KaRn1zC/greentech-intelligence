# Procedure de Mise a Jour du Modele IA en Production

>
> Cette procedure permet de deployer un nouveau modele IA sans interruption de service.

---

## Architecture actuelle

- **Modele en production** : Llama 3.2 3B + LoRA (adapter_model.safetensors, 18 Mo)
- **Emplacement** : `models/production/`
- **Versioning** : DVC (remote MinIO s3://models/dvc)
- **Chargement** : Lazy-loading au premier appel d'inference via `get_classifier()`

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
- `models/challenger-llama/` : adapter LoRA entraine
- `models/cv_report.json` (uniquement en mode train-cv) : metriques par fold + agregees
- `models/baseline_metrics.json` : reference permanente du modele brut
- MLflow : tracking des runs (http://localhost:5000)

Verifier dans le rapport ou MLflow :
- **MCC** (Matthews Correlation Coefficient) : **critere principal**, robuste au
  desequilibre. Doit etre strictement superieur a 0 (mieux que l'aleatoire).
- **Recall Green IT** : doit etre >= 0.5 (garde-fou metier pour ne pas rater les positifs).
- **F1 score** : au plus -5% par rapport au meilleur modele historique.
- **Stabilite entre folds** (K-fold uniquement) : std du MCC <= 0.15.
- **Temps d'inference** : acceptable (< 15s sur CPU).
- **Empreinte carbone** : documentee (CodeCarbon).

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
| 4 | `std(MCC) entre folds <= 0.15` | `MAX_MCC_STD = 0.15` | Stabilite CV (applique si K-fold utilise) |

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

