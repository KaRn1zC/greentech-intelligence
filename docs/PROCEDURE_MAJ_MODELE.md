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

## Etape 1 : Entrainer le nouveau modele

```bash
# Lancer l'entrainement (exemple avec un nouveau challenger)
uv run python -m greentech.ai.models.training challenger-llama

# Benchmark pour comparer avec le modele actuel
uv run python -m greentech.ai.models.training benchmark
```

Verifier dans MLflow (http://localhost:5000) :
- Le F1 score du nouveau modele est **superieur** au modele actuel (0.667)
- Le temps d'inference est **acceptable** (< 15s sur CPU)
- L'empreinte carbone est **documentee** (CodeCarbon)

---

## Etape 2 : Valider avec Deepchecks

```bash
# Executer la suite de tests de validation
uv run pytest tests/ai/ -v
```

Verifier que :
- Pas de data leakage detecte
- Pas de biais significatif
- Robustesse au bruit acceptable

---

## Etape 3 : Packager le modele

```bash
# Copier les artefacts du nouveau modele
cp -r models/challenger-llama/* models/production/

# Mettre a jour la Model Card
# Editer models/production/README.md avec les nouvelles metriques

# Versionner avec DVC
dvc add models/production/
dvc push
```

---

## Etape 4 : Deployer sans interruption (Blue-Green)

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

## Etape 5 : Verifier en production

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

- [ ] Nouveau modele entraine et benchmark > modele actuel
- [ ] Tests Deepchecks passes
- [ ] Model Card mise a jour
- [ ] DVC push effectue
- [ ] Image Docker reconstruite
- [ ] Health check OK apres deploiement
- [ ] Inference de test reussie
- [ ] Monitoring Grafana verifie (pas de regression)
- [ ] Rollback teste ou documente

