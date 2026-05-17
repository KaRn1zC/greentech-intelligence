# Sélection du champion 2026-04 : Qwen3-4B + LoRA (TIES)

> **Auteur** : KaRn1zC
> **Date** : 2026-05-17
> **Phase projet** : B4 (benchmark comparatif) → B5 (promotion production)
> **Documents complémentaires** : `docs/BENCHMARK_BRUT_2026-04.md` (baseline avant entraînement), `docs/CHOIX_DEBERTA.md` (justification du challenger), `docs/BENCHMARK_FINAL_2026-04.md` (résultats benchmark P5.1)

## 1. Contexte de la décision

À l'issue de la phase B4 (avril-mai 2026), deux candidats ont été entraînés via le protocole unifié K-fold stratifié sur le golden dataset post-Phase 2 (11 664 articles, 2 124 Green IT, EN 74.75 % / FR 25.25 %) :

- **Champion en titre** : Qwen3-4B + LoRA all-linear (decoder causal, 4B params, fine-tuné en K=3×2 réduit pour tenir dans 8h)
- **Challenger** : mDeBERTa-v3-base (encoder bidirectionnel, 278M params, fine-tuné en K=5×3 full)

Le présent document formalise le verdict de cette comparaison et acte la promotion de l'un des deux en production.

## 2. Méthodologie d'évaluation

### 2.1 Métrique principale : MCC K-fold

Le MCC (Matthews Correlation Coefficient) est la métrique de décision car :

- Robuste au déséquilibre (1:4.5 dans notre dataset)
- Inclut TP, TN, FP, FN dans son calcul (vs F1 qui ignore TN)
- Symétrique : `MCC(swap classes) = MCC`
- Sortie bornée [-1, 1] avec 0 = aléatoire, 1 = parfait

Le MCC est mesuré **par fold de validation**, jamais sur le train set, pour éviter le data leakage. Les 15 trainings mDeBERTa (5×3) et 6 trainings Qwen3 (3×2) produisent chacun une mesure MCC indépendante sur leur fold de validation, et on agrège mean ± std.

### 2.2 Métriques secondaires

- **F1, Precision, Recall** : pour caractériser la nature des erreurs (FP vs FN)
- **Latence p50 / p95 / p99** : exigence produit < 200 ms par article
- **VRAM peak** : contrainte hardware production
- **Empreinte CO2** : mesure CodeCarbon (rapport `models/benchmark_final_metrics.json`)
- **Stabilité σ MCC** : seuil critique 0.05 ; au-delà, le modèle est trop sensible au choix de seed et n'est pas reproductible

### 2.3 Benchmark P5.1 (validation croisée vs production)

Le `scripts/benchmark_models.py` exécute l'inférence des deux modèles sur l'intégralité du dataset (11 664 articles) avec :

- Modèles chargés via `get_classifier()` (point d'entrée production)
- Calibration moyenne K-fold appliquée (temperature scaling + threshold tuning)
- Mesure latence et VRAM en conditions réelles d'inférence

**⚠️ Note critique sur le leakage** : chaque article a été vu en TRAIN par 4 des 5 modèles mDeBERTa (et 2 des 3 folds Qwen3). Les MCC du benchmark P5.1 sont donc **optimistes**. Pour la décision, on s'appuie sur les **MCC K-fold honnêtes** (mesurés sur les folds de validation, sans leakage). Le benchmark P5.1 sert à :

1. Comparer les deux modèles sur la **même** distribution
2. Mesurer les **latences réelles** sur volume
3. Vérifier la **VRAM peak** observée
4. Détecter d'éventuelles régressions inattendues entre les MCC K-fold et les MCC benchmark

## 3. Résultats comparatifs

### 3.1 Métriques K-fold honnêtes (décision)

| Critère | Qwen3-4B TIES (K=3×2) | mDeBERTa K=5×3 | Écart |
|---------|----------------------|------------------|-------|
| **MCC** | **0.6238** | 0.5941 | **+0.030 (+5.0%)** |
| F1 | 0.6861 | 0.6600 | +0.026 |
| Recall | 0.8913 | 0.8926 | -0.001 (équivalent) |
| Precision | 0.5573 | 0.5243 | +0.033 |
| Balanced accuracy | 0.8617 | 0.8558 | +0.006 |
| σ MCC (stabilité) | ~0.010 | 0.0093 | équivalent |

Qwen3 dépasse mDeBERTa sur le critère principal (MCC) avec un écart de +0.030 et une stabilité équivalente. Le rappel est identique (~0.89), mais Qwen3 améliore la précision de +0.033, ce qui réduit le taux de faux positifs.

### 3.2 Métriques benchmark P5.1 (avec leakage, comparaison homogène)

| Critère | Qwen3-4B TIES | mDeBERTa K=5×3 | Écart |
|---------|---------------|------------------|-------|
| MCC | **0.7565** | 0.6066 | +0.150 (+24.7%) |
| F1 | **0.7945** | 0.6679 | +0.127 |
| Recall | **0.9557** | 0.9143 | +0.041 |
| Precision | **0.6798** | 0.5261 | +0.154 |
| Specificity | **0.8998** | 0.8167 | +0.083 |

L'écart amplifié sur le benchmark (+0.150 MCC vs +0.030 K-fold) suggère que Qwen3 **bénéficie davantage du leakage** que mDeBERTa : il mémorise mieux les exemples vus en TRAIN. C'est cohérent avec sa capacité 14x supérieure (4000M vs 278M params). Le bon chiffre pour la communication externe reste le K-fold honnête (+0.030).

### 3.3 Performances opérationnelles

| Critère | Qwen3-4B TIES | mDeBERTa K=5×3 | Écart |
|---------|---------------|------------------|-------|
| Latence p50 | 59.5 ms | 66.0 ms | **-10% (Qwen3)** |
| Latence p95 | 83.5 ms | 71.4 ms | +17% (mDeBERTa) |
| Latence p99 | 87.5 ms | 74.7 ms | +17% (mDeBERTa) |
| VRAM peak | 7.68 GB | 2.79 GB | +175% (Qwen3) |
| Params actifs | 4022 M (1 modèle fusionné) | 1390 M (5 × 278 M) | +189% (Qwen3) |
| Inférences cumulées K-fold | 6 (TIES collapse en 1) | 5 (logit_average) | équivalent |

Les deux modèles tiennent largement la cible < 200 ms par article. Qwen3 est plus rapide en médian mais a une queue plus longue (p99 = 87.5 ms vs 74.7 ms pour mDeBERTa). VRAM Qwen3 trois fois supérieure mais reste sous les 8 GB, donc compatible avec les GPU prod cibles (A10 24GB, L4 24GB, RTX 4090 24GB).

## 4. Verdict : **Qwen3-4B + LoRA (TIES) promu champion**

### 4.1 Critères primaires (MCC, stabilité)

1. **MCC K-fold honnête** : Qwen3 +0.030 (significatif, hors marge d'erreur σ=0.010)
2. **Stabilité** : équivalente (σ MCC ≈ 0.010 pour les deux)
3. **Conclusion** : Qwen3 gagne sur le critère principal

### 4.2 Critères secondaires (UX, infrastructure)

| Critère | Vainqueur | Justification |
|---------|-----------|---------------|
| Précision (UX) | **Qwen3** | +0.033 K-fold → moins de faux positifs visibles |
| Recall | équivalent | 0.89 dans les deux cas |
| Latence médiane | **Qwen3** | -10% (59.5 ms vs 66 ms) |
| Latence queue p99 | mDeBERTa | -17% (74.7 ms vs 87.5 ms), mais les deux sous 200 ms |
| VRAM | mDeBERTa | 2.8 GB vs 7.7 GB, mais 8 GB OK sur GPU prod |
| Empreinte CO2 inférence | mDeBERTa | ~2.5x moins de paramètres actifs |
| Empreinte CO2 entraînement | mDeBERTa | full FT 278M < LoRA 4B sur durée comparable |
| Compatibilité ROCm AMD | équivalent | Les deux testés OK sur RX 7900 XTX |

### 4.3 Pondération finale

Le projet GreenTech Intelligence cible la **qualité de classification** comme valeur produit principale (un utilisateur veut savoir si un article est Green IT, pas juste "vite"). Latence et VRAM sont des contraintes à respecter, pas des objectifs à optimiser tant que la qualité reste insuffisante (MCC < 0.75).

Sur cette pondération :

- ✅ **MCC** : Qwen3 +5% relatif K-fold, +25% relatif benchmark — différence significative
- ✅ **Précision** : Qwen3 +3.3 points K-fold, +15.4 points benchmark
- ✅ **Latence** : sous 100 ms p95 pour les deux, contrainte respectée
- ✅ **VRAM** : sous 8 GB pour Qwen3, compatible cibles prod

**Décision : Qwen3-4B + LoRA TIES devient le champion en production.**

## 5. Plan de promotion en production

### 5.1 Artefacts à promouvoir

Source : `models/qwen3/` (résultat de `train_with_unified_protocol(model_type="qwen3")`)

Cible : `models/production/`

Fichiers nécessaires :

```text
models/production/
├── model.safetensors            # 8.04 GB, depuis merged/model.safetensors
├── config.json                  # depuis merged/config.json
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
├── ensemble_config.json         # strategy=ties_manual (point d'entrée inférence)
├── temperature.json             # calibration T=1.395 K-fold mean
└── optimal_threshold.json       # seuil=0.155 K-fold mean
```

### 5.2 Procédure DVC + MinIO

```bash
# Préparer le dossier production
mkdir -p models/production
cp models/qwen3/merged/* models/production/
cp models/qwen3/ensemble_config.json models/production/
cp models/qwen3/temperature.json models/production/
cp models/qwen3/optimal_threshold.json models/production/

# Versionner via DVC vers MinIO
uv run dvc add models/production
uv run dvc push -r minio_models

# Commit du .dvc file + push
git add models/production.dvc .dvc/config
git commit -m "feat(prod): promote Qwen3-4B TIES champion 2026-05-17"
git push
```

### 5.3 Tag Git de release

```bash
git tag -a v2026.05.17-prod-qwen3-ties -m "Champion Qwen3-4B + LoRA TIES, MCC K-fold=0.6238, benchmark=0.7565"
git push origin v2026.05.17-prod-qwen3-ties
```

### 5.4 Mise à jour API et monitoring

- **API FastAPI** : aucun changement code — `get_classifier()` lit `ensemble_config.json` et redirige vers `merged/` automatiquement
- **MLflow Model Registry** : promouvoir le run MLflow `qwen3-unified-k3-s2` en stage `Production` via UI ou CLI
- **Grafana dashboard** : vérifier que la métrique `greentech_classification_mcc` est mise à jour avec la nouvelle baseline (0.6238)
- **Loki alerte** : ajouter une règle "MCC en production < 0.55 sur 24h" comme garde-fou de régression

## 6. Limites et roadmap d'amélioration

### 6.1 Limites du champion sélectionné

1. **Précision encore basse (0.56 K-fold)** : ~44% des articles classés Green IT sont des faux positifs. La précision benchmark (0.68) est plus haute mais biaisée. À surveiller en production via Grafana.
2. **Test set non hold-out** : les MCC sont mesurés en K-fold ou en benchmark à leakage. Aucun test set vraiment indépendant n'existe — c'est la limite structurelle du dataset actuel.
3. **TIES sur K=3** : la moyenne de K=3 folds est moins robuste qu'une moyenne K=5. Pour le prochain retrain, augmenter K=5 si le temps de training Qwen3 peut être divisé par 2 (optimisations VRAM, FlashAttention sur AMD ROCm 8.x ?).
4. **Pas de calibration par sous-population** : la calibration temperature scaling est globale. Sur les articles FR (25% du dataset), elle peut être sous-optimale. À tester en P6 (refonte agentic).
5. **Coût VRAM 7.7 GB** : exclut le déploiement sur GPU < 16 GB (T4 16GB OK, mais pas L4 24GB en mode partagé). Migration vers quantization 4-bit (bitsandbytes ou AWQ) pour la prochaine release majeure.

### 6.2 Roadmap post-promotion

| Phase | Action | Métrique cible | Échéance |
|-------|--------|----------------|----------|
| P4.11 | Générer résumés écologiques pour les 1106 nouveaux Green IT | Couverture 100% Green IT | Post-promotion immédiat |
| P5.5 | Validation end-to-end API + 10 analyses + dashboard Grafana | Toutes routes 200 OK | Post-promotion immédiat |
| P5.6 | MODEL_CARD.md final + tag Git v2026.05.17 | Doc complète + release publique | Post-validation |
| P6 (optionnel) | Refonte agentic LangGraph (B5) avec sous-modèles spécialisés EN/FR | MCC > 0.70 par sous-population | Q3 2026 |
| Future | Quantization 4-bit AWQ pour réduire VRAM à ~2 GB | VRAM < 3 GB sans perte MCC | Q4 2026 |
| Future | Hold-out test set 10% (reservé du prochain retrain) | MCC honnête non-biaisé | Prochain retrain |

## 7. Reproductibilité

### 7.1 Commande complète pour reproduire le champion

```bash
# Hardware : RX 7900 XTX 24 GB + ROCm 7.2.1
# Durée estimée : ~12h pour K=3×2 Qwen3 (P4.13 stratégie hybride)

# 1. Synchroniser le dataset (DVC pull)
uv run dvc pull data/golden_dataset_augmented.csv.dvc

# 2. Lancer le pipeline complet
uv run python scripts/retrain_pipeline.py train-cv --model=qwen3

# 3. Vérifier les artefacts
ls models/qwen3/merged/   # doit contenir model.safetensors + config.json
cat models/qwen3/ensemble_config.json | jq '.strategy'  # "ties_manual"
```

### 7.2 Artefacts versionnés

- **Code** : tag git `v2026.05.17-prod-qwen3-ties`
- **Dataset** : `data/golden_dataset_augmented.csv.dvc` SHA `38104c7346a5b360`
- **Modèle** : `models/production.dvc` versionné dans MinIO bucket `models`
- **MLflow runs** : expérience `greentech-classification`, run `qwen3-unified-k3-s2`
- **Métriques benchmark** : `models/benchmark_final_metrics.json` + `docs/BENCHMARK_FINAL_2026-04.md`

## 8. Références

- Matthews, "Comparison of the predicted and observed secondary structure of T4 phage lysozyme", Biochim. Biophys. Acta 1975 (origine MCC)
- Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023 (arXiv:2306.01708)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022 (arXiv:2106.09685)
- Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", ICML 2022 (arXiv:2203.05482)
- Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA", 2023 (arXiv:2312.03732, rsLoRA)
