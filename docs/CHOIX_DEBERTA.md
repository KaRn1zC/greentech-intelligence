# Choix du challenger encoder : mDeBERTa-v3-base

> **Auteur** : KaRn1zC
> **Date** : 2026-05-17
> **Phase projet** : B4 (benchmark comparatif)
> **Document complémentaire** : `docs/BENCHMARK_BRUT_2026-04.md` (mesure point zéro), `docs/BENCHMARK_FINAL_2026-04.md` (mesure post-entraînement), `docs/SELECTION_CHAMPION_2026-04.md` (verdict final)

## 1. Pourquoi un challenger encoder ?

Le champion de production en début de phase B4 est **Qwen3-4B + LoRA all-linear** (decoder causal de 4 milliards de paramètres). Pour valider scientifiquement ce choix face aux alternatives raisonnables, le projet B4 introduit un **modèle challenger** de famille différente :

- **Architecture** : encoder bidirectionnel, supposé mieux adapté à la classification de texte court (titre + résumé ~450 tokens) car il voit le contexte des deux côtés à chaque position
- **Taille** : un ordre de grandeur plus petit que Qwen3 (~300M vs 4 000M), pour mesurer le coût/bénéfice d'un decoder massif
- **Famille** : full fine-tuning standard (pas LoRA), pour vérifier qu'un FT classique sur un encoder pré-entraîné suffit à atteindre le SOTA sur ce dataset déséquilibré bilingue

Sans challenger, on ne peut pas distinguer ce qui vient de Qwen3 (l'architecture, la taille) de ce qui vient du fine-tuning bien fait. Le benchmark final (P5.1) tranche.

## 2. Alternatives considérées

### 2.1 Liste des candidats encoders multilingues

| Modèle | Params | Langues | Licence | Avantage | Inconvénient |
|--------|--------|---------|---------|----------|--------------|
| **mDeBERTa-v3-base** | 278 M | 100 (FR + EN natifs) | MIT | SOTA encoder 2023, disentangled attention, vocabulaire 250k | ELECTRA pre-training plus rare, support transformers parfois conditionnel |
| XLM-RoBERTa-base | 278 M | 100 | MIT | Largement testé, baselines abondantes en littérature | Moins performant que mDeBERTa sur GLUE/XGLUE benchmark, attention standard |
| mBERT (bert-base-multilingual-cased) | 178 M | 104 | Apache-2.0 | Référence historique, code mature | Architecture 2018 dépassée, MCC inférieur de 2-4 points sur classification multilingue |
| XLM-RoBERTa-large | 560 M | 100 | MIT | Souvent +1-2 MCC vs base | Coût VRAM (~5x base), latence ~3x |
| InfoXLM-base | 277 M | 94 | MIT | Aligné cross-lingue pour transfert | Moins diffusé, support tooling fragile |
| mT5-base | 580 M | 101 | Apache-2.0 | Seq2seq versatile | Inutile pour classif pure, surcoût inférence (decoder), pas le bon outil |

### 2.2 Critères de filtrage

1. **Multilinguisme natif FR + EN** : éliminé tout modèle EN-only (RoBERTa, DeBERTa-v3-base, ELECTRA-base anglais) car notre dataset est bilingue (EN 74.75 %, FR 25.25 %)
2. **Taille raisonnable** : cible 200-400M params pour un full FT abordable sur RX 7900 XTX 24 GB en bf16 (largeXL retire de la liste)
3. **Licence permissive** : MIT, Apache-2.0 ou équivalent (pas de modèle gated)
4. **Architecture encoder** : pour le contraste avec Qwen3 decoder (mT5 = decoder hybride, retiré)
5. **Support transformers stable** : pas d'architecture exotique nécessitant flash-attention ou ops non-supportées sous ROCm

## 3. Pourquoi mDeBERTa-v3-base spécifiquement

### 3.1 Architecture supérieure

mDeBERTa-v3 hérite des trois innovations clés de DeBERTa-v3 (He et al., ICLR 2021 + 2023) :

- **Disentangled attention** : sépare le contenu de la position relative dans le calcul d'attention, ce qui améliore la modélisation des dépendances longue distance par rapport au standard scaled dot-product (XLM-R, mBERT)
- **Enhanced mask decoder** : pré-entraînement avec un objectif RTD (Replaced Token Detection, type ELECTRA) plus efficace que le MLM standard de XLM-R en termes de compute
- **Vocabulaire SentencePiece 250k** : couverture multilingue large incluant français accentué sans surdécoupage

### 3.2 Performances de référence publiées

| Benchmark | mDeBERTa-v3-base | XLM-R-base | mBERT |
|-----------|-----------------|------------|-------|
| XNLI (15 langues, accuracy moyenne) | **79.8** | 75.0 | 65.7 |
| XCOPA (11 langues, accuracy moyenne) | **75.4** | 68.2 | 62.5 |
| WikiAnn-FR (NER F1) | **89.2** | 87.1 | 85.6 |

Sur les tâches de classification multilingue, mDeBERTa-v3-base dépasse XLM-R-base de +3 à +5 points en moyenne pour la même empreinte (278M params, ~600 MB VRAM en bf16). Pour la classification binaire Green IT FR + EN, ces gains se traduisent typiquement par +1 à +2 MCC absolus.

### 3.3 Confirmation par baseline zero-shot rigoureuse (P4.6)

Avant tout entraînement, la baseline linear probing (backbone gelé, régression logistique sklearn sur embeddings mean-pooled) a établi :

- **mDeBERTa-v3-base** : MCC = **0.4827** (5-fold CV)
- Qwen3-4B (last-token pooling) : MCC = 0.4156 (5-fold CV)

L'encoder produit des features mieux séparées en classification linéaire (cf. `docs/BENCHMARK_BRUT_2026-04.md`). Cela en faisait un challenger crédible et a justifié l'investissement dans son entraînement K=5×3 complet.

## 4. Hyperparamètres retenus pour le fine-tuning

### 4.1 Choix d'un full fine-tuning (pas LoRA)

À 278M paramètres et avec 24 GB VRAM disponible, un full fine-tuning rentre largement en mémoire avec bf16 + gradient checkpointing. LoRA n'apporterait pas de gain de mémoire significatif et perdrait ~0.5-1 MCC selon Hu et al. 2021 sur des modèles de cette taille (LoRA brille sur >1B params).

### 4.2 Configuration `MDeBERTaClassifier`

```python
# src/greentech/ai/models/training.py
class MDeBERTaClassifier(DeBERTaClassifier):
    base_model = "microsoft/mdeberta-v3-base"
    learning_rate = 2e-5             # linear warmup 0.06
    num_train_epochs = 5             # early stopping MCC patience 2
    per_device_train_batch_size = 16
    gradient_accumulation_steps = 2  # batch effectif 32
    max_length = 384                 # vs 512 pour DeBERTa EN (mDeBERTa vocab plus dense)
    weight_decay = 0.01
    warmup_ratio = 0.06
    bf16 = True                      # conditionnel transformers >= 4.48
    gradient_checkpointing = True
    # fp16 INTERDIT (= NaN sur DeBERTa, bug documenté HF)
```

### 4.3 Décisions critiques et leur justification

| Décision | Valeur | Justification |
|----------|--------|---------------|
| `learning_rate` | 2e-5 | DeBERTa converge mieux à lr bas que BERT/RoBERTa (cf. He et al. 2021), warmup linéaire 0.06 |
| `batch effectif` | 32 | Compromis stabilité gradient vs VRAM ; gradient_accumulation 2 plus simple à reproduire que gradient_checkpointing massif |
| `max_length` | 384 | mDeBERTa SentencePiece 250k → tokens plus denses qu'un BPE 50k. 384 mDeBERTa tokens ≈ 512 BPE tokens en couverture sémantique. Économise ~30% latence et VRAM |
| `bf16` (pas fp16) | conditionnel | DeBERTa-v3 produit des NaN systématiques en fp16 sur certaines couches (issue HF #15067). bf16 résout, fallback fp32 si transformers < 4.48 |
| `class_weight=[1.0, ~10.5]` | WeightedLossTrainer | Remplacement de l'oversampling x84 historique. Plus stable, plus rapide, équivalent statistique selon Chawla et al. 2002 |
| `MultilabelStratifiedKFold (langue × label)` | iterative-stratification | Garantit ~75% EN / ~25% FR dans chaque fold val avec ratio Green IT respecté. Sans cela, certains folds avaient < 5% FR et le modèle dégradait sur les articles français |

## 5. Workflow d'entraînement validé

### 5.1 Reproductibilité

```bash
# Lance le protocole unifié K=5 × 3 seeds sur mDeBERTa seul
uv run python scripts/retrain_pipeline.py train-cv --model=mdeberta

# Ou via le pipeline hybride P4.4 qui orchestre les deux modèles
uv run python scripts/train_p4_hybrid.py --mdeberta-only
```

Durée mesurée le 2026-05-17 : 188 min pour 15 trainings (12.5 min / training), sur RX 7900 XTX 24 GB + ROCm 7.2.1.

### 5.2 Résultats K-fold

| Métrique | Valeur | Écart-type | Stable ? |
|----------|--------|------------|----------|
| MCC | **0.5941** | 0.0093 | ✅ σ < 0.01 |
| F1 | 0.6600 | 0.0088 | ✅ |
| Recall | 0.8926 | 0.0281 | ✅ |
| Precision | 0.5243 | 0.0177 | ⚠️ relativement bas |
| Balanced accuracy | 0.8558 | 0.0071 | ✅ |

Stabilité σ MCC = 0.0093 nettement sous le seuil critique de 0.05, ce qui valide l'entraînement comme reproductible.

## 6. Limites identifiées

### 6.1 Précision basse (0.52)

mDeBERTa privilégie le rappel (0.89) au prix d'une précision faible (0.52). En production, cela se traduit par ~50% de faux positifs (articles non-Green IT classifiés Green IT). Le seuil moyen K-fold (0.599) tente de compenser mais le modèle reste structurellement biaisé vers la classe minoritaire à cause du `class_weight` agressif (×10.5).

### 6.2 Coût ensemble inférence

L'inférence en production utilise un ensemble logit_average sur les 5 folds top-1 (un modèle par fold gardé, celui avec le meilleur MCC). Cela représente :

- **VRAM** : 5 × 600 MB = ~3 GB
- **Latence** : 5 inférences séquentielles, mesurée à ~66 ms p50 / 71 ms p95
- **Code** : `EnsembleClassifier` dans `src/greentech/ai/models/inference.py`

Le surcoût ensemble apporte +1 à +2 MCC vs un seul fold, ce qui reste favorable.

### 6.3 Pas de SWA ni de TIES-merging

Le protocole appliqué à mDeBERTa **n'inclut pas** Stochastic Weight Averaging ni TIES-merging (les deux sont réservés à Qwen3 + LoRA dans `training.py`). Pour un encoder full FT, ces techniques d'ensemble par fusion de poids sont moins efficaces selon la littérature (Wortsman 2022 mesure des gains principalement sur des LoRA et des fine-tuning très courts).

## 7. Verdict — entrée au benchmark P5.1

Le challenger mDeBERTa-v3-base entre au benchmark final P5.1 avec :

- ✅ Architecture justifiée (SOTA encoder multilingue 2023)
- ✅ Hyperparams validés (recommandations He et al. 2021 + tests internes)
- ✅ Reproductibilité prouvée (σ MCC = 0.0093 sur 15 trainings)
- ✅ Baseline supérieure à Qwen3 (linear probing MCC 0.483 vs 0.416)
- ⚠️ Précision faible (0.52) — limite potentielle en production

Le verdict final (Qwen3 vs mDeBERTa) est documenté dans `docs/SELECTION_CHAMPION_2026-04.md`.

## 8. Références

- He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention", ICLR 2021 (arXiv:2006.03654)
- He et al., "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing", ICLR 2023 (arXiv:2111.09543)
- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (XLM-R), ACL 2020 (arXiv:1911.02116)
- Sechidis et al., "On the Stratification of Multi-Label Data", ECML 2011 (iterative-stratification reference)
- Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR 2002 (sur class_weight vs oversampling)
