# Model Card - Classifieur Green IT (GreenTech Intelligence)

> **Auteur** : KaRn1zC
> **Date de création** : 2026-05-16 (v1.0) — **mise à jour majeure 2026-05-17 (v2.0)**
> **Version du modèle documenté** : `v20260517_180500`
> **Statut** : **Modèle de production officiel** (promu suite au benchmark B4 du 17 mai 2026)

---

## 1. Identification du modèle

| Champ | Valeur |
|-------|--------|
| **Nom** | GreenTech Classifier — Qwen3-4B + LoRA TIES |
| **Tâche** | Classification binaire de texte (Green IT vs Non Green IT) |
| **Domaine d'application** | Articles technologiques (presse, blogs techniques, publications scientifiques) |
| **Langues supportées** | Anglais (74.75% du dataset), Français (25.25%) — multilingue natif Qwen3 |
| **Modèle de base** | `Qwen/Qwen3-4B` (Alibaba, licence Apache-2.0, publié le 26 juillet 2025) |
| **Adaptation** | LoRA (Low-Rank Adaptation) via la librairie PEFT 0.19.0 |
| **Stratégie d'ensemble** | **TIES-merging** (Yadav et al. NeurIPS 2023, arXiv:2306.01708) sur 3 folds top-1 |
| **Type de tâche PEFT** | `SEQ_CLS` (classification de séquence à 2 classes) |
| **Version promue** | `v20260517_180500` (promotion le 17 mai 2026) |
| **Emplacement** | `models/production/` (versionné DVC vers MinIO `s3://models/dvc`) |
| **Taille des poids** | **8.04 GB** (modèle merged base + LoRA fusionnés, format full `model.safetensors`) |
| **Référence sélection** | `docs/SELECTION_CHAMPION_2026-04.md` |

### 1.1 Architecture LoRA — protocole B3 unifié

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| Rang LoRA `r` | **16** | Sweet spot Unsloth 2026 pour Qwen3-4B (r=32 plus lent, gain marginal sur ce dataset) |
| `lora_alpha` | **32** | Ratio `alpha/r = 2` (canonique LoRA original) |
| `lora_dropout` | 0.05 | Régularisation modérée |
| `bias` | `none` | Standard LoRA |
| **`use_rslora`** | **`True`** | Rank-Stabilized LoRA (Kalajdzievski 2023, arXiv:2312.03732) — réduit la variance d'init |
| **Modules ciblés** | **`all-linear`** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) | Attention + MLP SwiGLU, conforme au protocole B3 d'avril 2026 |
| Modules sauvegardés | `score` (tête de classification `Linear(2560, 2)`) | Wrappé dans `ModulesToSaveWrapper` PEFT |
| Paramètres entraînables | **33M / 4022M** (~0.82 %) | Le delta des 504 tenseurs LoRA sur les 36 layers Qwen3 |

### 1.2 Stratégie d'ensemble — TIES manuel sur safetensors

PEFT 0.19.0 refuse `add_weighted_adapter(combination_type="ties")` quand un adapter contient un `modules_to_save` (la tête de classification ici). Pour contourner cette limitation, le projet implémente TIES **manuellement** sur les safetensors via `_merge_lora_adapters` dans `training.py` :

| Étape TIES (Yadav 2023, Algorithm 1) | Application au projet |
|---|---|
| **1. Trim** : garde la fraction `density` des paramètres avec la plus grande magnitude par tenseur | `density=0.5` sur les 504 tenseurs LoRA A/B |
| **2. Sign-elect** : signe gagnant par position via somme algébrique pondérée par magnitude | Appliqué élément par élément |
| **3. Disjoint merge** : moyenne arithmétique sur les contributions du signe gagnant uniquement | Évite les annulations destructives |
| **Tête de classification** | Moyenne arithmétique simple (pas TIES sur poids absolus) |

Le merge se fait sur les **3 adapters top-1 par fold** (sélection par MCC maximal), pas sur les 6 trainings.

---

## 2. Cas d'usage prévus

### 2.1 Usage principal

Classifier automatiquement des articles technologiques selon leur pertinence "Green IT" (informatique éco-responsable), pour alimenter le tableau de bord et les statistiques de l'application GreenTech Intelligence.

### 2.2 Pipeline d'inférence

Le modèle est appelé par l'API FastAPI dans la route `POST /analyze` après l'étape de génération du résumé de classification, dispatché via la queue Celery + Redis :

1. POST `/analyze` enqueue une tâche `classify_article` dans Redis (réponse 202 immédiate)
2. Le worker Celery (pool `solo`, concurrency=1) consomme la tâche
3. Extraction du contenu de l'article (URL, texte brut ou fichier uploadé)
4. **Génération du résumé** via `classification_summarizer.py` (max 450 tokens, abstract scientifique dense)
5. **Classification** par le modèle documenté ici (`titre + résumé` en entrée, max_length=512)
6. Application de la calibration : `softmax(logits/T)` puis seuil de décision
7. Si Green IT confirmé : génération d'un résumé écologique séparé via Qwen3-Instruct
8. Persistance des résultats en BDD (table `articles` + `analysis_logs`)

GET `/analyze/{job_id}` retourne le statut/résultat via `AsyncResult` Redis.

### 2.3 Public visé

Utilisateurs de la plateforme GreenTech Intelligence : étudiants, chercheurs, ingénieurs sensibilisés à l'écoconception numérique, journalistes spécialisés.

### 2.4 Usages hors périmètre

- **Pas de génération de texte** : le modèle est utilisé en mode classification (`AutoModelForSequenceClassification`), pas en mode causal LM. Les générations sont assurées par `Qwen3-4B-Instruct-2507` via API HF Serverless avec fallback `Qwen2.5-3B-Instruct` local.
- **Pas de notation continue** : sortie binaire (Green IT ou non) avec score de confiance ∈ [0, 1], mais ne constitue pas un index quantitatif d'écoresponsabilité.
- **Pas d'évaluation d'impact carbone** : le modèle classifie le contenu textuel, il ne mesure pas l'empreinte carbone réelle des solutions mentionnées.

---

## 3. Données d'entraînement

### 3.1 Source

`data/golden_dataset_augmented.csv` (snapshot du 17 mai 2026, **post-enrichissement B2 + back-translation B3**)

| Caractéristique | Valeur |
|-----------------|--------|
| Volume total | **13 111 articles** (11 664 originaux + 1 447 variantes augmentées) |
| Articles Green IT confirmés | **2 124 originaux + 1 447 augmentés = 3 571** |
| Articles Non Green IT | 9 540 |
| Ratio Green IT / Non Green IT (originaux) | **1 / 4.49** (18.21 % Green IT) |
| Répartition linguistique | EN 74.75 % / FR 25.25 % |
| Format de la feature | `titre + resume_classification` (résumé LLM Qwen3, max 450 tokens) |
| Label | Binaire (`est_green_it` ∈ {0, 1}) |
| Versioning | DVC vers MinIO `s3://models/dvc`, SHA-256 `38104c7346a5b360` |

### 3.2 Méthode d'annotation hybride (deux étages + audit manuel)

1. **Étage 1 — Pré-filtre mots-clés permissif** (`scripts/auto_annotate_dataset.py`) : scoring multi-critères, marque chaque article `NON_GREEN` ou `CANDIDATE`
2. **Étage 2 — LLM judge Qwen3-4B-Instruct** (`scripts/classify_candidates.py`) : tranche les candidats en zero-shot sur le contenu brut
3. **Phase 2 d'audit manuel** (mai 2026) : correction de ~110 borderline (30 GreenIT.fr + 76 Crossref + audit multi-agents), faux positifs/négatifs identifiés et reclassés
4. **Back-translation EN↔FR** (`scripts/augment_positives.py`) : opus-mt génère 1 447 variantes des positifs (ratio 1:6.3), exclues du val/test pour éviter le data leakage

### 3.3 Gestion du déséquilibre

Contrairement à la version précédente (oversampling x84), le protocole B3 utilise :

- **`class_weight = [1.0, ~5.49]`** sur la CrossEntropy (`WeightedLossTrainer`), pondère naturellement la classe minoritaire sans dupliquer les exemples
- **Back-translation EN↔FR** : ajoute de la diversité sémantique sur les positifs (paraphrase via Helsinki-NLP/opus-mt-fr-en et opus-mt-en-fr)
- **Stratification croisée `(langue × label)`** via `MultilabelStratifiedKFold` : garantit ~75% EN / ~25% FR avec distribution Green IT respectée dans chaque fold de validation

---

## 4. Procédure d'entraînement

### 4.1 Infrastructure

- **GPU** : AMD Radeon RX 7900 XTX 24 Go (architecture RDNA 3, `gfx1100`)
- **ROCm** : version 7.2.1 stable (wheels-only depuis le 18 avril 2026)
- **PyTorch** : 2.9.1+rocm7.2.1
- **Précision** : `bf16=True` (fp16 produit des NaN sur certaines couches DeBERTa, donc bannī globalement)
- **`gradient_checkpointing`** : `True` (`use_reentrant=False`)
- **`model.enable_input_require_grads()`** : appelé AVANT `get_peft_model()` (piège HF #42947)

### 4.2 Pipeline d'entraînement P4.4 hybride

Stratégie validée le 2026-05-17 face à la lenteur intrinsèque de Qwen3-4B sur RX 7900 XTX (~10s par optimizer-step) :

| Hyperparamètre | Valeur | Justification |
|----------------|--------|---------------|
| K-fold | **K=3** | Réduit (vs K=5 du protocole B3 standard) pour tenir en ~8h |
| Seeds par fold | **2** | Réduit (vs 3 standard) — variance σ MCC ≈ 0.010 reste acceptable |
| Total trainings | **6** | (K=3 × 2 seeds) |
| Stratification | `MultilabelStratifiedKFold (langue × label)` | iterative-stratification, garantit représentativité bilingue |
| Loss | `WeightedLossTrainer` (CrossEntropy pondérée) | Évite l'oversampling x84 historique |
| Optimizer | AdamW | Standard HF Trainer |
| Learning rate | **1e-4** | Cosine warmup ratio 0.06 |
| Batch size | 2 × `gradient_accumulation=16` = **32 effectif** | Tient en VRAM avec bf16 + checkpointing |
| Epochs | **2** | Recommandation Unsloth 2026 pour LoRA classif (3 = overfit léger) |
| `max_length` | 512 tokens | Couvre la majorité des `titre + resume` |
| Augmentation | Back-translation EN↔FR | Variantes exclues du val (flag `augmentation_source`) |

### 4.3 Calibration post-fold

Pour chaque fold, après training :

1. **Temperature scaling** (`TemperatureScaler` dans `mlops/calibration.py`) : optimise T par L-BFGS sur NLL du val set → réduit la sur-confiance du modèle
2. **Threshold tuning** : recherche en grille MCC sur `threshold ∈ [0.05, 0.95]` → seuil de décision optimal pour ce fold

Les valeurs T et seuil sont **moyennées sur les K folds top-1** et persistées dans `temperature.json` (T=**1.395**) et `optimal_threshold.json` (seuil=**0.155**) à la racine de `models/production/`.

### 4.4 Tracking

- **MLflow** : expérience `greentech-classification`, run `qwen3-unified-k3-s2` (backend PostgreSQL + artifacts MinIO `s3://mlflow/`)
- **CodeCarbon** : empreinte mesurée lors de chaque training, persistée dans MLflow

---

## 5. Évaluation

### 5.1 Métriques K-fold honnêtes (estimation non-biaisée)

Mesurées sur les **folds de validation** (chaque article vu une seule fois en val), sans data leakage. **Métrique de référence pour la communication externe.**

| Métrique | Mean | Std (σ) | Min | Max |
|----------|------|---------|-----|-----|
| **MCC** | **0.6238** | 0.0103 | 0.6132 | 0.6361 |
| F1 | 0.6861 | 0.0078 | 0.6754 | 0.6948 |
| Recall (Green IT) | 0.8913 | 0.0257 | 0.8545 | 0.9223 |
| Precision (Green IT) | 0.5573 | 0.0145 | 0.5341 | 0.5749 |
| Balanced accuracy | 0.8617 | 0.0067 | 0.8536 | 0.8704 |
| Latence inférence (RX 7900 XTX, BF16) | ~800 ms | — | — | — |

**σ MCC = 0.010** → modèle reproductible et stable (σ < 0.05 seuil critique).

### 5.2 Métriques benchmark P5.1 (avec leakage K-fold, comparaison homogène)

Le `scripts/benchmark_models.py` évalue le modèle sur l'intégralité du dataset (11 664 originaux), incluant les articles vus en TRAIN par 2/3 folds. Les métriques sont donc **optimistes** mais permettent une comparaison équitable avec mDeBERTa K=5×3.

| Métrique | Valeur |
|----------|--------|
| **MCC** | **0.7565** |
| F1 | 0.7945 |
| Recall | 0.9557 |
| Precision | 0.6798 |
| Specificité | 0.8998 |
| Balanced accuracy | 0.9278 |
| Vrais positifs | 2 030 / 2 124 |
| Faux négatifs | 94 / 2 124 |
| Latence moyenne | 58.1 ms |
| Latence p50 / p95 / p99 | 59.5 / 83.5 / 87.5 ms |
| VRAM peak | 7.68 GB |

### 5.3 Validation end-to-end P5.5 (smoke test API + Celery, CPU Docker)

10 analyses sur des articles caractéristiques (mix Green IT / Non-Green / Borderline en EN et FR), via la chaîne complète FastAPI → Celery → worker :

- **10/10 tâches traitées** avec statut `termine`
- **9/10 prédictions correctes** (1 borderline assumé : parc éolien offshore sans angle IT)
- Latence moyenne : ~13 s/article en CPU dans le conteneur Docker
- Latence en GPU local (mode hybride RX 7900 XTX) : ~800 ms/article

### 5.4 Comparatifs internes (architectures testées)

| Modèle | MCC K-fold | F1 | Recall | Latence p50 | VRAM | Note |
|--------|-----------|-----|--------|-------------|------|------|
| `microsoft/deberta-v3-base` (févr. 2026) | — | 0.444 | 0.50 | — | — | Encoder EN-only, écarté |
| `Qwen/Qwen2.5-3B` + LoRA (févr. 2026) | — | 0.400 | 0.25 | — | — | Recall trop faible |
| `meta-llama/Llama-3.2-3B` + LoRA (mars 2026) | — | 0.667 | 0.50 | — | — | Gated Meta, abandonné |
| `Qwen/Qwen3-4B` + LoRA r=32 (avril 2026) | 0.762 (σ=0.25) | 0.762 | 0.74 | 34 ms | ~14 GB | Forte variance K-fold, dataset déséquilibré |
| `microsoft/mdeberta-v3-base` K=5×3 (mai 2026) | 0.5941 (σ=0.009) | 0.6600 | 0.89 | 66 ms | 2.8 GB | Challenger encoder benchmark B4 |
| **`Qwen/Qwen3-4B` + LoRA TIES K=3×2 (mai 2026)** | **0.6238 (σ=0.010)** | **0.6861** | **0.89** | **60 ms** | **7.7 GB** | **Champion actuel** |

Le verdict détaillé est dans `docs/SELECTION_CHAMPION_2026-04.md`.

---

## 6. Limites connues

1. **Précision modérée (0.56 K-fold honnête)** : ~44% des articles prédits Green IT sont des faux positifs. Trade-off assumé pour maximiser le recall (sensibilité). À surveiller en production via Grafana.
2. **Pas de test set hold-out indépendant** : toutes les métriques sont issues du K-fold ou du benchmark avec leakage. Pour la prochaine release, réserver 10% du dataset.
3. **TIES sur K=3 (au lieu de K=5)** : la moyenne de 3 adapters est moins robuste qu'une moyenne K=5. La stratégie hybride a été retenue pour rentrer dans la fenêtre temporelle ; un futur retrain pourra augmenter K si le temps de training Qwen3 est réduit (FlashAttention sur AMD ROCm 8.x, quantization, etc.).
4. **Coût VRAM 7.7 GB** : exclut les GPU < 16 GB (T4 OK, mais pas les tiers cloud les plus petits). Migration vers quantization 4-bit AWQ envisagée.
5. **Calibration globale** : pas de calibration par sous-population. Sur le sous-ensemble FR (25%), la calibration peut être sous-optimale. Une refonte agentic (P6 optionnel) pourrait y remédier.
6. **Pas de monitoring de drift en prod** : pas encore d'alerte si la distribution des prédictions dérive significativement de la baseline historique. À ajouter post-déploiement.

---

## 7. Biais et risques

### 7.1 Biais identifiés

- **Biais linguistique partiellement mitigé** : 25.25% du dataset est en français (vs ~5% dans la version d'avril). Stratification `MultilabelStratifiedKFold (langue × label)` garantit la représentativité dans chaque fold de val. Mais EN reste dominant (74.75%).
- **Biais éditorial** : sur-représentation de The Guardian (Environment) et arXiv (preprints scientifiques). Les sources généralistes (TechCrunch, Dev.to) et françaises (GreenIT.fr, Green Software Foundation) équilibrent partiellement.
- **Biais de définition** : la définition "Green IT" est inclusive (techniques d'optimisation énergétique, sustainable AI, écoconception, datacenters durables). Un article portant uniquement sur le climat (sans angle informatique) est classé Non Green IT.
- **Confiance asymétrique** : le modèle est plus confiant pour Non Green IT (proba > 0.99 régulièrement) que pour Green IT (proba souvent entre 0.5 et 0.9). Conséquence : le seuil de décision a été abaissé à 0.155 par calibration.

### 7.2 Risques d'utilisation

- **Pas de décision automatisée à haut enjeu** : ce modèle n'est pas destiné à des décisions impactant des personnes physiques (RGPD article 22). Il sert exclusivement à classer du contenu éditorial public.
- **Pas de modération de contenu** : aucune capacité à détecter du contenu inapproprié, illégal ou désinformant.
- **Pas un outil de notation greenwashing** : un article peut être classé Green IT sans que les solutions présentées soient nécessairement les plus vertueuses. La classification porte sur le **thème**, pas sur la **véracité de l'impact**.

### 7.3 Surveillance en production

Le pipeline expose des métriques Prometheus surveillées dans Grafana :

- `http_requests_total{handler="/analyze",method="POST"}` : volume de soumissions
- `http_request_duration_seconds_bucket{handler="/analyze/{job_id}"}` : latence polling p50/p95/p99
- Taux de classifications Green IT (drift potentiel si déséquilibre brutal)
- Disponibilité de l'API (alerte si down > 1 min)
- Logs centralisés Loki via Promtail (LogQL `{compose_service="celery-worker"} |= "ERROR"`)

---

## 8. Empreinte environnementale

| Indicateur | Valeur |
|------------|--------|
| Durée d'entraînement K=3×2 hybride | ~10h cumulées (mDeBERTa K=5×3 ~3h + Qwen3 K=3×2 ~6h) |
| TDP GPU effectif | ~250 W (bf16 + gradient checkpointing) |
| Région d'entraînement | France (PC fixe local) |
| Mix électrique | EDF (~85 g CO2eq/kWh en moyenne) |
| Estimation CO2 K=3×2 Qwen3 | ~127 g CO2eq (6 h × 250 W × 85 g/kWh) |
| CodeCarbon (run MLflow) | Voir `qwen3-unified-k3-s2` dans MLflow UI |
| Coût inférence GPU | ~0.2 W·h par classification (RX 7900 XTX, ~800 ms à 250 W actif) |
| Coût inférence CPU | ~3.6 W·h par classification (~13 s à 1000 W TDP CPU complet) |

> Les métriques détaillées par run sont disponibles dans MLflow (`mlflow ui --port 5000`) et dans le fichier `mlruns/...emissions.csv` de CodeCarbon.

---

## 9. Reproductibilité

### 9.1 Récupérer le modèle promu

```bash
# Re-télécharger le modèle promu depuis DVC + MinIO
uv run dvc pull models/production.dvc
```

### 9.2 Charger pour inférence (depuis Python)

```python
import asyncio
from pathlib import Path
from greentech.ai.models.inference import get_classifier

async def predict_example():
    clf = await get_classifier(model_path=Path("models/production"))
    result = await clf.predict(
        "Comment réduire la consommation énergétique des datacenters via du scheduling dynamique"
    )
    print(f"Green IT: {result.est_green_it}, score: {result.score_confiance:.3f}")

asyncio.run(predict_example())
```

### 9.3 Re-entraîner depuis zéro

```bash
# Pipeline complet (collecte + classification + training)
uv run python scripts/retrain_pipeline.py

# Re-training Qwen3 K=3×2 uniquement (~6 h sur RX 7900 XTX)
uv run python scripts/retrain_pipeline.py train-cv --model=qwen3

# Pipeline hybride P4.4 (mDeBERTa K=5×3 + Qwen3 K=3×2, ~10 h)
uv run python scripts/train_p4_hybrid.py
```

### 9.4 Compatibilité

- **OS** : Linux/Windows (testé sur Windows 11 Pro)
- **Python** : 3.12 via `uv`
- **Backend GPU local** : PyTorch + ROCm 7.2.1 (RX 7900 XTX) → ~800 ms/inférence
- **Backend Docker CPU** : PyTorch CPU → ~13 s/inférence (mode démo uniquement)
- **Backend Docker Linux + GPU AMD** : possible avec `--device=/dev/kfd` (non testé sur ce projet)

---

## 10. Évolutions prévues (roadmap)

| Phase | Description | Date cible |
|-------|-------------|------------|
| **P4.11** ✅ | Génération des résumés écologiques pour les 1 106 nouveaux Green IT | 17 mai 2026 — fait |
| **N1** ✅ | Intégration Celery + Redis pour file d'attente classifications | 17 mai 2026 — fait |
| **Future 1** | Hold-out test set 10% lors du prochain retrain (métriques honnêtes non-biaisées) | Prochain retrain |
| **Future 2** | Quantization 4-bit AWQ pour ramener VRAM à ~2-3 GB sans perte MCC > 0.01 | Q4 2026 |
| **Future 3** | Sous-modèles spécialisés EN/FR (refonte agentic LangGraph, P6 optionnel) | Q3-Q4 2026 |
| **Future 4** | Drift detection en production (alerte si distribution Green IT s'éloigne > 5% de baseline) | Post-déploiement |
| **Future 5** | Augmentation back-translation multilingue DE/ES/ZH (Qwen3 supporte ces langues nativement) | Q1 2027 |

---

## 11. Contacts et licences

- **Auteur** : KaRn1zC
- **Repository** : [GitHub.com/KaRn1zC/greentech-intelligence](https://github.com/KaRn1zC/greentech-intelligence)
- **Licence du code** : MIT (cf. `LICENSE` du dépôt)
- **Licence du modèle de base** : Apache-2.0 (`Qwen/Qwen3-4B`)
- **Licence des poids LoRA + merge TIES** : Apache-2.0 (héritée du modèle de base)

---

## 12. Documents connexes

- `docs/BENCHMARK_BRUT_2026-04.md` : baseline zero-shot avant entraînement (linear probing)
- `docs/BENCHMARK_FINAL_2026-04.md` : benchmark comparatif post-entraînement (P5.1)
- `docs/CHOIX_DEBERTA.md` : justification du challenger mDeBERTa-v3-base (P5.2)
- `docs/SELECTION_CHAMPION_2026-04.md` : verdict de promotion et critères composites (P5.3)
- `docs/PLAN_ETAPES.md` section 3.3 et section 7 : feuille de route d'entraînement
- `docs/CHECKLIST_SUIVI.md` BLOC E3 (C9-C13) : compétences validées
- `docs/SPECIFICATIONS_DATA.md` : inventaire des 10 sources de données
- `docs/PROCEDURE_MAJ_MODELE.md` : procédure opérationnelle de promotion d'un nouveau modèle
- `docs/PROCEDURE_MAJ_ROCM.md` : procédure de migration ROCm
- `models/production/promotion_info.json` : métadonnées exhaustives de la promotion (métriques K-fold, hyperparams, références)
- `models/p4_hybrid_summary.json` : bilan détaillé de l'entraînement P4.4 hybride
- `documentation interne` section "Classifieur fine-tune (Qwen3-4B + LoRA)" : référence opérationnelle

---

**Date de dernière mise à jour** : 2026-05-17
**Version du document** : 2.0 (refonte complète post-benchmark B4 + sélection champion)
