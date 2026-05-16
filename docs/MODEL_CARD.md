# Model Card - Classifieur Green IT (GreenTech Intelligence)

> **Auteur** : KaRn1zC
> **Date de création** : 2026-05-16
> **Version du modèle documenté** : `v20260415_204408`
> **Statut** : Version transitoire en production (sera remplacée après le benchmark B4 d'avril-mai 2026)

---

## 1. Identification du modèle

| Champ | Valeur |
|-------|--------|
| **Nom** | GreenTech Classifier — Qwen3-4B + LoRA |
| **Tâche** | Classification binaire de texte (Green IT vs Non Green IT) |
| **Domaine d'application** | Articles technologiques (presse, blogs techniques, publications scientifiques) |
| **Langues supportées** | Anglais, Français (multilingue natif Qwen3) |
| **Modèle de base** | `Qwen/Qwen3-4B` (Alibaba, licence Apache-2.0, publié le 26 juillet 2025) |
| **Adaptation** | LoRA (Low-Rank Adaptation) via la librairie PEFT 0.19.0 |
| **Type de tâche PEFT** | `SEQ_CLS` (classification de séquence à 2 classes) |
| **Version promue** | `v20260415_204408` (promotion le 15 avril 2026) |
| **Emplacement** | `models/production/` (tracé via DVC vers MinIO `s3://models/dvc`) |
| **Taille des poids** | ~46 Mo (adapter LoRA seul, base Qwen3-4B téléchargée séparément) |

### 1.1 Architecture LoRA

| Hyperparamètre | Valeur |
|----------------|--------|
| Rang LoRA `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `bias` | `none` |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention uniquement) |
| Modules sauvegardés | `classifier`, `score` (tête de classification) |

> **Note transitoire** : la version courante n'utilise PAS les `target_modules="all-linear"` du protocole B3 unifié (qui ajouterait `gate_proj`, `up_proj`, `down_proj` du MLP SwiGLU). La prochaine version intégrera cette extension après le benchmark B4.

---

## 2. Cas d'usage prévus

### 2.1 Usage principal

Classifier automatiquement des articles technologiques selon leur pertinence "Green IT" (informatique éco-responsable), pour alimenter le tableau de bord et les statistiques de l'application GreenTech Intelligence.

### 2.2 Pipeline d'inférence

Le modèle est appelé par l'API FastAPI dans la route `POST /analyze` après l'étape de génération du résumé de classification :

1. Extraction du contenu de l'article (URL, texte brut ou fichier uploadé)
2. Nettoyage et détection de langue
3. **Génération du résumé** via `classification_summarizer.py` (max 450 tokens, abstract dense)
4. **Classification** par le modèle documenté ici (`titre + résumé` en entrée)
5. Si Green IT confirmé : génération d'un résumé écologique séparé
6. Persistance des résultats en BDD

### 2.3 Public visé

Utilisateurs de la plateforme GreenTech Intelligence : étudiants, chercheurs, ingénieurs sensibilisés à l'écoconception numérique, journalistes spécialisés.

### 2.4 Usages hors périmètre

- **Pas de génération de texte** : le modèle est utilisé en mode classification (`AutoModelForSequenceClassification`), pas en mode causal LM. Les générations Qwen3 sont assurées par un autre modèle (Qwen3-4B-Instruct-2507 via API HF).
- **Pas de notation continue** : la sortie est binaire (Green IT ou non) avec un score de confiance entre 0 et 1, mais ne constitue pas un index quantitatif d'écoresponsabilité.
- **Pas d'évaluation d'impact carbone** : le modèle classifie le contenu textuel, il ne mesure pas l'empreinte carbone réelle des solutions mentionnées.

---

## 3. Données d'entraînement

### 3.1 Source

`data/golden_dataset.csv` (snapshot du 15 avril 2026, **avant l'enrichissement B2**)

| Caractéristique | Valeur |
|-----------------|--------|
| Volume total | 6 354 articles |
| Articles Green IT confirmés | ~22 (0.35 %) |
| Articles Non Green IT | ~6 332 (99.65 %) |
| Format de la feature | `titre + resume_classification` (résumé LLM Qwen3, max 450 tokens) |
| Label | Binaire (`est_green_it` ∈ {0, 1}) |
| Versioning | DVC vers MinIO `s3://models/dvc` |

### 3.2 Méthode d'annotation (deux étages)

1. **Étage 1 - Pré-filtre mots-clés** (`scripts/auto_annotate_dataset.py`) : marque chaque article `NON_GREEN` ou `CANDIDATE`
2. **Étage 2 - LLM judge Qwen** (`scripts/classify_candidates.py`) : tranche les candidats en zero-shot via Qwen3-4B-Instruct sur le contenu brut

### 3.3 Déséquilibre de classes

Le ratio Green IT / Non Green IT (~1 / 290) est **extrêmement déséquilibré**. La version actuelle a été entraînée avec un oversampling x84 de la classe minoritaire (22 Green IT dupliqués pour atteindre ~20 % du dataset).

> **Limite documentée** : cet oversampling tend à faire mémoriser les 22 textes positifs, ce qui se traduit par une variance K-fold élevée (σ MCC = 0.25, fold 1 à 0.51 et fold 4-5 à 1.00). Le protocole B3 unifié (à venir) remplace cet oversampling par une `class_weight=[1.0, 10.46]` sur la CrossEntropy + back-translation EN↔FR sur les positifs (cible σ < 0.10).

---

## 4. Procédure d'entraînement

### 4.1 Infrastructure

- **GPU** : AMD Radeon RX 7900 XTX 24 Go (architecture RDNA 3, `gfx1100`)
- **ROCm** : version 7.2.1 stable (wheels-only depuis le 18 avril 2026)
- **PyTorch** : 2.9.1+rocm7.2.1
- **Précision** : `bf16=True`
- **Activation `gradient_checkpointing`** : `True` (use_reentrant=False)

### 4.2 Pipeline

Lancé via `uv run python scripts/retrain_pipeline.py train-cv --model=qwen3`. Validation croisée K-fold (K=5) avec `StratifiedKFold` sur le label uniquement (pas encore stratifié par langue dans cette version).

### 4.3 Tracking

- **MLflow** : run loggué dans l'expérience `greentech-classification` (backend PostgreSQL + artifacts MinIO)
- **CodeCarbon** : empreinte mesurée lors de l'entraînement (à compléter dans la prochaine version)

---

## 5. Évaluation

### 5.1 Métriques sur le test set agrégé (n = 6 354)

| Métrique | Valeur |
|----------|--------|
| MCC (Matthews Correlation Coefficient) | **0.7620** |
| F1-score | 0.7619 |
| Accuracy | 0.9984 |
| Balanced accuracy | 0.8633 |
| Precision | 0.8000 |
| Recall | 0.7273 |
| Spécificité | 0.9994 |
| Latence moyenne (RX 7900 XTX) | **34 ms** |

### 5.2 Matrice de confusion (test set)

| | Prédit Green IT | Prédit Non Green IT |
|---|------------------|----------------------|
| **Réel Green IT** | 16 (VP) | 6 (FN) |
| **Réel Non Green IT** | 4 (FP) | 6 328 (VN) |

### 5.3 Détail des 5 folds K-fold

| Fold | MCC | F1 | Precision | Recall | Note |
|------|-----|-----|-----------|--------|------|
| 1 | 0.515 | 0.500 | 0.667 | 0.400 | Plus faible (mémorisation insuffisante) |
| 2 | 0.799 | 0.800 | 0.800 | 0.800 | Bon équilibre |
| 3 | 0.498 | 0.500 | 0.500 | 0.500 | Faible (variance) |
| 4 | 1.000 | 1.000 | 1.000 | 1.000 | Mémorisation parfaite |
| 5 | 1.000 | 1.000 | 1.000 | 1.000 | Mémorisation parfaite |
| **Moyenne** | **0.7625** | **0.760** | **0.793** | **0.740** | σ MCC = 0.248 |

> **Lecture critique** : la variance inter-fold est très forte (folds 4 et 5 à 1.00, fold 3 à 0.50). C'est le signe d'un dataset trop petit et trop déséquilibré pour une évaluation K-fold robuste. Le protocole B3 unifié, l'enrichissement B2 (1 018 Green IT au lieu de 22) et la stratification multilingue doivent corriger cela.

### 5.4 Comparatifs internes (modèles précédents)

| Modèle | F1 (best fold) | MCC | Note |
|--------|----------------|-----|------|
| `microsoft/deberta-v3-base` (févr. 2026) | 0.44 | — | Encoder EN-only, fp32 forcé |
| `Qwen/Qwen2.5-3B` + LoRA (févr. 2026) | 0.40 | — | Recall=0.25 (mémorisation insuffisante) |
| `meta-llama/Llama-3.2-3B` + LoRA (mars 2026) | 0.667 | — | Modèle gated, abandonné pour le multilinguisme |
| **`Qwen/Qwen3-4B` + LoRA (avril 2026)** | **0.762** | **0.762** | **Modèle actuellement promu** |

---

## 6. Limites connues

1. **Forte variance K-fold** : σ MCC = 0.25 sur 5 folds, dû à un dataset trop déséquilibré (~22 positifs sur 6 354).
2. **Dataset historique** : ce modèle a été entraîné AVANT l'enrichissement B2 (passage à 1 018 Green IT). Il n'a donc jamais vu les articles de GreenIT.fr, Green Software Foundation, Sustainable Web Design, Climate Action Tech, ni l'extension Crossref/arXiv API.
3. **LoRA réduit (`q,k,v,o` seulement)** : le protocole B3 cible désormais `all-linear` (attention + MLP), ce qui devrait améliorer la capacité d'adaptation sans surcoût notable.
4. **Pas de calibration** : la sortie de probabilité n'est pas calibrée par température scaling. Le seuil de décision est fixé arbitrairement à 0.5. Le protocole B3 ajoute `temperature.json` + `optimal_threshold.json` post-fold.
5. **Stratification simple** : `StratifiedKFold` sur le label uniquement. La nouvelle version utilisera `MultilabelStratifiedKFold` sur `(langue × label)` pour garantir 75/25 EN/FR sur chaque fold.

---

## 7. Biais et risques

### 7.1 Biais identifiés

- **Biais linguistique** : la version actuelle a été entraînée majoritairement sur de l'anglais. Les articles français risquent d'être sous-classés. Mitigé par l'enrichissement B2 (25 % du nouveau dataset est en français, principalement GreenIT.fr).
- **Biais éditorial** : sur-représentation des sources Guardian Environment et arXiv Kaggle (4 957 articles) dans le dataset historique. Le nouveau dataset rééquilibre mieux.
- **Biais de définition** : la définition "Green IT" est inclusive (techniques d'optimisation énergétique, sustainable AI, écoconception, datacenters durables). Un article portant uniquement sur le climat (sans angle informatique) sera classé Non Green IT.

### 7.2 Risques d'utilisation

- **Pas de décision automatisée à haut enjeu** : ce modèle n'est pas destiné à des décisions impactant des personnes physiques (RGPD article 22). Il sert exclusivement à classer du contenu éditorial public.
- **Pas de modération de contenu** : aucune capacité à détecter du contenu inapproprié, illégal ou désinformant.

### 7.3 Surveillance

Le pipeline expose des métriques Prometheus surveillées dans Grafana :
- Temps d'inférence (alerte si > 10 s)
- Taux de classifications Green IT (drift potentiel si déséquilibre brutal)
- Disponibilité de l'API (alerte si down > 1 min)

---

## 8. Empreinte environnementale

| Indicateur | Valeur |
|------------|--------|
| Mesure CodeCarbon (entraînement complet K-fold) | À compléter (artefact MLflow `v20260415_204408`) |
| Région d'entraînement | France (PC fixe local) |
| Mix électrique | EDF (~85 g CO2eq/kWh hors heures de pointe) |
| GPU | AMD RX 7900 XTX (TDP ~355 W, mais bf16 + grad_checkpointing limitent à ~250 W effectifs) |
| Durée d'entraînement K-fold | ~2-3 h sur 5 folds |

> Les métriques détaillées par run sont disponibles dans MLflow (`mlflow ui --port 5000`) et dans le fichier `mlruns/...emissions.csv` de CodeCarbon.

---

## 9. Reproductibilité

```bash
# Re-télécharger le modèle promu depuis DVC + MinIO
uv run dvc pull models/production.dvc

# Charger pour inférence (depuis Python)
from greentech.ai.models.inference import get_classifier
classifier = get_classifier()
result = classifier.predict(title="Optimiser le cloud", summary="...")
# -> {"is_green_it": True, "confidence": 0.87, "latency_ms": 34}
```

### 9.1 Compatibilité

- **OS** : Linux/Windows (testé sur Windows 11 Pro)
- **Python** : 3.12 via `uv`
- **Backend** : PyTorch + ROCm 7.2.1 (PC fixe AMD) ou `torch_directml.device()` (PC portable AMD)
- **CPU fallback** : possible mais lent (~3 s/inférence vs 34 ms sur GPU)

---

## 10. Évolutions prévues (roadmap)

| Phase | Description | Date cible |
|-------|-------------|------------|
| **B3** | Protocole d'entraînement unifié (stratification langue×label, class_weight, back-translation, calibration, ensemble K=5×3 seeds) | Avril-mai 2026 |
| **B4.2** | Benchmark BRUT (zero-shot) Qwen3-4B vs `microsoft/mdeberta-v3-base` | Mai 2026 |
| **B4.3** | Entraînement K-fold des 2 modèles avec le protocole B3 (~6-8 h) | Mai 2026 |
| **B4.4** | Benchmark comparatif post-entraînement | Mai 2026 |
| **B4.5** | Sélection du champion et promotion en production | Mai 2026 |
| **B5** | (Bonus optionnel) Refonte agentic LangGraph | Mai 2026 |

Cette Model Card sera **réécrite** après la sélection finale B4.5 pour refléter le nouveau modèle promu (Qwen3 ou mDeBERTa selon le benchmark).

---

## 11. Contacts et licences

- **Auteur** : KaRn1zC
- **Repository** : [GitHub.com/KaRn1zC/greentech-intelligence](https://github.com/KaRn1zC/greentech-intelligence)
- **Licence du code** : MIT (cf. `LICENSE` du dépôt)
- **Licence du modèle de base** : Apache-2.0 (`Qwen/Qwen3-4B`)
- **Licence des poids LoRA** : Apache-2.0 (héritée du modèle de base)

---

## 12. Documents connexes

- `docs/PLAN_ETAPES.md` section 3.3 et section 7 : feuille de route d'entraînement
- `docs/CHECKLIST_SUIVI.md` BLOC E3 (C9-C13) : compétences validées
- `docs/SPECIFICATIONS_DATA.md` : inventaire des 10 sources de données
- `docs/PROCEDURE_MAJ_MODELE.md` : procédure opérationnelle de promotion d'un nouveau modèle
- `docs/PROCEDURE_MAJ_ROCM.md` : procédure de migration ROCm
- `documentation interne` section "Classifieur fine-tune (Qwen3-4B + LoRA)" : référence opérationnelle

---

**Date de dernière mise à jour** : 2026-05-16
**Version du document** : 1.0
