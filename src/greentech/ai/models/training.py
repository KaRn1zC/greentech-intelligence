"""Scripts d'entraînement pour les modèles Champion et Challengers.

Champion     : DeBERTa-v3-base (fine-tuning classique via Transformers Trainer)
Challenger 1 : Qwen2.5-3B (fine-tuning LoRA via PEFT, 3085M params, legacy)
Challenger 2 : Llama 3.2 3B (fine-tuning LoRA via PEFT, 3213M params, gated, legacy)
Challenger 3 : Qwen3-4B (fine-tuning LoRA via PEFT, ~4000M params, Apache-2.0)

Le challenger Qwen3-4B est le modele d'entrainement cible depuis le
15 avril 2026 : il remplace Llama 3.2 3B comme base du pipeline de
production (`scripts/retrain_pipeline.py`). Avantages :
  - Architecture dense transformer standard, pleinement supportee par
    `transformers` et compatible ROCm sans dependance exotique.
  - Multilingue natif (FR/EN/DE/ES/ZH) pour traiter les articles scrapes
    sans etape de traduction.
  - Licence Apache-2.0 (pas de gated access).
  - Meme famille que `Qwen3-4B-Instruct-2507` deja utilise pour les
    summarizers et le LLM judge : chat template et tokenizer unifies.

La tentative precedente avec `Qwen/Qwen3.5-4B` a ete abandonnee parce qu'il
s'agit en realite d'un modele multimodal vision-langage (image-text-to-text)
avec une architecture a attention lineaire hybride necessitant
`flash-linear-attention` + `causal-conv1d` - aucun support ROCm fiable.
L'utilisation d'`AutoModelForSequenceClassification` chargeait les blocs
visuels comme poids "UNEXPECTED" et saturait la VRAM au premier step.

Tous les modeles sont entraines sur le Golden Dataset annote et compares
via MLflow (accuracy, F1, MCC, latence, emissions CO2).

Gere le desequilibre extreme du dataset (22 Green IT / 5786 Non Green IT)
via oversampling de la minorite a ~20%.

Usage:
    uv run python -m greentech.ai.models.training                   # Tous les modeles
    uv run python -m greentech.ai.models.training deberta  # Champion seul
    uv run python -m greentech.ai.models.training qwen2.5   # Qwen2.5-3B (legacy)
    uv run python -m greentech.ai.models.training llama3.2  # Llama 3.2 3B (legacy)
    uv run python -m greentech.ai.models.training qwen3  # Qwen3-4B (recommande)
    uv run python -m greentech.ai.models.training benchmark         # Benchmark comparatif

"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from greentech.ai.mlops.tracking import ExperimentConfig, log_model_artifact, tracked_experiment
from greentech.ai.models.classifier import (
    BaseClassifier,
    LabelGreenIT,
    LoraConfig,
    PredictionResult,
    TrainingConfig,
)
from greentech.config import BASE_DIR, get_settings

# Labels pour la classification binaire
LABEL2ID = {"Non Green IT": 0, "Green IT": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Identifiants valides pour le CLI
VALID_MODELS = (
    "deberta",
    "mdeberta",
    "qwen2.5",
    "llama3.2",
    "qwen3",
)

# Modules cibles LoRA pour Qwen3-4B : `all-linear` inclut les 4 projections
# d'attention (q/k/v/o) PLUS les 3 projections MLP SwiGLU (gate/up/down).
# Gain attendu +1-2 pts MCC documente (QLoRA paper Dettmers et al., reproduit
# par Unsloth et Databricks). Budget VRAM sur RX 7900 XTX 24 Go : base ~8 Go
# bf16 + adapters r=32 all-linear ~400 Mo + activations batch 2 x seq 512 ~7 Go
# + AdamW adapters ~1.5 Go ≈ 17-20 Go, confortable.
# L'ancienne liste attention-only est conservee pour reference et legacy.
QWEN3_LORA_TARGET_MODULES_ATTENTION_ONLY: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]
QWEN3_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Calcule les metriques de classification pour le Trainer.

    Inclut le MCC (Matthews Correlation Coefficient) qui est la metrique
    principale retenue pour le benchmark B4. MCC est robuste au desequilibre
    de classe (ratio 1:10.5 ici) la ou F1 peut masquer les faiblesses.

    Args:
        eval_pred: Tuple (logits, labels) fourni par le Trainer.

    Returns:
        Dictionnaire de metriques (accuracy, f1, precision, recall,
        matthews_correlation).
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
    }


class WeightedLossTrainer(Trainer):
    """Trainer avec CrossEntropy ponderee pour gerer le desequilibre de classe.

    Remplace l'oversampling x84 historique par une loss ponderee
    ``class_weight=[1.0, N_neg/N_pos]`` (typiquement ~[1.0, 10.5]). Les 3
    agents de recherche convergent : pour un ratio modere (1:10.5), la
    weighted CE est la SOTA, Focal Loss n'apporte de gain qu'au-dela de 1:50
    et degrade la calibration.

    La class_weight est passee via l'attribut ``class_weight`` (torch.Tensor)
    sur l'instance. Si ``None``, fallback sur CrossEntropy standard non-ponderee.
    """

    def __init__(
        self,
        *args,
        class_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Override : CrossEntropy avec class_weight si disponible.

        Signature compatible transformers >= 5.0 (ajout de
        ``num_items_in_batch`` en v5.2 pour gradient accumulation).
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weight.to(logits.device) if self.class_weight is not None else None

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        # Restaurer labels pour que le Trainer puisse logger les predictions
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss


class SWACallback(TrainerCallback):
    """Stochastic Weight Averaging via collecte de snapshots de fin d'entrainement.

    Reference : Izmailov et al. 2018 (arXiv:1803.05407), validee 2024-2025
    sur transformers (MDPI 10.3390/app13052935). SWA moyenne les snapshots
    des poids des derniers epochs pour produire un modele plus generalisant
    qu'un point unique d'entrainement, sans cout d'inference supplementaire
    (0 ralentissement) et avec un gain typique +0.5-1 MCC sur des problemes
    imbalanced.

    Strategie : collecte un snapshot du ``state_dict()`` du modele a la fin
    de chaque epoch dans la fenetre ``[swa_start_ratio * total_epochs ; fin]``,
    stocke les snapshots sur CPU (eviter explosion VRAM), puis en
    ``on_train_end`` calcule la moyenne arithmetique des N snapshots et la
    re-uploade dans le modele.

    Compatible avec :
    - Modeles full fine-tune (mDeBERTa) : snapshot du modele complet
    - Modeles PEFT/LoRA (Qwen3) : snapshot inclut les adapters seulement
      (les poids backbone sont frozen, donc identiques entre snapshots)
    - ``gradient_checkpointing=True`` : OK, le state_dict() est independant

    Attributes:
        swa_start_ratio: Fraction des epochs au-dela de laquelle commencer
            la collecte (defaut 0.75 = derniere quart d'entrainement).
            Pour 3 epochs Qwen3 : snapshot a la fin de l'epoch 3 seulement.
            Pour 5 epochs mDeBERTa : snapshots aux epochs 4 et 5.
        snapshots: Liste interne des state_dicts collectes (CPU, float32).
    """

    def __init__(self, swa_start_ratio: float = 0.75) -> None:
        if not 0.0 < swa_start_ratio < 1.0:
            msg = f"swa_start_ratio doit etre dans ]0, 1[, recu {swa_start_ratio}"
            raise ValueError(msg)
        self.swa_start_ratio = swa_start_ratio
        self.snapshots: list[dict[str, torch.Tensor]] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs) -> None:
        """Collecte un snapshot CPU des parametres trainables si on est dans la fenetre SWA."""
        if model is None or state.epoch is None:
            return
        total_epochs = args.num_train_epochs
        epoch_ratio = state.epoch / total_epochs
        if epoch_ratio < self.swa_start_ratio:
            return

        # Snapshot CPU float32 des SEULS parametres trainables.
        # Pour Qwen3 + LoRA : seuls les adapters LoRA + tete de classification
        # sont trainables (~50M params, ~200 MB en fp32 par snapshot).
        # Pour mDeBERTa full fine-tune : tous les 278M params (~1.1 GB par snapshot).
        # Sans ce filtre, on snapshoterait les 4B params du backbone Qwen3
        # (8 GB par snapshot, explose la RAM CPU).
        snapshot = {
            name: param.detach().cpu().to(torch.float32).clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.snapshots.append(snapshot)
        logger.info(
            f"  [SWA] Snapshot {len(self.snapshots)} collecte "
            f"(epoch {state.epoch:.1f}/{total_epochs}, "
            f"{len(snapshot)} tenseurs trainables, "
            f"~{sum(t.numel() for t in snapshot.values()) / 1e6:.1f}M params)"
        )

    def on_train_end(self, args, state, control, model=None, **kwargs) -> None:
        """Applique la moyenne des snapshots collectes aux parametres trainables."""
        if model is None or len(self.snapshots) < 2:
            if len(self.snapshots) < 2:
                logger.info(
                    f"  [SWA] {len(self.snapshots)} snapshot(s) collecte(s) "
                    "(< 2 requis pour moyenner), SWA desactive pour ce fold."
                )
            return

        n = len(self.snapshots)
        # Moyenne arithmetique des N snapshots, tenseur par tenseur
        avg_state: dict[str, torch.Tensor] = {}
        for key in self.snapshots[0]:
            avg_state[key] = sum(s[key] for s in self.snapshots) / n

        # Re-uploader la moyenne dans les parametres correspondants du modele
        applied = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in avg_state:
                    param.copy_(avg_state[name].to(param.device, dtype=param.dtype))
                    applied += 1

        logger.info(
            f"  [SWA] Final : moyenne de {n} snapshots appliquee au modele "
            f"({applied}/{len(avg_state)} parametres synchronises)"
        )

        # Liberer la memoire CPU des snapshots
        self.snapshots.clear()


def compute_class_weight(labels: list[int] | np.ndarray) -> torch.Tensor:
    """Calcule class_weight pour CrossEntropy pondere : [1.0, N_neg/N_pos].

    La classe negative (0) a toujours poids 1.0 ; la classe positive (1) a
    poids egal au ratio de desequilibre, pour equilibrer la contribution
    des deux classes dans le gradient. Le ratio est calcule sur le train
    set de chaque fold, donc il varie legerement (~10.4-10.6) selon le split.

    Args:
        labels: Labels binaires (0/1) du train set courant.

    Returns:
        Tensor float32 de shape ``(2,)`` : [poids_neg, poids_pos].
    """
    labels_arr = np.asarray(labels)
    n_pos = int(labels_arr.sum())
    n_neg = int(len(labels_arr) - n_pos)
    if n_pos == 0:
        logger.warning("Aucun positif dans le train set, class_weight=[1.0, 1.0]")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    pos_weight = n_neg / n_pos
    logger.info(
        f"class_weight : [1.0, {pos_weight:.2f}] "
        f"(N_neg={n_neg}, N_pos={n_pos}, ratio 1:{pos_weight:.1f})"
    )
    return torch.tensor([1.0, pos_weight], dtype=torch.float32)


class DeBERTaClassifier(BaseClassifier):
    """Classifieur Champion basé sur DeBERTa-v3-base.

    Fine-tuning classique via Hugging Face Trainer avec évaluation
    à chaque époque. Optimisé pour la précision de classification.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
    ) -> None:
        if config is None:
            config = TrainingConfig(
                nom_modele="microsoft/deberta-v3-base",
                # DeBERTa-v3-base (EN-only) est conserve en legacy apres la
                # bascule vers mDeBERTa-v3-base en avril 2026 (dataset bilingue
                # EN 74.75 % / FR 25.25 %). Aucun nouveau run n'est attendu sur
                # cette classe, d'ou le suffixe "-legacy" qui evite toute
                # confusion avec un hypothetique re-entrainement futur.
                output_dir=BASE_DIR / "models" / "deberta-legacy",
                epochs=5,
                batch_size=16,
                learning_rate=3e-5,
            )
        super().__init__(config)
        # Class weight optionnel pour CrossEntropy ponderee, set par la
        # boucle K-fold avant chaque train (fallback None = CE standard).
        self.class_weight: torch.Tensor | None = None

    async def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> dict[str, float]:
        """Entraîne DeBERTa-v3-base sur le dataset annoté.

        Args:
            train_texts: Textes d'entraînement.
            train_labels: Labels (0/1).
            val_texts: Textes de validation.
            val_labels: Labels de validation.

        Returns:
            Métriques finales (accuracy, f1, precision, recall).
        """
        device = self.detect_device()
        logger.info(
            f"Entraînement Champion sur {device} : {len(train_texts)} train, {len(val_texts)} val"
        )

        # Precision : bf16 est OK sur DeBERTa-v3 depuis transformers >= 4.48
        # (bug #35332 DisentangledSelfAttention corrige en decembre 2024,
        # PR #35336). fp16 strictement interdit (NaN garanti sur DeBERTa).
        # Fallback fp32 pour les versions anciennes par securite.
        try:
            import transformers as _tf

            _tf_version = tuple(int(p) for p in _tf.__version__.split(".")[:2] if p.isdigit())
        except (ImportError, AttributeError, ValueError):
            _tf_version = (0, 0)
        use_bf16 = _tf_version >= (4, 48) and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        logger.info(
            f"Precision Champion : {'bf16' if use_bf16 else 'fp32'} "
            f"(transformers {'.'.join(map(str, _tf_version))})"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nom_modele)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            dtype=dtype,
        )

        # Tokenizer les datasets
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)

        # Calculer warmup_steps a partir du ratio (warmup_ratio deprecie en v5.2)
        import math

        steps_per_epoch = math.ceil(len(train_dataset) / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Configuration du Trainer
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            # Linear decay standard pour DeBERTa (convention paper DeBERTaV3).
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            # MCC comme metrique de selection (robuste au desequilibre).
            metric_for_best_model="matthews_correlation",
            greater_is_better=True,
            seed=self.config.seed,
            bf16=use_bf16,
            fp16=False,  # strictement interdit sur DeBERTa (NaN garanti)
            logging_steps=10,
            report_to="none",
        )

        # WeightedLossTrainer : CrossEntropy ponderee si self.class_weight
        # est defini par la boucle K-fold, sinon fallback CE standard.
        # + SWACallback : moyenne les snapshots des derniers 25 % d'epochs
        # (Izmailov et al. 2018), gain attendu +0.5-1 MCC, cout d'inference nul.
        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weight=self.class_weight,
            callbacks=[SWACallback(swa_start_ratio=0.75)],
        )

        # Entraînement
        logger.info(f"Demarrage de l'entrainement Champion ({self.config.nom_modele})...")
        train_result = trainer.train()

        # Évaluation finale
        eval_result = trainer.evaluate()
        logger.info(f"Champion — Résultats : {eval_result}")

        return {
            "train_loss": train_result.training_loss,
            **{k.replace("eval_", ""): v for k, v in eval_result.items()},
        }

    def _prepare_dataset(self, texts: list[str], labels: list[int]) -> Dataset:
        """Tokenize les textes et crée un Dataset Hugging Face.

        Args:
            texts: Liste de textes bruts.
            labels: Liste de labels correspondants.

        Returns:
            Dataset tokenizé prêt pour le Trainer.
        """
        dataset = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    async def predict(self, text: str) -> PredictionResult:
        """Classifie un texte avec DeBERTa.

        Args:
            text: Texte de l'article à classifier.

        Returns:
            Résultat avec label et score de confiance.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Modèle non chargé. Appelez load() ou train() d'abord."
            raise RuntimeError(msg)

        start = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs[0][predicted_class].item()
        proba_positive = float(probs[0][LabelGreenIT.GREEN.value].item())
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return PredictionResult(
            label=LabelGreenIT(predicted_class),
            score_confiance=confidence,
            temps_ms=elapsed_ms,
            modele=self.config.nom_modele,
            proba_positive=proba_positive,
        )

    def save(self, output_dir: Path | None = None) -> Path:
        """Sauvegarde le modèle Champion.

        Args:
            output_dir: Dossier de destination.

        Returns:
            Chemin du modèle sauvegardé.
        """
        save_path = output_dir or self.config.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info(f"Champion sauvegardé : {save_path}")
        return save_path

    def load(self, model_path: Path) -> None:
        """Charge un modèle Champion sauvegardé.

        Args:
            model_path: Chemin vers le dossier du modèle.
        """
        device = self.detect_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        logger.info(f"Champion chargé depuis {model_path} sur {device}")


class MDeBERTaClassifier(DeBERTaClassifier):
    """Classifieur Champion base sur `microsoft/mdeberta-v3-base`.

    Specialisation de ``DeBERTaClassifier`` pour mDeBERTa-v3-base
    (encoder multilingue 278M params, pre-entraine sur 100 langues via
    CC100). Retenu apres analyse du dataset bilingue EN 74.75 % / FR 25.25 %
    (avril 2026) : `deberta-v3-base` EN-pur encoderait mal les 600 Green IT
    francais et fausserait le benchmark contre Qwen3-4B en faveur de ce
    dernier. mDeBERTa conserve l'architecture DeBERTa-v3
    (DisentangledSelfAttention, RTD pre-training) et reste la meilleure
    baseline encoder multilingue 2024-2026 pour classification binaire
    FR+EN sur texte technique/scientifique (+3.6 pts XNLI vs XLM-R base).

    Hyperparametres issus de l'agent de recherche C (2026-04-21) :

    - `lr=2e-5`, scheduler linear, `warmup_ratio=0.06`
    - `batch=16, grad_accum=2` (batch effectif 32)
    - `epochs=5` avec early stopping sur val MCC (patience 2)
    - `max_length=384` (couvre 98 % des resumes FR+titre, tokenizer
      SentencePiece FR genere ~1.6 tokens/mot)
    - `weight_decay=0.01`, dropout 0.1 (default)
    - Precision : `bf16` si `transformers >= 4.48`, sinon `fp32` (fp16
      strictement interdit : NaN garanti sur DeBERTa)
    - `attn_implementation="sdpa"` (Flash-Attention indisponible RDNA3)
    - `gradient_checkpointing=True` (libere ~30 % VRAM)
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
    ) -> None:
        if config is None:
            config = TrainingConfig(
                nom_modele="microsoft/mdeberta-v3-base",
                output_dir=BASE_DIR / "models" / "mdeberta",
                epochs=5,
                batch_size=16,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_ratio=0.06,
                max_length=384,
            )
        super().__init__(config)


class LoRAClassifier(BaseClassifier):
    """Classifieur Challenger generique avec LoRA.

    Fine-tuning efficient via PEFT/LoRA pour adapter un modèle génératif
    à la classification binaire. Supporte Qwen2.5-3B et Llama 3.2 3B.
    Nécessite le GPU AMD 7900 XTX via ROCm.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        lora_config: LoraConfig | None = None,
    ) -> None:
        if config is None:
            config = TrainingConfig(
                nom_modele="Qwen/Qwen2.5-3B",
                output_dir=BASE_DIR / "models" / "qwen2.5",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            )
        super().__init__(config)
        self.lora_config = lora_config or LoraConfig()
        # Class weight optionnel pour CrossEntropy ponderee, set par la
        # boucle K-fold avant chaque train (fallback None = CE standard).
        self.class_weight: torch.Tensor | None = None

    async def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> dict[str, float]:
        """Entraine le modele de base configure avec LoRA sur le dataset annote.

        La base effectivement utilisee depend de `self.config.nom_modele` (Qwen3-4B
        par defaut via `Qwen3Classifier`, Qwen2.5-3B ou Llama 3.2 3B pour
        les sous-classes legacy).

        Args:
            train_texts: Textes d'entrainement.
            train_labels: Labels (0/1).
            val_texts: Textes de validation.
            val_labels: Labels de validation.

        Returns:
            Metriques finales (accuracy, f1, precision, recall).
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        device = self.detect_device()
        logger.info(f"Entraînement Challenger (LoRA) sur {device} : {len(train_texts)} train")

        # Token HF (optionnel, utile pour les modeles gated)
        hf_token = get_settings().huggingface_token or None

        # Charger tokenizer et modèle
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nom_modele, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            dtype=torch.bfloat16,
            token=hf_token,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Piege PEFT + gradient_checkpointing : sans cet appel, les adapters
        # LoRA ne recoivent pas les gradients quand `gradient_checkpointing=True`
        # (issue HF transformers #42947). Doit etre appele AVANT get_peft_model.
        self.model.enable_input_require_grads()

        # Appliquer LoRA. ``use_rslora`` active rank-stabilized LoRA
        # (Kalajdzievski 2023) : remplace le facteur d'echelle α/r par α/√r
        # qui preserve la stabilite des gradients a rang eleve (r=32).
        # Cout zero a l'inference, gain mesure +0.5-1 MCC sur Qwen3-4B.
        peft_config = PeftLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules or ["q_proj", "v_proj"],
            use_rslora=self.lora_config.use_rslora,
        )
        self.model = get_peft_model(self.model, peft_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"LoRA appliqué : {trainable_params:,} / {total_params:,} paramètres entraînables "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        # Préparer les datasets
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)

        # Calculer warmup_steps a partir du ratio (warmup_ratio deprecie en v5.2)
        import math

        effective_batch = self.config.batch_size * 4  # gradient_accumulation_steps
        steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
        total_steps = steps_per_epoch * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Scheduler cosine (vs linear historique) : recommande par Unsloth
        # et Databricks pour 3 epochs sur LoRA, evite les oscillations en
        # fin d'entrainement sur des metriques imbalanced-sensitive comme MCC.
        #
        # Choix critiques pour Qwen3-4B + LoRA :
        # - ``eval_strategy="no"`` : on n'evalue PAS a chaque epoch pendant
        #   le training. Sur Qwen3-4B, l'eval avec gradient_checkpointing
        #   actif prend ~10s/batch (vs 0.2s en training), ce qui exploserait
        #   le temps total (~1h45 par eval x 3 epochs = inacceptable).
        #   L'eval finale est faite manuellement APRES avoir desactive
        #   gradient_checkpointing et reactive use_cache (cf. ci-dessous).
        # - ``save_strategy="no"`` : pas de checkpoint a chaque epoch.
        #   SWA + calibration finale remplacent ``load_best_model_at_end``.
        # - ``per_device_eval_batch_size=8`` : 4x le batch train pour
        #   accelerer l'eval finale (pas de backward = on peut tasser).
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=8,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=self.config.seed,
            bf16=True,
            gradient_accumulation_steps=16,
            # Gradient checkpointing : echange le cout memoire des activations
            # contre un leger surcout en temps de calcul (~20%). Indispensable
            # pour tenir un LoRA 4B en BF16 sur 24 Go de VRAM avec `use_reentrant=False`
            # (recommande par transformers >=5).
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            report_to="none",
        )

        # Le gradient checkpointing exige que le cache KV soit desactive :
        # les activations ne sont pas stockees pour le backward et elles
        # seraient recalculees depuis le cache, ce qui casse l'entrainement.
        self.model.config.use_cache = False

        # WeightedLossTrainer : CrossEntropy ponderee si self.class_weight est
        # defini par la boucle K-fold, sinon fallback CE standard. Remplace
        # l'oversampling historique (qui dupliquait les 22 memes textes 84x,
        # cause d'overfit sur les chaines exactes).
        # + SWACallback : moyenne les snapshots des derniers 25 % d'epochs
        # (Izmailov et al. 2018), filtre sur requires_grad pour ne moyenner
        # que les adapters LoRA + tete classification (~50M params CPU).
        trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            class_weight=self.class_weight,
            callbacks=[SWACallback(swa_start_ratio=0.75)],
        )

        logger.info(f"Demarrage de l'entrainement Challenger ({self.config.nom_modele} + LoRA)...")
        train_result = trainer.train()

        # Eval finale : reactiver KV cache et desactiver gradient_checkpointing
        # pour eviter le slowdown 30-50x observe sur Qwen3-4B (10s/batch ->
        # < 0.5s/batch). Le SWA a deja moyenne les poids finaux, c'est cet
        # etat qu'on evalue.
        self.model.config.use_cache = True
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

        logger.info("Eval finale (use_cache=True, gradient_checkpointing=False)...")
        eval_result = trainer.evaluate()
        logger.info(f"Challenger — Résultats : {eval_result}")

        return {
            "train_loss": train_result.training_loss,
            **{k.replace("eval_", ""): v for k, v in eval_result.items()},
        }

    def _prepare_dataset(self, texts: list[str], labels: list[int]) -> Dataset:
        """Tokenize les textes pour le challenger (Qwen ou Llama).

        Args:
            texts: Liste de textes bruts.
            labels: Liste de labels correspondants.

        Returns:
            Dataset tokenise.
        """
        dataset = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    async def predict(self, text: str) -> PredictionResult:
        """Classifie un texte avec la base du challenger + LoRA.

        Args:
            text: Texte de l'article a classifier.

        Returns:
            Resultat avec label et score de confiance.
        """
        if self.model is None or self.tokenizer is None:
            msg = "Modèle non chargé. Appelez load() ou train() d'abord."
            raise RuntimeError(msg)

        start = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs[0][predicted_class].item()
        proba_positive = float(probs[0][LabelGreenIT.GREEN.value].item())
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return PredictionResult(
            label=LabelGreenIT(predicted_class),
            score_confiance=confidence,
            temps_ms=elapsed_ms,
            modele=f"{self.config.nom_modele}+LoRA",
            proba_positive=proba_positive,
        )

    def save(self, output_dir: Path | None = None) -> Path:
        """Sauvegarde le modèle Challenger (adapters LoRA uniquement).

        Args:
            output_dir: Dossier de destination.

        Returns:
            Chemin du modèle sauvegardé.
        """
        save_path = output_dir or self.config.output_dir
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info(f"Challenger sauvegardé (LoRA adapters) : {save_path}")
        return save_path

    def load(self, model_path: Path) -> None:
        """Charge un modèle Challenger avec adapters LoRA.

        Args:
            model_path: Chemin vers les adapters LoRA sauvegardés.
        """
        from peft import PeftModel

        device = self.detect_device()
        hf_token = get_settings().huggingface_token or None

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            dtype=torch.bfloat16,
            token=hf_token,
        )
        self.model = PeftModel.from_pretrained(base_model, str(model_path))
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info(f"Challenger chargé depuis {model_path} sur {device}")


class Qwen3Classifier(LoRAClassifier):
    """Classifieur Challenger base sur `Qwen/Qwen3-4B` + LoRA.

    Specialise la classe generique `LoRAClassifier` avec les choix
    adaptes a Qwen3-4B :

    - Base : `Qwen/Qwen3-4B` (~4B parametres, Apache-2.0, multilingue natif,
      architecture dense transformer standard compatible ROCm).
    - Sortie : 2 classes (Green IT / Non Green IT) via tete de classification
      sequence (SEQ_CLS), adaptateurs LoRA appliques aux projections de
      l'attention uniquement (q/k/v/o). Le MLP est volontairement exclu
      pour garder la consommation VRAM compatible avec une RX 7900 XTX.
    - Hyperparametres par defaut calibres pour RX 7900 XTX 24 Go en BF16 :
      batch 2 + gradient accumulation x4 = batch effectif 8, sequences
      tronquees a la longueur definie dans `settings.trainer_max_length`
      (512 par defaut), gradient checkpointing active pour tenir la memoire.

    Le modele final (LoRA adapter) pese ~30 Mo vs ~8 Go pour le base model :
    c'est l'adaptateur seul qui est versionne dans `models/production/`
    et pousse vers DVC/MinIO.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        lora_config: LoraConfig | None = None,
    ) -> None:
        settings = get_settings()
        if config is None:
            config = TrainingConfig(
                nom_modele=settings.huggingface_model_trainer_base,
                output_dir=BASE_DIR / "models" / "qwen3",
                # epochs=2 : reduit de 3 a 2 le 2026-05-17 apres benchmark de
                # vitesse sur RX 7900 XTX. Qwen3-4B + LoRA all-linear + GC
                # tourne a ~10 sec/optimizer-step (50x plus lent que mDeBERTa)
                # ce qui rend K=5x3 = 15 trainings irrealiste (~30h). Avec
                # K=3x2 + 2 epochs = ~6-8h, on conserve la rigueur du
                # benchmark equitable. 2 epochs reste suffisant pour LoRA
                # (Unsloth recommande 1-3 epochs), surtout avec rsLoRA +
                # SWA en fin qui stabilise la convergence.
                epochs=2,
                batch_size=2,
                # lr=1e-4 (baisse vs 2e-4 historique) : avec target_modules
                # etendu a all-linear, la norme des gradients augmente -
                # 1e-4 est plus stable (Unsloth 2026, PEFT guide).
                learning_rate=1e-4,
                max_length=settings.trainer_max_length,
            )
        if lora_config is None:
            # r=16 (reduit de 32 le 2026-05-17 pour gagner du temps de
            # training, cf. commentaire epochs ci-dessus). Avec r=16 +
            # rsLoRA, la stabilite est preservee selon Unsloth's
            # "LoRA Hyperparameters Guide" qui considere r=16 alpha=16 ou
            # alpha=32 (ratio 1:1 ou 1:2) comme le sweet spot par defaut.
            # Gain training : ~10-15 % plus rapide vs r=32 + qualite quasi
            # identique sur classification simple (2 classes).
            # alpha=32 (ratio 2:1 maintenu).
            # use_rslora=True : essentiel pour preserver les gradients
            # avec target_modules="all-linear" (recommandation SOTA 2026).
            lora_config = LoraConfig(
                r=16,
                alpha=32,
                dropout=0.05,
                target_modules=list(QWEN3_LORA_TARGET_MODULES),
                use_rslora=True,
            )
        super().__init__(config, lora_config)


def _oversample_minority(
    texts: list[str],
    labels: list[int],
    target_ratio: float = 0.2,
) -> tuple[list[str], list[int]]:
    """Duplique les exemples de la classe minoritaire pour equilibrer le dataset.

    L'oversampling est essentiel quand la classe positive represente <1% du dataset.
    On cible un ratio de 20% pour que le modele puisse apprendre les patterns
    de la classe Green IT sans que les gradients explosent.

    Args:
        texts: Textes d'entree.
        labels: Labels correspondants (0 ou 1).
        target_ratio: Ratio cible pour la classe minoritaire (defaut 0.2 = 20%).

    Returns:
        Tuple (textes oversamples, labels oversamples) melanges aleatoirement.
    """
    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    minority_mask = labels_arr == 1
    majority_mask = ~minority_mask

    n_majority = int(majority_mask.sum())
    n_minority = int(minority_mask.sum())

    # Calculer combien de copies de la minorite on a besoin
    # target_ratio = n_minority_new / (n_majority + n_minority_new)
    n_minority_target = int(target_ratio * n_majority / (1.0 - target_ratio))
    n_copies = max(1, n_minority_target // max(n_minority, 1))

    logger.info(
        f"Oversampling : {n_minority} → {n_minority * n_copies} exemples positifs "
        f"(x{n_copies}), ratio cible {target_ratio:.0%}"
    )

    # Dupliquer la minorite
    minority_texts = np.tile(texts_arr[minority_mask], n_copies)
    minority_labels = np.tile(labels_arr[minority_mask], n_copies)

    # Combiner et melanger
    all_texts = np.concatenate([texts_arr[majority_mask], minority_texts])
    all_labels = np.concatenate([labels_arr[majority_mask], minority_labels])

    rng = np.random.default_rng(42)
    shuffle_idx = rng.permutation(len(all_texts))

    return all_texts[shuffle_idx].tolist(), all_labels[shuffle_idx].tolist()


def load_full_dataset(
    dataset_path: Path | None = None,
) -> tuple[list[str], list[int]]:
    """Charge l'integralite du Golden Dataset sans aucun split.

    Utile pour evaluer un modele non entraine (baseline) sur toutes les donnees
    disponibles, ou pour alimenter une boucle K-fold qui gere son propre split.
    Contrairement a `load_golden_dataset`, aucun oversampling n'est applique.

    Args:
        dataset_path: Chemin vers le CSV annote (defaut: data/golden_dataset.csv).

    Returns:
        Tuple (textes, labels) contenant tous les articles annotes.

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
        ValueError: Si la colonne `label_green_it` est absente.
    """
    path = dataset_path or (BASE_DIR / "data" / "golden_dataset.csv")
    if not path.exists():
        msg = f"Golden Dataset introuvable : {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)
    if "label_green_it" not in df.columns:
        msg = "Colonne 'label_green_it' manquante dans le dataset"
        raise ValueError(msg)

    df = df[df["label_green_it"].isin([0, 1])].copy()
    df["text"] = _build_text_column(df)

    n_green = int(df["label_green_it"].sum())
    logger.info(
        f"Dataset complet charge : {len(df)} articles (Green: {n_green}, Non: {len(df) - n_green})"
    )

    return df["text"].tolist(), df["label_green_it"].tolist()


def load_full_dataset_with_language(
    dataset_path: Path | None = None,
) -> tuple[list[str], list[int], list[str], list[str]]:
    """Charge le Golden Dataset avec les metadonnees utilisees par le K-fold.

    Comparee a ``load_full_dataset``, retourne en plus la langue de chaque
    article (necessaire pour la stratification croisee ``(langue x label)``)
    et le flag d'augmentation (pour exclure les variantes back-translation
    du val/test split et eviter la fuite d'evaluation).

    Accepte en entree le CSV standard (``golden_dataset.csv``) ou le CSV
    augmente (``golden_dataset_augmented.csv``). Si la colonne
    ``augmentation_source`` est absente, elle est peuplee avec des chaines
    vides (retro-compat).

    Args:
        dataset_path: Chemin CSV. Par defaut, privilegie le CSV augmente
            si present (``data/golden_dataset_augmented.csv``), sinon
            retombe sur ``data/golden_dataset.csv``.

    Returns:
        Tuple ``(texts, labels, langues, augmentation_sources)`` de 4 listes
        de meme longueur. ``augmentation_sources[i] == ""`` indique un
        article original, sinon c'est une variante a exclure du val/test.

    Raises:
        FileNotFoundError: Si aucun CSV exploitable n'existe.
        ValueError: Si la colonne ``langue`` est absente (re-executer
            ``scripts/export_golden_dataset.py`` pour la regenerer).
    """
    if dataset_path is None:
        augmented = BASE_DIR / "data" / "golden_dataset_augmented.csv"
        standard = BASE_DIR / "data" / "golden_dataset.csv"
        if augmented.exists():
            dataset_path = augmented
            logger.info(f"Utilisation du CSV augmente : {augmented.name}")
        else:
            dataset_path = standard

    if not dataset_path.exists():
        msg = f"Golden Dataset introuvable : {dataset_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(dataset_path)
    if "label_green_it" not in df.columns:
        msg = "Colonne 'label_green_it' manquante dans le dataset"
        raise ValueError(msg)
    if "langue" not in df.columns:
        msg = (
            "Colonne 'langue' manquante dans le dataset. "
            "Re-executer 'uv run python scripts/export_golden_dataset.py' "
            "pour regenerer le CSV avec la colonne langue."
        )
        raise ValueError(msg)

    df = df[df["label_green_it"].isin([0, 1])].copy()
    df["text"] = _build_text_column(df)
    df["langue"] = df["langue"].fillna("unk")

    # Flag d'augmentation : vide pour les originaux, non-vide pour les variantes.
    # Si la colonne n'existe pas (vieux CSV pre-augmentation), on considere
    # tous les articles comme originaux.
    if "augmentation_source" not in df.columns:
        df["augmentation_source"] = ""
    df["augmentation_source"] = df["augmentation_source"].fillna("")

    n_green = int(df["label_green_it"].sum())
    n_augmented = int((df["augmentation_source"] != "").sum())
    logger.info(
        f"Dataset complet charge : {len(df)} articles "
        f"(Green: {n_green}, Non: {len(df) - n_green}, "
        f"augmentes: {n_augmented})"
    )

    return (
        df["text"].tolist(),
        df["label_green_it"].tolist(),
        df["langue"].tolist(),
        df["augmentation_source"].tolist(),
    )


def _build_text_column(df: pd.DataFrame) -> pd.Series:
    """Construit la colonne de feature d'entrainement ``titre + resume``.

    Accepte deux formats de golden dataset pour une transition en douceur :

    - **Nouveau** (recommande) : colonne ``resume_classification`` produite
      par le LLM (Qwen3-4B-Instruct-2507 ou fallback local). Format uniforme
      et aligne train/inference.
    - **Legacy** : colonne ``contenu_extrait`` (500 premiers chars du
      contenu brut). Un warning est emis pour inciter a regenerer le CSV
      via ``scripts/generate_classification_summaries.py`` puis
      ``scripts/export_golden_dataset.py``.

    Args:
        df: DataFrame charge depuis le CSV golden.

    Returns:
        Serie pandas contenant la concatenation ``titre\\n\\nfeature``.

    Raises:
        ValueError: Si aucune des deux colonnes de feature n'est presente.
    """
    titre = df["titre"].fillna("") if "titre" in df.columns else pd.Series([""] * len(df))

    if "resume_classification" in df.columns:
        return titre + "\n\n" + df["resume_classification"].fillna("")

    if "contenu_extrait" in df.columns:
        logger.warning(
            "Golden dataset au format legacy (colonne 'contenu_extrait') : le modele "
            "sera entraine sur les 500 premiers caracteres du contenu brut au lieu "
            "du resume LLM uniforme. Regenerer via "
            "'uv run python scripts/generate_classification_summaries.py' puis "
            "'uv run python scripts/export_golden_dataset.py' pour beneficier du "
            "nouveau format aligne train/inference."
        )
        return titre + "\n\n" + df["contenu_extrait"].fillna("")

    msg = (
        "Golden dataset incomplet : ni 'resume_classification' ni 'contenu_extrait' "
        "ne sont presents. Regenerer le CSV via 'scripts/export_golden_dataset.py'."
    )
    raise ValueError(msg)


def load_golden_dataset(
    dataset_path: Path | None = None,
    *,
    oversample: bool = True,
    target_ratio: float = 0.2,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Charge et split le Golden Dataset annote avec gestion du desequilibre.

    Lit le CSV annote, combine titre + contenu pour des features plus riches,
    separe en train/test (80/20), et applique un oversampling de la classe
    minoritaire sur le train set uniquement (le test set reste intact).

    Args:
        dataset_path: Chemin vers le CSV annote (defaut: data/golden_dataset.csv).
        oversample: Active l'oversampling de la minorite sur le train set.
        target_ratio: Ratio cible pour la classe positive (defaut 0.2 = 20%).

    Returns:
        Tuple (train_texts, train_labels, test_texts, test_labels).

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
    """
    path = dataset_path or (BASE_DIR / "data" / "golden_dataset.csv")
    if not path.exists():
        msg = f"Golden Dataset introuvable : {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)
    logger.info(f"Golden Dataset charge : {len(df)} articles")

    # Verifier la colonne label
    if "label_green_it" not in df.columns:
        msg = "Colonne 'label_green_it' manquante dans le dataset"
        raise ValueError(msg)

    # Filtrer les entrees non annotees
    df = df[df["label_green_it"].isin([0, 1])].copy()

    # Combiner titre + resume de classification pour construire la feature
    # d'entrainement. Voir `_build_text_column` pour la gestion du fallback
    # legacy sur `contenu_extrait`.
    df["text"] = _build_text_column(df)

    n_green = int(df["label_green_it"].sum())
    n_non_green = len(df) - n_green
    logger.info(f"Articles annotes : {len(df)} (Green: {n_green}, Non: {n_non_green})")

    # Split stratifie 80/20
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_green_it"]
    )

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label_green_it"].tolist()

    # Oversampling sur le train set uniquement (test set intact)
    if oversample and n_green > 0:
        train_texts, train_labels = _oversample_minority(
            train_texts, train_labels, target_ratio=target_ratio
        )

    return (
        train_texts,
        train_labels,
        test_df["text"].tolist(),
        test_df["label_green_it"].tolist(),
    )


def _build_classifier_and_config(
    model_type: str,
) -> tuple[BaseClassifier, ExperimentConfig]:
    """Construit le classifieur et sa config MLflow selon le type demande.

    Args:
        model_type: Un de 'deberta', 'qwen2.5',
            'llama3.2', 'qwen3'.

    Returns:
        Tuple (classifieur, config MLflow).

    Raises:
        ValueError: Si model_type n'est pas reconnu.
    """
    if model_type not in VALID_MODELS:
        msg = f"Type inconnu : {model_type}. Valides : {', '.join(VALID_MODELS)}"
        raise ValueError(msg)

    if model_type == "deberta":
        classifier = DeBERTaClassifier()
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="deberta-v3-base",
            tags={"type": "deberta", "modele": "deberta-v3-base"},
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "learning_rate": classifier.config.learning_rate,
                "method": "full-finetuning",
            },
        )
    elif model_type == "mdeberta":
        classifier = MDeBERTaClassifier()
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="mdeberta-v3-base",
            tags={
                "type": "mdeberta",
                "modele": "mdeberta-v3-base",
                "multilingue": "oui",
                "method": "full-finetuning",
            },
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "learning_rate": classifier.config.learning_rate,
                "weight_decay": classifier.config.weight_decay,
                "warmup_ratio": classifier.config.warmup_ratio,
                "max_length": classifier.config.max_length,
                "method": "full-finetuning",
            },
        )
    elif model_type == "qwen2.5":
        classifier = LoRAClassifier(
            config=TrainingConfig(
                nom_modele="Qwen/Qwen2.5-3B",
                output_dir=BASE_DIR / "models" / "qwen2.5",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            ),
        )
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="qwen2.52.5-3b-lora",
            tags={"type": "qwen2.5", "modele": "qwen2.5-3b", "method": "lora"},
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "learning_rate": classifier.config.learning_rate,
                "method": "lora",
                "lora_r": classifier.lora_config.r,
                "lora_alpha": classifier.lora_config.alpha,
            },
        )
    elif model_type == "llama3.2":
        classifier = LoRAClassifier(
            config=TrainingConfig(
                nom_modele="meta-llama/Llama-3.2-3B",
                output_dir=BASE_DIR / "models" / "llama3.2",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            ),
        )
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="llama3.2-3.2-3b-lora",
            tags={"type": "llama3.2", "modele": "llama-3.2-3b", "method": "lora"},
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "learning_rate": classifier.config.learning_rate,
                "method": "lora",
                "lora_r": classifier.lora_config.r,
                "lora_alpha": classifier.lora_config.alpha,
            },
        )
    else:  # qwen3
        classifier = Qwen3Classifier()
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="qwen3-4b-lora",
            tags={
                "type": "qwen3",
                "modele": "qwen3-4b",
                "method": "lora",
                "multilingue": "oui",
            },
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "learning_rate": classifier.config.learning_rate,
                "max_length": classifier.config.max_length,
                "method": "lora",
                "lora_r": classifier.lora_config.r,
                "lora_alpha": classifier.lora_config.alpha,
                "lora_target_modules": ",".join(classifier.lora_config.target_modules or []),
            },
        )

    return classifier, exp_config


async def train_with_legacy_cv(
    model_type: str = "qwen3",
    *,
    n_splits: int = 5,
    random_state: int = 42,
    oversample_ratio: float = 0.2,
    train_final: bool = True,
) -> dict:
    """Entraine un challenger (Llama ou Qwen) via K-fold stratifie.

    Le K-fold permet d'evaluer la capacite du modele a generaliser de maniere
    robuste malgre le desequilibre extreme du dataset. Chaque article est utilise
    en test une seule fois (dans le fold ou il est tire) et en train K-1 fois.

    Pour chaque fold :
        1. Split stratifie des indices train/test.
        2. Oversampling de la classe minoritaire sur le train du fold uniquement.
        3. Entrainement d'un nouveau LoRAClassifier.
        4. Evaluation sur le test set du fold (sans oversampling).
        5. Stockage des predictions et des metriques par fold.

    A l'issue des K folds, un modele final est optionnellement ré-entraine sur
    l'integralite des donnees (c'est celui qui sera servi en production). Les
    metriques reportees sont celles du K-fold, car seules elles donnent une
    estimation honnete de la performance sur donnees non vues.

    Args:
        model_type: "llama3.2", "qwen2.5" ou "qwen3".
        n_splits: Nombre de folds (defaut 5).
        random_state: Seed pour la reproductibilite du split.
        oversample_ratio: Ratio cible de la minorite apres oversampling (sur le train de chaque fold).
        train_final: Si True, re-entraine un modele final sur tout le dataset apres le K-fold.

    Returns:
        Dictionnaire contenant :
            - `folds` : liste des metriques par fold (avec numero de fold et taille du test)
            - `aggregated` : moyenne et ecart-type pour chaque metrique cle
            - `global` : metriques calculees sur la concatenation des predictions des K folds
            - `final_model_trained` : True si le modele final a ete entraine et sauvegarde
            - `n_splits`, `random_state`, `oversample_ratio` : parametres utilises

    Raises:
        ValueError: Si `model_type` n'est pas un challenger valide.
    """
    import mlflow
    from sklearn.model_selection import StratifiedKFold

    from greentech.ai.mlops import prometheus_metrics as promm

    valid_challengers = ("llama3.2", "qwen2.5", "qwen3")
    if model_type not in valid_challengers:
        msg = (
            f"K-fold disponible uniquement pour les challengers "
            f"({', '.join(valid_challengers)}), pas pour {model_type}"
        )
        raise ValueError(msg)

    all_texts, all_labels = load_full_dataset()
    texts_arr = np.array(all_texts, dtype=object)
    labels_arr = np.array(all_labels)
    n_green = int(labels_arr.sum())

    logger.info(
        f"K-fold stratifie : K={n_splits}, {len(all_texts)} articles total "
        f"(Green: {n_green}, Non: {len(all_texts) - n_green})"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: list[dict] = []
    all_preds: list[int] = []
    all_true: list[int] = []
    all_latencies: list[float] = []

    # Un seul run MLflow parent pour tout le K-fold : metriques par fold loggees
    # avec step=fold_idx, puis metriques agregees finales en fin de run. Permet
    # de visualiser la stabilite du modele entre folds directement dans l'UI.
    cv_run_name = f"{model_type}-cv-k{n_splits}"
    cv_exp_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run=cv_run_name,
        tags={"phase": "k-fold-cv", "model_type": model_type},
        params={
            "model_type": model_type,
            "n_splits": n_splits,
            "random_state": random_state,
            "oversample_ratio": oversample_ratio,
            "dataset_total": len(all_texts),
            "dataset_green": n_green,
            "dataset_non_green": len(all_texts) - n_green,
            "train_final": train_final,
        },
    )

    with tracked_experiment(cv_exp_config):
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr), 1):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"  FOLD {fold_idx}/{n_splits}")
            logger.info("=" * 70)

            fold_start = time.perf_counter()

            train_texts_fold = texts_arr[train_idx].tolist()
            train_labels_fold = labels_arr[train_idx].tolist()
            test_texts_fold = texts_arr[test_idx].tolist()
            test_labels_fold = labels_arr[test_idx].tolist()

            n_green_test = int(sum(test_labels_fold))
            n_green_train = int(sum(train_labels_fold))
            logger.info(
                f"Train : {len(train_texts_fold)} (Green: {n_green_train})  "
                f"Test : {len(test_texts_fold)} (Green: {n_green_test})"
            )

            train_texts_fold, train_labels_fold = _oversample_minority(
                train_texts_fold, train_labels_fold, target_ratio=oversample_ratio
            )

            fold_output_dir = BASE_DIR / "models" / f"cv_fold_{fold_idx}"
            classifier, _ = _build_classifier_and_config(model_type)
            classifier.config.output_dir = fold_output_dir

            await classifier.train(
                train_texts_fold,
                train_labels_fold,
                test_texts_fold,
                test_labels_fold,
            )
            classifier.save()

            fold_preds: list[int] = []
            fold_latencies: list[float] = []
            for text in test_texts_fold:
                pred = await classifier.predict(text)
                fold_preds.append(pred.label.value)
                fold_latencies.append(pred.temps_ms)

            fold_metrics_dict = _compute_fold_metrics(test_labels_fold, fold_preds, fold_latencies)
            fold_metrics_dict["fold"] = fold_idx
            fold_metrics_dict["n_test"] = len(test_texts_fold)
            fold_metrics_dict["n_green_test"] = n_green_test
            fold_metrics.append(fold_metrics_dict)

            fold_duration = time.perf_counter() - fold_start

            # Logger toutes les metriques du fold dans MLflow, step=fold_idx :
            # permet de visualiser la courbe MCC/F1/Recall sur les K folds.
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in fold_metrics_dict.items()
                    if isinstance(v, int | float) and k != "fold"
                },
                step=fold_idx,
            )

            # Push vers Prometheus Pushgateway : alimente les dashboards
            # Grafana en temps reel (progression des folds + metriques live).
            promm.record_fold_metrics(
                model_type=model_type,
                run_name=cv_run_name,
                fold=fold_idx,
                total_folds=n_splits,
                metrics={k: v for k, v in fold_metrics_dict.items() if isinstance(v, int | float)},
                duration_seconds=fold_duration,
            )

            logger.info(
                f"Fold {fold_idx} : MCC={fold_metrics_dict['mcc']:.4f}, "
                f"F1={fold_metrics_dict['f1']:.4f}, "
                f"Recall={fold_metrics_dict['recall']:.4f}"
            )

            all_preds.extend(fold_preds)
            all_true.extend(test_labels_fold)
            all_latencies.extend(fold_latencies)

            del classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aggregated = _aggregate_fold_metrics(fold_metrics)
        global_metrics = _compute_fold_metrics(all_true, all_preds, all_latencies)

        # Metriques agregees (mean/std par critere) et globales (sur la
        # concatenation des predictions des K folds).
        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict):
                mlflow.log_metric(f"cv_{metric_name}_mean", float(stats["mean"]))
                mlflow.log_metric(f"cv_{metric_name}_std", float(stats["std"]))
        for metric_name, value in global_metrics.items():
            if isinstance(value, int | float):
                mlflow.log_metric(f"cv_global_{metric_name}", float(value))

        # Snapshot final pour Prometheus : moyenne + ecart-type du MCC,
        # essentiels pour le garde-fou de promotion (std <= 0.15).
        promm.record_cv_aggregated(
            model_type=model_type,
            run_name=cv_run_name,
            mcc_mean=aggregated["mcc"]["mean"],
            mcc_std=aggregated["mcc"]["std"],
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  K-FOLD TERMINE (K={n_splits})")
    logger.info("=" * 70)
    logger.info(
        f"  MCC        : {aggregated['mcc']['mean']:.4f} (+/- {aggregated['mcc']['std']:.4f})"
    )
    logger.info(
        f"  F1         : {aggregated['f1']['mean']:.4f} (+/- {aggregated['f1']['std']:.4f})"
    )
    logger.info(
        f"  Recall GIT : {aggregated['recall']['mean']:.4f} (+/- {aggregated['recall']['std']:.4f})"
    )
    logger.info(f"  Global MCC (toutes predictions concatenees) : {global_metrics['mcc']:.4f}")

    final_trained = False
    if train_final:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  ENTRAINEMENT DU MODELE FINAL SUR TOUT LE DATASET")
        logger.info("=" * 70)
        final_train_texts, final_train_labels = _oversample_minority(
            all_texts, all_labels, target_ratio=oversample_ratio
        )
        classifier, _ = _build_classifier_and_config(model_type)
        # Pas de validation set : on a deja mesure la performance via K-fold
        split = max(1, len(final_train_texts) // 20)
        await classifier.train(
            final_train_texts[split:],
            final_train_labels[split:],
            final_train_texts[:split],
            final_train_labels[:split],
        )
        classifier.save()
        final_trained = True
        logger.info(f"Modele final sauvegarde : {classifier.config.output_dir}")
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "folds": fold_metrics,
        "aggregated": aggregated,
        "global": global_metrics,
        "final_model_trained": final_trained,
        "n_splits": n_splits,
        "random_state": random_state,
        "oversample_ratio": oversample_ratio,
    }


async def train_with_unified_protocol(
    model_type: str,
    *,
    n_splits: int = 5,
    n_seeds: int = 3,
    base_random_state: int = 42,
    strict_stratification: bool = False,
) -> dict:
    """Entraine un modele selon le protocole unifie B3 (avril 2026).

    Remplace ``train_with_legacy_cv`` pour les modeles cibles du
    benchmark B4 (``qwen3`` et ``mdeberta``). Protocole
    issu de la synthese de 3 agents de recherche :

    1. **Stratification croisee ``(langue x label)``** via
       ``MultilabelStratifiedKFold`` (Sechidis 2011) au lieu d'une simple
       stratification sur le label. Necessaire car une langue (FR 25 %
       du volume) porte 59 % des positifs : sans cela, l'ecart-type MCC
       inter-fold peut exploser au-dela de la cible 0.10.
    2. **3 seeds par fold** (15 trainings au total par modele). Reduit la
       variance inter-seed connue pour BERT/DeBERTa sur petit dataset
       (±0.03-0.05 MCC). Moyenner sur 3 seeds stabilise sigma < 0.10.
    3. **``class_weight=[1.0, N_neg/N_pos]``** sur la CrossEntropy en
       remplacement de l'oversampling x84 historique. Les 3 agents
       convergent sur ce choix pour ratio modere 1:10.5 (Focal Loss
       reservee a ratio > 1:50).
    4. **Exclusion des variantes back-translation du val/test** : si le
       CSV augmente est utilise, les articles ``augmentation_source != ""``
       ne vont QUE dans le train split de chaque fold, jamais en val/test,
       pour eviter la fuite d'evaluation (tester sur un texte derive d'un
       texte vu en train).
    5. **Calibration post-fold** via ``calibration.TemperatureScaler``
       (LBFGS sur NLL val) + ``find_optimal_threshold`` (scan [0.05, 0.95]
       argmax MCC). T et threshold sont moyennes au niveau modele,
       persistes dans ``models/<model>/{temperature,optimal_threshold}.json``.
    6. **Sauvegarde par fold** dans
       ``models/<model>/folds/fold_X_seed_Y/``. L'ensembling (moyenne des
       logits ou fusion d'adapters LoRA) est effectue a l'inference
       (cf. ``inference.py``).

    Args:
        model_type: ``"qwen3"`` (LoRA Qwen3-4B) ou
            ``"mdeberta"`` (mDeBERTa-v3-base full fine-tune).
        n_splits: Nombre de folds K-fold (defaut 5).
        n_seeds: Nombre de seeds par fold (defaut 3).
        base_random_state: Seed de base pour le split K-fold. Les seeds
            intra-fold sont derives : ``base_random_state + seed_idx``.
        strict_stratification: Si ``True``, leve ``AssertionError`` des
            qu'un fold devie de plus de 2 points de pourcentage par
            rapport aux ratios cible (EN/FR/Green global/Green par langue).
            Defaut ``False`` : logge un warning et poursuit l'entrainement.

    Returns:
        Dictionnaire contenant :

        - ``runs`` : liste des metriques par (fold, seed)
        - ``aggregated`` : moyennes et ecarts-types inter-fold
        - ``calibration`` : T et threshold finaux agreges
        - ``metadata`` : parametres utilises, chemins des artefacts

    Raises:
        ValueError: Si ``model_type`` n'est pas un modele cible du B4.
        AssertionError: Si ``strict_stratification=True`` et un fold devie
            au-dela de la tolerance de 2 pp.
    """
    import mlflow
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    from greentech.ai.mlops import prometheus_metrics as promm
    from greentech.ai.mlops.calibration import (
        TemperatureScaler,
        find_optimal_threshold,
        save_calibration,
    )

    supported_models = ("qwen3", "mdeberta")
    if model_type not in supported_models:
        msg = (
            f"Protocole unifie disponible uniquement pour {supported_models}, "
            f"pas pour {model_type}. Utiliser train_with_legacy_cv pour "
            f"les modeles legacy."
        )
        raise ValueError(msg)

    # --- Chargement du dataset avec metadonnees ---
    all_texts, all_labels, all_langues, all_aug_sources = load_full_dataset_with_language()
    texts_arr = np.array(all_texts, dtype=object)
    labels_arr = np.array(all_labels, dtype=int)
    langues_arr = np.array(all_langues, dtype=object)
    aug_sources_arr = np.array(all_aug_sources, dtype=object)

    # Separer originaux et variantes : le K-fold s'applique uniquement sur les
    # originaux pour eviter la fuite d'evaluation (une variante augmentee
    # d'un article garde la meme semantique, donc si original en train et
    # variante en val, le modele a vu la "reponse").
    original_mask = aug_sources_arr == ""
    n_originals = int(original_mask.sum())
    n_augmented = int((~original_mask).sum())

    logger.info(
        f"Protocole unifie : {n_originals} originaux + {n_augmented} augmentes "
        f"(K={n_splits} folds x {n_seeds} seeds = {n_splits * n_seeds} trainings)"
    )

    # Labels composes pour MultilabelStratifiedKFold : chaque article devient
    # un vecteur binaire multi-label (lang_en, lang_fr, label_1). Le stratifier
    # equilibre les 3 dimensions simultanement dans chaque fold.
    lang_en = (langues_arr[original_mask] == "en").astype(int)
    lang_fr = (langues_arr[original_mask] == "fr").astype(int)
    label_pos = labels_arr[original_mask]
    strat_labels = np.stack([lang_en, lang_fr, label_pos], axis=1)

    original_indices = np.where(original_mask)[0]

    # Precompute : pour chaque original positif, la liste de ses variantes
    # indexees par position dans original_indices. Necessaire pour injecter
    # les variantes uniquement dans le train split de chaque fold.
    variant_indices_by_original = _build_variant_index(texts_arr, original_mask)

    cv_run_name = f"{model_type}-unified-k{n_splits}-s{n_seeds}"
    cv_exp_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run=cv_run_name,
        tags={
            "phase": "b3-unified-protocol",
            "model_type": model_type,
            "stratification": "multilabel_langue_label",
            "loss_strategy": "weighted_ce",
            "augmentation": "opus-mt-backtranslation" if n_augmented > 0 else "none",
        },
        params={
            "model_type": model_type,
            "n_splits": n_splits,
            "n_seeds": n_seeds,
            "base_random_state": base_random_state,
            "dataset_total_originals": n_originals,
            "dataset_total_augmented": n_augmented,
            "dataset_green_originals": int(label_pos.sum()),
        },
    )

    model_output_root = _resolve_model_output_root(model_type)
    folds_root = model_output_root / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)

    run_metrics: list[dict] = []
    temperatures: list[float] = []
    thresholds: list[float] = []

    with tracked_experiment(cv_exp_config):
        mskf = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=base_random_state
        )

        for fold_idx, (train_idx_local, val_idx_local) in enumerate(
            mskf.split(strat_labels, strat_labels), 1
        ):
            # Convertir les indices locaux (dans original_indices) en indices globaux
            train_orig_global = original_indices[train_idx_local]
            val_orig_global = original_indices[val_idx_local]

            # Ajouter les variantes correspondantes au train (jamais au val)
            augment_global = _collect_variants_for_train(
                train_orig_global, variant_indices_by_original
            )
            train_global = np.concatenate([train_orig_global, augment_global])

            train_texts_fold = texts_arr[train_global].tolist()
            train_labels_fold = labels_arr[train_global].tolist()
            val_texts_fold = texts_arr[val_orig_global].tolist()
            val_labels_fold = labels_arr[val_orig_global].tolist()
            val_langues_fold = langues_arr[val_orig_global].tolist()

            observed_ratios = _log_fold_split_stats(
                fold_idx=fold_idx,
                n_splits=n_splits,
                train_texts_fold=train_texts_fold,
                train_labels_fold=train_labels_fold,
                val_texts_fold=val_texts_fold,
                val_labels_fold=val_labels_fold,
                val_langues_fold=val_langues_fold,
                n_augmented_in_train=len(augment_global),
                strict_stratification=strict_stratification,
            )

            # Traçabilité MLflow : les ratios observes par fold permettent de
            # verifier la qualite de la stratification apres coup dans l'UI.
            mlflow.log_metrics(
                {
                    f"fold_{fold_idx}_val_{key}": float(value)
                    for key, value in observed_ratios.items()
                },
                step=fold_idx,
            )

            # class_weight calcule sur le train set (ratio varie legerement par fold)
            class_weight = compute_class_weight(train_labels_fold)

            for seed_idx in range(n_seeds):
                seed = base_random_state + seed_idx
                logger.info("")
                logger.info(
                    f"  --- FOLD {fold_idx}/{n_splits} SEED {seed_idx + 1}/{n_seeds} "
                    f"(seed={seed}) ---"
                )

                fold_start = time.perf_counter()
                fold_output_dir = folds_root / f"fold_{fold_idx}_seed_{seed_idx + 1}"

                classifier, _ = _build_classifier_and_config(model_type)
                classifier.config.output_dir = fold_output_dir
                classifier.config.seed = seed
                classifier.class_weight = class_weight

                await classifier.train(
                    train_texts_fold,
                    train_labels_fold,
                    val_texts_fold,
                    val_labels_fold,
                )
                classifier.save()

                # Inference val pour recuperer probabilites et latences
                val_probas: list[float] = []
                val_preds_argmax: list[int] = []
                val_latencies: list[float] = []
                for text in val_texts_fold:
                    pred = await classifier.predict(text)
                    val_preds_argmax.append(pred.label.value)
                    val_latencies.append(pred.temps_ms)
                    # proba_positive est None pour les classifieurs tiers, on
                    # fallback sur score_confiance*sign dans ce cas marginal.
                    if pred.proba_positive is not None:
                        val_probas.append(pred.proba_positive)
                    elif pred.est_green_it:
                        val_probas.append(pred.score_confiance)
                    else:
                        val_probas.append(1.0 - pred.score_confiance)

                val_probas_arr = np.asarray(val_probas, dtype=np.float32)
                val_labels_arr = np.asarray(val_labels_fold, dtype=np.int64)

                # Calibration : temperature sur logits approximatifs + seuil MCC
                logits_approx = _probas_to_binary_logits(val_probas_arr)
                t_scaler = TemperatureScaler()
                t_result = t_scaler.fit(logits_approx, val_labels_arr)
                calibrated_probs = t_scaler.transform(logits_approx)[:, 1]
                threshold_result = find_optimal_threshold(
                    val_labels_arr, calibrated_probs, metric="mcc"
                )

                save_calibration(
                    fold_output_dir,
                    temperature=t_result,
                    threshold=threshold_result,
                )

                # Recalculer les predictions avec seuil optimal pour logger les metriques
                calibrated_preds = (calibrated_probs >= threshold_result.threshold).astype(int)
                fold_metrics_dict = _compute_fold_metrics(
                    val_labels_arr.tolist(),
                    calibrated_preds.tolist(),
                    val_latencies,
                )
                fold_metrics_dict.update(
                    {
                        "fold": fold_idx,
                        "seed_idx": seed_idx + 1,
                        "seed": seed,
                        "n_val": len(val_texts_fold),
                        "n_green_val": int(val_labels_arr.sum()),
                        "temperature": t_result.temperature,
                        "threshold": threshold_result.threshold,
                        "duration_s": time.perf_counter() - fold_start,
                    }
                )
                run_metrics.append(fold_metrics_dict)
                temperatures.append(t_result.temperature)
                thresholds.append(threshold_result.threshold)

                mlflow.log_metrics(
                    {
                        k: float(v)
                        for k, v in fold_metrics_dict.items()
                        if isinstance(v, int | float) and k not in ("fold", "seed_idx", "seed")
                    },
                    step=(fold_idx - 1) * n_seeds + seed_idx + 1,
                )
                promm.record_fold_metrics(
                    model_type=model_type,
                    run_name=cv_run_name,
                    fold=fold_idx,
                    total_folds=n_splits,
                    metrics={
                        k: v for k, v in fold_metrics_dict.items() if isinstance(v, int | float)
                    },
                    duration_seconds=fold_metrics_dict["duration_s"],
                )

                logger.info(
                    f"Fold {fold_idx}.{seed_idx + 1} | MCC={fold_metrics_dict['mcc']:.4f} "
                    f"F1={fold_metrics_dict['f1']:.4f} "
                    f"Recall={fold_metrics_dict['recall']:.4f} "
                    f"T={t_result.temperature:.3f} "
                    f"seuil={threshold_result.threshold:.2f}"
                )

                del classifier
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        aggregated = _aggregate_fold_metrics(run_metrics)
        mean_temperature = float(np.mean(temperatures))
        mean_threshold = float(np.mean(thresholds))

        # Persister T et threshold moyens au niveau modele pour l'inference prod
        from greentech.ai.mlops.calibration import TemperatureResult, ThresholdResult

        save_calibration(
            model_output_root,
            temperature=TemperatureResult(
                temperature=mean_temperature,
                nll_before=float(np.mean([r.get("duration_s", 0.0) * 0.0 for r in run_metrics])),
                nll_after=0.0,
                n_iterations=0,
            ),
            threshold=ThresholdResult(
                threshold=mean_threshold,
                metric="mcc",
                value=float(aggregated["mcc"]["mean"]),
            ),
        )

        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict):
                mlflow.log_metric(f"cv_{metric_name}_mean", float(stats["mean"]))
                mlflow.log_metric(f"cv_{metric_name}_std", float(stats["std"]))
        mlflow.log_metric("cv_temperature_mean", mean_temperature)
        mlflow.log_metric("cv_threshold_mean", mean_threshold)

        promm.record_cv_aggregated(
            model_type=model_type,
            run_name=cv_run_name,
            mcc_mean=aggregated["mcc"]["mean"],
            mcc_std=aggregated["mcc"]["std"],
        )

        # Assemblage final B3.5 : selection de la meilleure seed par fold,
        # fusion LoRA (Qwen3) ou generation d'un ensemble logit-average (mDeBERTa).
        # Produit `ensemble_config.json` consomme par `inference.py` au chargement.
        ensemble_info = _build_ensemble(
            model_type=model_type,
            model_output_root=model_output_root,
            run_metrics=run_metrics,
            folds_root=folds_root,
            mean_temperature=mean_temperature,
            mean_threshold=mean_threshold,
            aggregated=aggregated,
        )

        mlflow.log_metric("ensemble_n_folds", float(len(ensemble_info["folds"])))
        mlflow.set_tag("ensemble_strategy", ensemble_info["strategy"])

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  PROTOCOLE UNIFIE TERMINE ({n_splits} folds x {n_seeds} seeds)")
    logger.info("=" * 70)
    logger.info(
        f"  MCC        : {aggregated['mcc']['mean']:.4f} (+/- {aggregated['mcc']['std']:.4f})"
    )
    logger.info(
        f"  F1         : {aggregated['f1']['mean']:.4f} (+/- {aggregated['f1']['std']:.4f})"
    )
    logger.info(
        f"  Recall GIT : {aggregated['recall']['mean']:.4f} (+/- {aggregated['recall']['std']:.4f})"
    )
    logger.info(f"  Temperature moyenne : {mean_temperature:.4f}")
    logger.info(f"  Threshold optimal moyen : {mean_threshold:.4f}")
    logger.info(f"  Ensemble strategy   : {ensemble_info['strategy']}")
    logger.info(f"  Ensemble folds      : {len(ensemble_info['folds'])}")
    logger.info(f"  Artefacts : {model_output_root}")

    return {
        "runs": run_metrics,
        "aggregated": aggregated,
        "calibration": {
            "temperature_mean": mean_temperature,
            "temperature_per_run": temperatures,
            "threshold_mean": mean_threshold,
            "threshold_per_run": thresholds,
        },
        "ensemble": ensemble_info,
        "metadata": {
            "model_type": model_type,
            "n_splits": n_splits,
            "n_seeds": n_seeds,
            "base_random_state": base_random_state,
            "n_originals": n_originals,
            "n_augmented": n_augmented,
            "model_output_root": str(model_output_root),
        },
    }


def _build_variant_index(
    texts_arr: np.ndarray,
    original_mask: np.ndarray,
) -> dict[int, list[int]]:
    """Mappe chaque indice global d'original aux indices globaux de ses variantes.

    Pour chaque variante (``augmentation_source != ""``), on rattache son
    indice global a l'original qui partage le meme titre (le titre est
    strictement identique entre original et variante puisque seul le
    resume est traduit).

    Args:
        texts_arr: Tous les textes (originaux + variantes), chaque texte
            etant au format ``titre\\n\\nresume``.
        original_mask: Mask booleen des originaux dans ``texts_arr``.

    Returns:
        Dict ``{indice_global_original: [indices_globaux_variantes]}``.
    """
    # Strategie simple et robuste : regrouper par titre (premiere ligne),
    # qui est strictement identique entre original et sa variante (on ne
    # traduit que le resume). Pour les originaux sans variante, la liste
    # est simplement vide.
    title_to_originals: dict[str, list[int]] = {}
    for idx in range(len(texts_arr)):
        if not original_mask[idx]:
            continue
        title = str(texts_arr[idx]).split("\n\n", 1)[0]
        title_to_originals.setdefault(title, []).append(idx)

    variants_by_original: dict[int, list[int]] = {
        idx: [] for idx in range(len(texts_arr)) if original_mask[idx]
    }
    for idx in range(len(texts_arr)):
        if original_mask[idx]:
            continue
        title = str(texts_arr[idx]).split("\n\n", 1)[0]
        matches = title_to_originals.get(title, [])
        if not matches:
            # Variante sans original : incoherence de dataset, on ignore
            # silencieusement (sera remontee par l'export stats).
            continue
        # Une variante est rattachee au premier original qui partage son titre.
        # En pratique, chaque titre est unique dans notre corpus, donc matches[0].
        variants_by_original[matches[0]].append(idx)

    return variants_by_original


def _collect_variants_for_train(
    train_orig_global: np.ndarray,
    variants_by_original: dict[int, list[int]],
) -> np.ndarray:
    """Retourne les indices globaux des variantes correspondant au train split."""
    variants: list[int] = []
    for orig_idx in train_orig_global:
        variants.extend(variants_by_original.get(int(orig_idx), []))
    return np.array(variants, dtype=int)


# Distribution cible du dataset final : un fold bien stratifie doit rester
# dans une fenetre de +/- 2 points de pourcentage autour de ces valeurs.
# Au-dela, la variance inter-fold du MCC peut exploser.
#
# Calibre le 2026-05-17 sur le dataset post-Phase 2 (11 664 articles dont
# 2 124 Green IT). La Phase 2 (auto-correction sources pures + LLM judge v2
# + audit multi-agents + annotation manuelle ciblee) a fait passer le
# nombre de Green IT confirmes de 1 018 (8.73 %) a 2 124 (18.21 %).
# Le ratio_green_fr atteint 51.95 % car GreenIT.fr est massivement
# represente en FR et est 100 % Green IT par construction.
#
# Calculs exacts (cf. scripts/_recalc_stratification_targets.py si besoin
# de regenerer apres futur enrichissement) :
#   Total                : 11 664 articles
#   EN                   : 8 719 (74.75 %)
#   FR                   : 2 945 (25.25 %)
#   Green IT total       : 2 124 (18.21 %)
#   Green IT EN          : 594 (6.81 % des EN)
#   Green IT FR          : 1 530 (51.95 % des FR)
_STRATIFICATION_TARGET_RATIOS: dict[str, float] = {
    "ratio_en": 0.7475,
    "ratio_fr": 0.2525,
    "ratio_green_global": 0.1821,
    "ratio_green_en": 0.0681,
    "ratio_green_fr": 0.5195,
}
_STRATIFICATION_TOLERANCE_PP: float = 0.02


def _check_fold_stratification(
    *,
    fold_idx: int,
    observed: dict[str, float],
    strict: bool,
) -> list[str]:
    """Compare les ratios observes du fold aux cibles et emet des alertes.

    Args:
        fold_idx: Numero du fold (1-indexe) pour le logging.
        observed: Ratios observes sur le val set (memes cles que
            ``_STRATIFICATION_TARGET_RATIOS``).
        strict: Si ``True``, leve ``AssertionError`` a la premiere
            deviation superieure a la tolerance. Si ``False``, logge un
            warning par ratio devie et retourne la liste.

    Returns:
        Liste des messages d'ecart detectes (vide si le fold est conforme).

    Raises:
        AssertionError: Si ``strict=True`` et au moins un ratio devie au-dela
            de la tolerance.
    """
    deviations: list[str] = []
    for key, target in _STRATIFICATION_TARGET_RATIOS.items():
        obs = observed.get(key)
        if obs is None:
            continue
        diff = abs(obs - target)
        if diff > _STRATIFICATION_TOLERANCE_PP:
            msg = (
                f"Fold {fold_idx} : {key}={obs:.4f} "
                f"(cible={target:.4f}, ecart={diff * 100:.2f}pp, "
                f"tolerance={_STRATIFICATION_TOLERANCE_PP * 100:.0f}pp)"
            )
            deviations.append(msg)
            if strict:
                raise AssertionError(
                    f"Stratification stricte violee : {msg}. "
                    "Desactive le flag strict_stratification pour un warning uniquement."
                )
            logger.warning(f"  [STRATIFICATION] {msg}")
    return deviations


def _log_fold_split_stats(
    *,
    fold_idx: int,
    n_splits: int,
    train_texts_fold: list[str],
    train_labels_fold: list[int],
    val_texts_fold: list[str],
    val_labels_fold: list[int],
    val_langues_fold: list[str],
    n_augmented_in_train: int,
    strict_stratification: bool = False,
) -> dict[str, float]:
    """Logge les statistiques de split du fold et verifie la stratification.

    Args:
        fold_idx: Numero du fold courant (1-indexe).
        n_splits: Nombre total de folds du K-fold.
        train_texts_fold: Textes du train split (originaux + variantes).
        train_labels_fold: Labels du train split.
        val_texts_fold: Textes du val split (originaux uniquement).
        val_labels_fold: Labels du val split.
        val_langues_fold: Codes langue du val split (``"en"`` / ``"fr"``).
        n_augmented_in_train: Nombre de variantes back-translation injectees
            dans le train split.
        strict_stratification: Si ``True``, leve ``AssertionError`` des qu'un
            ratio observe devie de plus de 2 pp par rapport a la cible
            attendue. Defaut ``False`` : logge un warning et poursuit.

    Returns:
        Dictionnaire des ratios observes pour logging MLflow ulterieur.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  FOLD {fold_idx}/{n_splits}")
    logger.info("=" * 70)
    n_train_green = int(sum(train_labels_fold))
    n_val_green = int(sum(val_labels_fold))
    val_en = sum(1 for lang in val_langues_fold if lang == "en")
    val_fr = sum(1 for lang in val_langues_fold if lang == "fr")
    n_val = len(val_texts_fold)
    val_green_en = sum(
        1
        for lang, label in zip(val_langues_fold, val_labels_fold, strict=True)
        if lang == "en" and label == 1
    )
    val_green_fr = sum(
        1
        for lang, label in zip(val_langues_fold, val_labels_fold, strict=True)
        if lang == "fr" and label == 1
    )
    logger.info(
        f"  Train : {len(train_texts_fold)} (Green: {n_train_green}, "
        f"augmentes: {n_augmented_in_train})"
    )
    logger.info(f"  Val   : {n_val} (Green: {n_val_green}, EN: {val_en}, FR: {val_fr})")

    observed: dict[str, float] = {
        "ratio_en": val_en / n_val if n_val else 0.0,
        "ratio_fr": val_fr / n_val if n_val else 0.0,
        "ratio_green_global": n_val_green / n_val if n_val else 0.0,
        "ratio_green_en": val_green_en / val_en if val_en else 0.0,
        "ratio_green_fr": val_green_fr / val_fr if val_fr else 0.0,
    }

    _check_fold_stratification(
        fold_idx=fold_idx,
        observed=observed,
        strict=strict_stratification,
    )

    return observed


def _probas_to_binary_logits(probas_positive: np.ndarray) -> np.ndarray:
    """Reconstruit une matrice logits (n, 2) depuis les probas de classe positive.

    Utilise pour alimenter ``TemperatureScaler`` qui attend des logits.
    Les probabilites sont mappees vers des logits via la fonction logit
    inverse, avec clipping numerique pour eviter inf/-inf sur les cas extremes.
    """
    eps = 1e-7
    clipped = np.clip(probas_positive, eps, 1.0 - eps)
    logit_pos = np.log(clipped / (1.0 - clipped))
    logit_neg = -logit_pos
    return np.stack([logit_neg, logit_pos], axis=1).astype(np.float32)


def _resolve_model_output_root(model_type: str) -> Path:
    """Retourne le dossier racine du modele selon son type."""
    mapping = {
        "qwen3": BASE_DIR / "models" / "qwen3",
        "mdeberta": BASE_DIR / "models" / "mdeberta",
    }
    if model_type not in mapping:
        msg = f"Pas de racine definie pour model_type={model_type}"
        raise ValueError(msg)
    return mapping[model_type]


def _compute_fold_metrics(
    y_true: list[int],
    y_pred: list[int],
    latencies_ms: list[float] | None = None,
) -> dict[str, float | int]:
    """Calcule les metriques pour un seul fold ou pour l'ensemble concatene.

    Version locale de la fonction utilisee dans le pipeline, pour eviter une
    dependance circulaire entre modules.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificite = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: dict[str, float | int] = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "specificite": specificite,
        "vrais_positifs": int(tp),
        "vrais_negatifs": int(tn),
        "faux_positifs": int(fp),
        "faux_negatifs": int(fn),
    }

    if latencies_ms:
        metrics["latence_moyenne_ms"] = float(np.mean(latencies_ms))

    return metrics


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict[str, dict[str, float | list[float]]]:
    """Agrege les metriques des K folds (moyenne, ecart-type, liste des valeurs)."""
    keys_to_aggregate = (
        "mcc",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificite",
    )
    aggregated: dict[str, dict[str, float | list[float]]] = {}
    for key in keys_to_aggregate:
        values = [float(f[key]) for f in fold_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }
    return aggregated


def _select_best_seed_per_fold(run_metrics: list[dict]) -> list[dict]:
    """Selectionne la meilleure seed (max MCC val) pour chaque fold.

    En cas d'egalite parfaite sur le MCC (tres rare), on conserve la seed
    avec le F1 le plus eleve pour departager. Resultat : un checkpoint par
    fold, soit K checkpoints au total pour l'ensemble.

    Args:
        run_metrics: Liste des dicts ``fold_metrics_dict`` produits par
            ``train_with_unified_protocol`` (un par (fold, seed)). Chaque
            entree doit contenir au minimum les cles ``fold``, ``seed_idx``,
            ``mcc``, ``f1``.

    Returns:
        Liste de K entrees, une par fold, correspondant a la meilleure seed.
    """
    folds_by_idx: dict[int, list[dict]] = {}
    for entry in run_metrics:
        folds_by_idx.setdefault(int(entry["fold"]), []).append(entry)

    best_per_fold: list[dict] = []
    for fold_idx in sorted(folds_by_idx):
        candidates = folds_by_idx[fold_idx]
        best = max(candidates, key=lambda e: (float(e["mcc"]), float(e["f1"])))
        best_per_fold.append(best)
    return best_per_fold


def _build_ensemble(
    *,
    model_type: str,
    model_output_root: Path,
    run_metrics: list[dict],
    folds_root: Path,
    mean_temperature: float,
    mean_threshold: float,
    aggregated: dict,
) -> dict:
    """Assemble l'ensemble final a partir des K*N checkpoints K-fold.

    Deux strategies selon l'architecture du modele :

    * **Qwen3-4B + LoRA** (``strategy="merge_lora"``) : fusionne les K adapters
      LoRA (meilleure seed par fold) dans les poids du base model via
      ``PeftModel.merge_and_unload()``. Les poids fusionnes sont moyennes
      arithmetiquement entre les K folds (equivalent SWA light sur les
      deltas LoRA). Sauvegarde dans ``models/qwen3/merged/`` + genere un
      ``adapter_config.json`` pointant vers ce merged model pour que
      ``inference.py`` le charge comme un modele LoRA classique. Cout
      inference : **1x** (un seul modele en VRAM).

    * **mDeBERTa-v3-base** (``strategy="logit_average"``) : pas de fusion
      de poids possible (architectures full fine-tune, pas LoRA). Les K
      checkpoints sont conserves a leur emplacement ``folds/fold_X_seed_Y/``
      et moyennes a l'inference (logit averaging, cf. ``EnsembleClassifier``
      dans ``inference.py``). Cout inference : **~5x** latence, ~5.5 Go VRAM.

    Dans les deux cas, le fichier ``ensemble_config.json`` est ecrit a la
    racine du modele (``model_output_root``) avec la liste des folds
    selectionnes, leurs metriques, et la strategie. C'est ce fichier qui
    active l'ensemble a l'inference.

    Args:
        model_type: ``"qwen3"`` ou ``"mdeberta"``.
        model_output_root: Dossier racine du modele (``models/qwen3`` ou
            ``models/mdeberta``).
        run_metrics: Liste de tous les (fold, seed) metrics.
        folds_root: Dossier contenant les K*N checkpoints fold_X_seed_Y.
        mean_temperature: Temperature moyenne K-fold (deja persistee via
            ``save_calibration``, reporte ici pour tracabilite).
        mean_threshold: Seuil moyen K-fold (idem).
        aggregated: Metriques agregees K-fold (``_aggregate_fold_metrics``).

    Returns:
        Dict avec ``strategy``, ``folds`` (liste), ``inference_model_path(s)``,
        ``metadata``. Ce dict est ecrit tel quel dans ``ensemble_config.json``.
    """
    best_per_fold = _select_best_seed_per_fold(run_metrics)

    folds_info = [
        {
            "fold": int(entry["fold"]),
            "seed_idx": int(entry["seed_idx"]),
            "seed": int(entry["seed"]),
            "mcc": float(entry["mcc"]),
            "f1": float(entry["f1"]),
            "recall": float(entry["recall"]),
            "precision": float(entry["precision"]),
            "temperature": float(entry["temperature"]),
            "threshold": float(entry["threshold"]),
            "checkpoint_path": str(folds_root / f"fold_{entry['fold']}_seed_{entry['seed_idx']}"),
        }
        for entry in best_per_fold
    ]

    if model_type == "qwen3":
        merged_dir = model_output_root / "merged"
        _merge_lora_adapters(
            fold_checkpoints=[Path(f["checkpoint_path"]) for f in folds_info],
            output_dir=merged_dir,
        )
        strategy = "merge_lora"
        inference_key: dict[str, str | list[str]] = {"inference_model_path": str(merged_dir)}
    elif model_type == "mdeberta":
        strategy = "logit_average"
        inference_key = {"inference_model_paths": [f["checkpoint_path"] for f in folds_info]}
    else:
        msg = f"Ensemble non supporte pour model_type={model_type}"
        raise ValueError(msg)

    ensemble_info = {
        "strategy": strategy,
        "model_type": model_type,
        "folds": folds_info,
        **inference_key,
        "calibration": {
            "temperature": float(mean_temperature),
            "threshold": float(mean_threshold),
        },
        "metadata": {
            "built_at": datetime.now(UTC).isoformat(),
            "n_folds": len(folds_info),
            "cv_mcc_mean": float(aggregated["mcc"]["mean"]),
            "cv_mcc_std": float(aggregated["mcc"]["std"]),
        },
    }

    config_path = model_output_root / "ensemble_config.json"
    config_path.write_text(
        json.dumps(ensemble_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Ensemble config ecrit : {config_path} (strategy={strategy})")

    return ensemble_info


def _merge_lora_adapters(
    *,
    fold_checkpoints: list[Path],
    output_dir: Path,
    combination_type: str = "ties",
    density: float = 0.5,
) -> None:
    """Fusionne K adapters LoRA dans un unique modele prod via TIES-merging.

    Strategie par defaut : **TIES-merging** (Yadav et al. NeurIPS 2023,
    arXiv:2306.01708) via ``PeftModel.add_weighted_adapter``. TIES applique
    trois etapes : (1) ``trim`` ne garde que les ``density`` fraction des
    parametres avec la plus forte magnitude par tenseur, (2) ``sign-elect``
    resout les conflits de signe par vote majoritaire pondere, (3) ``merge``
    moyenne les parametres survivants. Documente comme superieur a la
    moyenne arithmetique naive sur des fold ensembles (gain typique
    +1 a +3 MCC en classification, HF PEFT blog 2025).

    En cas d'echec (lib PEFT incompatible ou autre), fallback automatique
    sur la moyenne arithmetique des deltas A/B (``_average_lora_deltas``,
    methode legacy "uniform soup" Wortsman et al. ICML 2022).

    Args:
        fold_checkpoints: Liste des dossiers de checkpoints LoRA (un par
            fold, issus de ``_select_best_seed_per_fold``).
        output_dir: Dossier destination pour le merged model (typiquement
            ``models/qwen3/merged/``). Sera cree si besoin.
        combination_type: Strategie de fusion PEFT. Valeurs supportees :
            ``"ties"`` (defaut, recommande 2026), ``"dare_ties"`` (TIES
            avec drop+rescale stochastique), ``"linear"`` (moyenne ponderee
            simple, equivalent a l'ancien comportement).
        density: Fraction des parametres conservee par le trim TIES (defaut
            0.5 = 50%, valeur standard de la litterature).

    Raises:
        FileNotFoundError: Si l'un des checkpoints n'existe pas.
        ValueError: Si la liste de checkpoints est vide.
    """
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    if not fold_checkpoints:
        msg = "fold_checkpoints vide, impossible de fusionner"
        raise ValueError(msg)

    for ckpt in fold_checkpoints:
        if not ckpt.exists():
            msg = f"Checkpoint LoRA introuvable : {ckpt}"
            raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    first_adapter_config = fold_checkpoints[0] / "adapter_config.json"
    adapter_meta = json.loads(first_adapter_config.read_text(encoding="utf-8"))
    base_model_name = adapter_meta.get("base_model_name_or_path")
    if not base_model_name:
        msg = f"adapter_config.json de {fold_checkpoints[0]} sans base_model_name_or_path"
        raise ValueError(msg)

    logger.info(f"Chargement base model {base_model_name} pour fusion LoRA...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
    )

    # Charger le premier adapter sous un nom explicite pour permettre la
    # combinaison avec add_weighted_adapter sur K adapters nommes.
    peft_model = PeftModel.from_pretrained(
        base_model, fold_checkpoints[0], adapter_name="adapter_0"
    )

    if len(fold_checkpoints) == 1:
        # Un seul fold : pas de fusion, juste merge_and_unload du seul adapter
        logger.info("Un seul checkpoint : merge_and_unload direct sans fusion.")
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        _copy_tokenizer_artifacts(fold_checkpoints[0], output_dir)
        logger.info(f"Modele LoRA sauvegarde : {output_dir}")
        return

    # Charger les K-1 adapters restants sous des noms distincts pour pouvoir
    # invoquer add_weighted_adapter avec combination_type=ties.
    for i, ckpt in enumerate(fold_checkpoints[1:], 1):
        peft_model.load_adapter(str(ckpt), adapter_name=f"adapter_{i}")

    adapter_names = [f"adapter_{i}" for i in range(len(fold_checkpoints))]
    weights = [1.0 / len(fold_checkpoints)] * len(fold_checkpoints)

    try:
        logger.info(
            f"Fusion TIES ({len(fold_checkpoints)} adapters, "
            f"combination_type={combination_type}, density={density})..."
        )
        kwargs: dict[str, Any] = {
            "adapters": adapter_names,
            "weights": weights,
            "adapter_name": "merged",
            "combination_type": combination_type,
        }
        if combination_type in ("ties", "dare_ties", "dare_linear", "magnitude_prune"):
            kwargs["density"] = density
        peft_model.add_weighted_adapter(**kwargs)
        peft_model.set_adapter("merged")
        # Liberer les K adapters source pour reduire la VRAM avant merge_and_unload
        for name in adapter_names:
            peft_model.delete_adapter(name)
    except Exception as exc:
        logger.warning(
            f"add_weighted_adapter({combination_type}) echoue : {exc}. "
            "Fallback sur moyenne arithmetique naive des deltas A/B (legacy)."
        )
        # Repartir d'un peft_model propre charge sur le premier adapter
        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        peft_model = PeftModel.from_pretrained(base_model, fold_checkpoints[0])
        _average_lora_deltas(peft_model, fold_checkpoints)

    logger.info("Fusion merge_and_unload() en cours...")
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    # Copie du tokenizer et d'adapter_config.json depuis le premier checkpoint
    # pour garder un format "chargeable comme LoRA" (meme si les poids sont
    # deja fusionnes). Facilite le rechargement via `LoRAClassifier.load()`.
    _copy_tokenizer_artifacts(fold_checkpoints[0], output_dir)
    logger.info(f"Modele LoRA fusionne sauvegarde : {output_dir}")


def _average_lora_deltas(peft_model: Any, fold_checkpoints: list[Path]) -> None:
    """Moyenne les tenseurs LoRA A et B sur les K adapters.

    Modifie ``peft_model`` en place : les poids actuellement charges
    (depuis le premier checkpoint) sont remplaces par la moyenne
    arithmetique des K. Appele uniquement si ``len(fold_checkpoints) >= 2``.
    """
    from safetensors.torch import load_file

    n = len(fold_checkpoints)
    state_dict_avg: dict[str, torch.Tensor] = {}

    for ckpt in fold_checkpoints:
        adapter_file = ckpt / "adapter_model.safetensors"
        if not adapter_file.exists():
            msg = f"adapter_model.safetensors manquant dans {ckpt}"
            raise FileNotFoundError(msg)
        state = load_file(str(adapter_file))
        for key, tensor in state.items():
            if key not in state_dict_avg:
                state_dict_avg[key] = tensor.clone().to(torch.float32) / n
            else:
                state_dict_avg[key] = state_dict_avg[key] + tensor.to(torch.float32) / n

    # Injecte la moyenne dans le peft_model charge. On caste en bfloat16
    # pour rester coherent avec la precision d'entrainement.
    peft_state = peft_model.state_dict()
    for key, value in state_dict_avg.items():
        if key in peft_state:
            peft_state[key] = value.to(peft_state[key].dtype)
    peft_model.load_state_dict(peft_state, strict=False)
    logger.info(f"Deltas LoRA moyennes sur {n} checkpoints")


def _copy_tokenizer_artifacts(source: Path, destination: Path) -> None:
    """Copie le tokenizer et les fichiers meta d'un checkpoint LoRA vers le merged.

    Fichiers concernes : ``tokenizer.json``, ``tokenizer_config.json``,
    ``chat_template.jinja``, ``special_tokens_map.json`` si presents.
    """
    import shutil

    files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    )
    for name in files:
        src = source / name
        if src.exists():
            shutil.copy2(src, destination / name)


async def train_single(model_type: str) -> dict[str, float]:
    """Entraîne un seul modele avec tracking MLflow et mesure carbone.

    Args:
        model_type: Un de 'deberta', 'qwen2.5',
            'llama3.2', 'qwen3'.

    Returns:
        Metriques finales du modele entraine.
    """
    import mlflow

    train_texts, train_labels, test_texts, test_labels = load_golden_dataset()
    classifier, exp_config = _build_classifier_and_config(model_type)

    with tracked_experiment(exp_config):
        metrics = await classifier.train(train_texts, train_labels, test_texts, test_labels)
        classifier.save()
        log_model_artifact(classifier.config.output_dir)
        mlflow.log_metrics(metrics)

    logger.info(f"=== {model_type.upper()} termine : {metrics} ===")
    return metrics


async def train_all() -> dict[str, dict[str, float]]:
    """Entraîne les 3 modeles sequentiellement et retourne les metriques.

    Returns:
        Dictionnaire {model_type: metriques} pour les 3 modeles.
    """
    results = {}
    for model_type in VALID_MODELS:
        logger.info(f"\n{'=' * 60}\n  ENTRAINEMENT : {model_type.upper()}\n{'=' * 60}")
        results[model_type] = await train_single(model_type)

    logger.info("\n=== RESULTATS ===")
    for name, m in results.items():
        logger.info(f"  {name}: accuracy={m.get('accuracy', 'N/A')}, f1={m.get('f1', 'N/A')}")
    return results


async def benchmark_models() -> dict[str, dict[str, float]]:
    """Compare tous les modeles disponibles sur le jeu de test.

    Charge chaque modele sauvegarde, execute l'inference sur le test set,
    mesure accuracy/f1/precision/recall/latence, recupere les emissions CO2
    depuis MLflow, et log le tout dans un run MLflow de benchmark.

    Returns:
        Dictionnaire {nom_modele: metriques} avec le vainqueur identifie.
    """
    import mlflow
    from sklearn.metrics import classification_report

    from greentech.ai.mlops.tracking import configure_mlflow

    _, _, test_texts, test_labels = load_golden_dataset(oversample=False)
    logger.info(f"Benchmark sur {len(test_texts)} articles de test")

    # Registre des modeles a evaluer : (cle, label affichage, path, factory)
    model_registry = [
        (
            "deberta_legacy",
            "DeBERTa-v3-base (EN-only, legacy)",
            BASE_DIR / "models" / "deberta-legacy",
            "deberta",
        ),
        (
            "qwen2_5",
            "Qwen2.5-3B + LoRA (legacy)",
            BASE_DIR / "models" / "qwen2.5",
            "qwen2.5",
        ),
        (
            "llama3_2",
            "Llama 3.2 3B + LoRA (legacy)",
            BASE_DIR / "models" / "llama3.2",
            "llama3.2",
        ),
        (
            "qwen3",
            "Qwen3-4B + LoRA",
            BASE_DIR / "models" / "qwen3",
            "qwen3",
        ),
    ]

    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, list[int]] = {}

    for key, label, model_path, model_type in model_registry:
        if not model_path.exists():
            logger.warning(f"{label} introuvable ({model_path}), ignore")
            continue

        logger.info(f"Chargement de {label}...")
        classifier, _ = _build_classifier_and_config(model_type)
        classifier.load(model_path)

        preds = []
        latencies = []
        for text in test_texts:
            pred = await classifier.predict(text)
            preds.append(pred.label.value)
            latencies.append(pred.temps_ms)

        results[key] = {
            "accuracy": accuracy_score(test_labels, preds),
            "f1": f1_score(test_labels, preds, average="binary"),
            "precision": precision_score(test_labels, preds, average="binary"),
            "recall": recall_score(test_labels, preds, average="binary"),
            "latence_moyenne_ms": float(np.mean(latencies)),
            "latence_p95_ms": float(np.percentile(latencies, 95)),
        }
        predictions[key] = preds
        logger.info(
            f"{label} — F1={results[key]['f1']:.4f}, "
            f"Latence={results[key]['latence_moyenne_ms']:.0f}ms"
        )

        # Liberer la memoire GPU entre les modeles
        del classifier
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        logger.error("Aucun modele trouve pour le benchmark")
        return {}

    # Recuperer les emissions CO2 depuis MLflow
    configure_mlflow()
    for key in results:
        tag_type = key.replace("_", "-")
        runs = mlflow.search_runs(
            experiment_names=["greentech-classification"],
            filter_string=f"tags.type = '{tag_type}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs.empty and "metrics.emissions_carbone_g" in runs.columns:
            co2 = runs.iloc[0].get("metrics.emissions_carbone_g", 0.0)
            results[key]["emissions_co2_g"] = float(co2) if pd.notna(co2) else 0.0

    # Determiner le vainqueur (meilleur F1)
    vainqueur = max(results, key=lambda k: results[k]["f1"])
    raison = f"Meilleur F1 = {results[vainqueur]['f1']:.4f}"

    # Logger dans MLflow
    models_str = ", ".join(results.keys())
    benchmark_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run="benchmark-3-modeles",
        tags={"type": "benchmark", "vainqueur": vainqueur},
        params={
            "n_test_articles": len(test_texts),
            "modeles_evalues": models_str,
            "selection_raison": raison,
        },
        mesurer_carbone=False,
    )

    with tracked_experiment(benchmark_config):
        for key, metrics in results.items():
            mlflow.log_metrics({f"{key}_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("vainqueur_f1", results[vainqueur]["f1"])

    # Rapport final
    logger.info("")
    logger.info("=" * 80)
    logger.info("  BENCHMARK FINAL — CHAMPION vs CHALLENGERS")
    logger.info("=" * 80)

    headers = list(results.keys())
    logger.info(f"  {'Metrique':<22}" + "".join(f" {h:<22}" for h in headers))
    logger.info("-" * 80)
    all_keys = sorted({k for m in results.values() for k in m})
    for metric_key in all_keys:
        vals = "".join(f" {results[h].get(metric_key, 0):<22.4f}" for h in headers)
        logger.info(f"  {metric_key:<22}{vals}")
    logger.info("-" * 80)
    logger.info(f"  VAINQUEUR : {vainqueur.upper()}")
    logger.info(f"  Raison    : {raison}")
    logger.info("=" * 80)

    # Rapports de classification detailles
    for key, preds in predictions.items():
        logger.info(f"\nRapport {key} :")
        logger.info(
            "\n"
            + classification_report(
                test_labels,
                preds,
                target_names=["Non Green IT", "Green IT"],
            )
        )

    return results


if __name__ == "__main__":
    import asyncio
    import sys

    from greentech.utils.logger import setup_logging

    # Loki active : permet de suivre l'entrainement en direct via
    # Grafana Explore (filtres {module="training"}, {level="info"}).
    setup_logging(level="INFO", enable_loki=True)

    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == "benchmark":
        asyncio.run(benchmark_models())
    elif arg in VALID_MODELS:
        asyncio.run(train_single(arg))
    elif arg is None:
        asyncio.run(train_all())
    else:
        logger.error(f"Argument inconnu : {arg}")
        logger.info(
            f"Usage : python -m greentech.ai.models.training [{' | '.join(VALID_MODELS)} | benchmark]"
        )
