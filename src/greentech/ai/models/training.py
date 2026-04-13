"""Scripts d'entraînement pour les modèles Champion et Challengers.

Champion  : DeBERTa-v3-base (fine-tuning classique via Transformers Trainer)
Challenger 1 : Qwen2.5-3B (fine-tuning LoRA via PEFT, 3085M params)
Challenger 2 : Llama 3.2 3B (fine-tuning LoRA via PEFT, 3213M params, gated)

Les trois modèles sont entraînés sur le Golden Dataset annoté et comparés
via MLflow (accuracy, F1, latence, émissions CO2).

Gere le desequilibre extreme du dataset (22 Green IT / 5786 Non Green IT)
via oversampling de la minorite a ~20%.

Usage:
    uv run python -m greentech.ai.models.training                  # Tous les modeles
    uv run python -m greentech.ai.models.training champion-deberta # Champion seul
    uv run python -m greentech.ai.models.training challenger-qwen  # Challenger Qwen seul
    uv run python -m greentech.ai.models.training challenger-llama # Challenger Llama seul
    uv run python -m greentech.ai.models.training benchmark        # Benchmark comparatif

"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
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
VALID_MODELS = ("champion-deberta", "challenger-qwen", "challenger-llama")


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Calcule les métriques de classification pour le Trainer.

    Args:
        eval_pred: Tuple (logits, labels) fourni par le Trainer.

    Returns:
        Dictionnaire de métriques (accuracy, f1, precision, recall).
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
    }


class ChampionClassifier(BaseClassifier):
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
                output_dir=BASE_DIR / "models" / "champion-deberta",
                epochs=5,
                batch_size=16,
                learning_rate=3e-5,
            )
        super().__init__(config)

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

        # Charger tokenizer et modèle (forcer fp32 car transformers 5.x charge
        # DeBERTa en fp16 par defaut, ce qui cause NaN sans loss scaling)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nom_modele)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            dtype=torch.float32,
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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Entraînement
        logger.info("Demarrage de l'entrainement Champion (DeBERTa-v3-base)...")
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
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return PredictionResult(
            label=LabelGreenIT(predicted_class),
            score_confiance=confidence,
            temps_ms=elapsed_ms,
            modele=self.config.nom_modele,
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


class ChallengerClassifier(BaseClassifier):
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
                output_dir=BASE_DIR / "models" / "challenger-qwen",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            )
        super().__init__(config)
        self.lora_config = lora_config or LoraConfig()

    async def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> dict[str, float]:
        """Entraîne Llama 3.2 3B avec LoRA sur le dataset annoté.

        Args:
            train_texts: Textes d'entraînement.
            train_labels: Labels (0/1).
            val_texts: Textes de validation.
            val_labels: Labels de validation.

        Returns:
            Métriques finales (accuracy, f1, precision, recall).
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

        # Appliquer LoRA
        peft_config = PeftLoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules or ["q_proj", "v_proj"],
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

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            bf16=True,
            gradient_accumulation_steps=4,
            logging_steps=10,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Demarrage de l'entrainement Challenger (Qwen2.5-3B + LoRA)...")
        train_result = trainer.train()

        eval_result = trainer.evaluate()
        logger.info(f"Challenger — Résultats : {eval_result}")

        return {
            "train_loss": train_result.training_loss,
            **{k.replace("eval_", ""): v for k, v in eval_result.items()},
        }

    def _prepare_dataset(self, texts: list[str], labels: list[int]) -> Dataset:
        """Tokenize les textes pour Llama.

        Args:
            texts: Liste de textes bruts.
            labels: Liste de labels correspondants.

        Returns:
            Dataset tokenizé.
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
        """Classifie un texte avec Llama + LoRA.

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
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        return PredictionResult(
            label=LabelGreenIT(predicted_class),
            score_confiance=confidence,
            temps_ms=elapsed_ms,
            modele=f"{self.config.nom_modele}+LoRA",
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
    df["text"] = df["titre"].fillna("") + "\n\n" + df["contenu_extrait"].fillna("")

    n_green = int(df["label_green_it"].sum())
    logger.info(
        f"Dataset complet charge : {len(df)} articles (Green: {n_green}, Non: {len(df) - n_green})"
    )

    return df["text"].tolist(), df["label_green_it"].tolist()


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

    # Combiner titre + contenu pour des features plus riches
    df["text"] = df["titre"].fillna("") + "\n\n" + df["contenu_extrait"].fillna("")

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
        model_type: Un de 'champion-deberta', 'challenger-qwen', 'challenger-llama'.

    Returns:
        Tuple (classifieur, config MLflow).

    Raises:
        ValueError: Si model_type n'est pas reconnu.
    """
    if model_type not in VALID_MODELS:
        msg = f"Type inconnu : {model_type}. Valides : {', '.join(VALID_MODELS)}"
        raise ValueError(msg)

    if model_type == "champion-deberta":
        classifier = ChampionClassifier()
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="champion-deberta-v3-base",
            tags={"type": "champion-deberta", "modele": "deberta-v3-base"},
            params={
                "model": classifier.config.nom_modele,
                "epochs": classifier.config.epochs,
                "batch_size": classifier.config.batch_size,
                "learning_rate": classifier.config.learning_rate,
                "method": "full-finetuning",
            },
        )
    elif model_type == "challenger-qwen":
        classifier = ChallengerClassifier(
            config=TrainingConfig(
                nom_modele="Qwen/Qwen2.5-3B",
                output_dir=BASE_DIR / "models" / "challenger-qwen",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            ),
        )
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="challenger-qwen2.5-3b-lora",
            tags={"type": "challenger-qwen", "modele": "qwen2.5-3b", "method": "lora"},
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
    else:  # challenger-llama
        classifier = ChallengerClassifier(
            config=TrainingConfig(
                nom_modele="meta-llama/Llama-3.2-3B",
                output_dir=BASE_DIR / "models" / "challenger-llama",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                max_length=512,
            ),
        )
        exp_config = ExperimentConfig(
            nom_experience="greentech-classification",
            nom_run="challenger-llama-3.2-3b-lora",
            tags={"type": "challenger-llama", "modele": "llama-3.2-3b", "method": "lora"},
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

    return classifier, exp_config


async def train_challenger_with_cv(
    model_type: str = "challenger-llama",
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
        3. Entrainement d'un nouveau ChallengerClassifier.
        4. Evaluation sur le test set du fold (sans oversampling).
        5. Stockage des predictions et des metriques par fold.

    A l'issue des K folds, un modele final est optionnellement ré-entraine sur
    l'integralite des donnees (c'est celui qui sera servi en production). Les
    metriques reportees sont celles du K-fold, car seules elles donnent une
    estimation honnete de la performance sur donnees non vues.

    Args:
        model_type: "challenger-llama" ou "challenger-qwen".
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

    if model_type not in ("challenger-llama", "challenger-qwen"):
        msg = f"K-fold disponible uniquement pour les challengers, pas pour {model_type}"
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

            fold_metrics_dict = _compute_fold_metrics(
                test_labels_fold, fold_preds, fold_latencies
            )
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
                metrics={
                    k: v
                    for k, v in fold_metrics_dict.items()
                    if isinstance(v, int | float)
                },
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


async def train_single(model_type: str) -> dict[str, float]:
    """Entraîne un seul modele avec tracking MLflow et mesure carbone.

    Args:
        model_type: Un de 'champion-deberta', 'challenger-qwen', 'challenger-llama'.

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
            "champion_deberta",
            "Champion (DeBERTa)",
            BASE_DIR / "models" / "champion-deberta",
            "champion-deberta",
        ),
        (
            "challenger_qwen",
            "Challenger Qwen+LoRA",
            BASE_DIR / "models" / "challenger-qwen",
            "challenger-qwen",
        ),
        (
            "challenger_llama",
            "Challenger Llama+LoRA",
            BASE_DIR / "models" / "challenger-llama",
            "challenger-llama",
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

    # Loki active : permet de suivre l'entrainement Llama en direct via
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
