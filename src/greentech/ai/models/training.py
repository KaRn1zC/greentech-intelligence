"""Scripts d'entraînement pour les modèles Champion et Challenger.

Champion : DeBERTa-v3-base (fine-tuning classique via Transformers Trainer)
Challenger : Llama 3.2 3B (fine-tuning LoRA via PEFT)

Les deux modèles sont entraînés sur le Golden Dataset annoté et comparés
via MLflow (accuracy, F1, latence, émissions CO2).

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import time
from pathlib import Path

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
from greentech.config import BASE_DIR

# Labels pour la classification binaire
LABEL2ID = {"Non Green IT": 0, "Green IT": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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
                output_dir=BASE_DIR / "models" / "champion",
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

        # Charger tokenizer et modèle
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nom_modele)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )

        # Tokenizer les datasets
        train_dataset = self._prepare_dataset(train_texts, train_labels)
        val_dataset = self._prepare_dataset(val_texts, val_labels)

        # Configuration du Trainer
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            logging_dir=str(self.config.output_dir / "logs"),
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
        logger.info("Démarrage de l'entraînement Champion (DeBERTa-v3-base)...")
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
    """Classifieur Challenger basé sur Llama 3.2 3B avec LoRA.

    Fine-tuning efficient via PEFT/LoRA pour adapter un modèle génératif
    à la classification binaire. Nécessite le GPU AMD 7900 XTX via ROCm.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        lora_config: LoraConfig | None = None,
    ) -> None:
        if config is None:
            config = TrainingConfig(
                nom_modele="meta-llama/Llama-3.2-3B",
                output_dir=BASE_DIR / "models" / "challenger",
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

        # Charger tokenizer et modèle
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.nom_modele)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            torch_dtype=torch.float16,
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

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=self.config.seed,
            fp16=True,
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

        logger.info("Démarrage de l'entraînement Challenger (Llama 3.2 3B + LoRA)...")
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
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nom_modele,
            num_labels=2,
            torch_dtype=torch.float16,
        )
        self.model = PeftModel.from_pretrained(base_model, str(model_path))
        self.model.to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        logger.info(f"Challenger chargé depuis {model_path} sur {device}")


def load_golden_dataset(
    dataset_path: Path | None = None,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Charge et split le Golden Dataset annoté.

    Lit le CSV annoté, sépare en train/test (80/20), et retourne
    les textes et labels pour chaque split.

    Args:
        dataset_path: Chemin vers le CSV annoté (défaut: data/golden_dataset.csv).

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
    logger.info(f"Golden Dataset chargé : {len(df)} articles")

    # Vérifier la colonne label
    if "label_green_it" not in df.columns:
        msg = "Colonne 'label_green_it' manquante dans le dataset"
        raise ValueError(msg)

    # Filtrer les entrées non annotées
    df = df[df["label_green_it"].isin([0, 1])].copy()
    logger.info(
        f"Articles annotés : {len(df)} (Green: {df['label_green_it'].sum()}, Non: {(~df['label_green_it'].astype(bool)).sum()})"
    )

    # Split stratifié 80/20
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_green_it"]
    )

    return (
        train_df["contenu_extrait"].tolist(),
        train_df["label_green_it"].tolist(),
        test_df["contenu_extrait"].tolist(),
        test_df["label_green_it"].tolist(),
    )


async def train_and_compare() -> dict[str, dict[str, float]]:
    """Entraîne les deux modèles et les compare via MLflow.

    Lance l'entraînement du Champion (DeBERTa) et du Challenger (Llama+LoRA),
    log les résultats dans MLflow avec mesure CodeCarbon, et retourne
    les métriques comparatives.

    Returns:
        Dictionnaire {nom_modèle: métriques} pour les deux modèles.
    """
    train_texts, train_labels, test_texts, test_labels = load_golden_dataset()
    results = {}

    # --- Champion : DeBERTa-v3-base ---
    champion = ChampionClassifier()
    champion_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run="champion-deberta-v3-base",
        tags={"type": "champion", "modele": "deberta-v3-base"},
        params={
            "model": champion.config.nom_modele,
            "epochs": champion.config.epochs,
            "batch_size": champion.config.batch_size,
            "learning_rate": champion.config.learning_rate,
            "method": "full-finetuning",
        },
    )

    with tracked_experiment(champion_config):
        metrics = await champion.train(train_texts, train_labels, test_texts, test_labels)
        champion.save()
        log_model_artifact(champion.config.output_dir)
        import mlflow

        mlflow.log_metrics(metrics)
        results["champion_deberta"] = metrics

    # --- Challenger : Llama 3.2 3B + LoRA ---
    challenger = ChallengerClassifier()
    challenger_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run="challenger-llama-3.2-3b-lora",
        tags={"type": "challenger", "modele": "llama-3.2-3b", "method": "lora"},
        params={
            "model": challenger.config.nom_modele,
            "epochs": challenger.config.epochs,
            "batch_size": challenger.config.batch_size,
            "learning_rate": challenger.config.learning_rate,
            "method": "lora",
            "lora_r": challenger.lora_config.r,
            "lora_alpha": challenger.lora_config.alpha,
        },
    )

    with tracked_experiment(challenger_config):
        metrics = await challenger.train(train_texts, train_labels, test_texts, test_labels)
        challenger.save()
        log_model_artifact(challenger.config.output_dir)
        import mlflow

        mlflow.log_metrics(metrics)
        results["challenger_llama_lora"] = metrics

    # Comparaison
    logger.info("=== COMPARAISON DES MODÈLES ===")
    for name, m in results.items():
        logger.info(f"  {name}: accuracy={m.get('accuracy', 'N/A')}, f1={m.get('f1', 'N/A')}")

    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(train_and_compare())
