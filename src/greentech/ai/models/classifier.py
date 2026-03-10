"""Classifieur Green IT — entraînement et inférence.

Fournit deux stratégies de classification :
- Champion : DeBERTa-v3-base (fine-tuning classique via Trainer)
- Challenger : Llama 3.2 3B (fine-tuning LoRA/PEFT)

Les deux modèles sont comparés via MLflow pour sélectionner le meilleur.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class LabelGreenIT(int, Enum):
    """Labels de classification Green IT."""

    NON_GREEN = 0
    GREEN = 1


@dataclass(frozen=True)
class PredictionResult:
    """Résultat d'une prédiction de classification.

    Attributes:
        label: Label prédit (GREEN ou NON_GREEN).
        score_confiance: Probabilité associée à la prédiction (0.0-1.0).
        temps_ms: Temps d'inférence en millisecondes.
        modele: Nom du modèle utilisé.
    """

    label: LabelGreenIT
    score_confiance: float
    temps_ms: int
    modele: str

    @property
    def est_green_it(self) -> bool:
        """Retourne True si l'article est classifié Green IT."""
        return self.label == LabelGreenIT.GREEN


@dataclass
class TrainingConfig:
    """Configuration d'entraînement commune aux deux modèles.

    Attributes:
        nom_modele: Identifiant du modèle Hugging Face.
        output_dir: Dossier de sortie pour les checkpoints.
        epochs: Nombre d'époques d'entraînement.
        batch_size: Taille des batchs.
        learning_rate: Taux d'apprentissage.
        weight_decay: Régularisation L2.
        warmup_ratio: Ratio de warmup pour le scheduler.
        max_length: Longueur maximale des séquences (tokens).
        seed: Graine pour la reproductibilité.
        device: Device cible ('cuda' pour ROCm, 'cpu' en fallback).
    """

    nom_modele: str
    output_dir: Path = Path("models/checkpoints")
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    seed: int = 42
    device: str = "cuda"


@dataclass
class LoraConfig:
    """Configuration LoRA pour le fine-tuning efficient (Challenger).

    Attributes:
        r: Rang de la décomposition LoRA.
        alpha: Facteur d'échelle alpha.
        dropout: Dropout sur les couches LoRA.
        target_modules: Modules cibles pour l'adaptation.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] | None = None


class BaseClassifier(ABC):
    """Interface commune pour les classifieurs Green IT.

    Chaque implémentation (Champion/Challenger) doit fournir
    les méthodes d'entraînement, de prédiction et de sauvegarde.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        logger.info(f"Classifieur initialisé : {config.nom_modele} (device={config.device})")

    @abstractmethod
    async def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
    ) -> dict[str, float]:
        """Entraîne le modèle sur les données fournies.

        Args:
            train_texts: Textes d'entraînement.
            train_labels: Labels d'entraînement (0/1).
            val_texts: Textes de validation.
            val_labels: Labels de validation.

        Returns:
            Métriques d'entraînement (loss, accuracy, f1, etc.).
        """

    @abstractmethod
    async def predict(self, text: str) -> PredictionResult:
        """Prédit le label Green IT pour un texte donné.

        Args:
            text: Texte de l'article à classifier.

        Returns:
            Résultat de classification avec score de confiance.
        """

    @abstractmethod
    def save(self, output_dir: Path | None = None) -> Path:
        """Sauvegarde le modèle entraîné.

        Args:
            output_dir: Dossier de destination (défaut: config.output_dir).

        Returns:
            Chemin du modèle sauvegardé.
        """

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Charge un modèle pré-entraîné.

        Args:
            model_path: Chemin vers le modèle sauvegardé.
        """

    def detect_device(self) -> str:
        """Détecte le meilleur device disponible (ROCm > DirectML > CPU).

        Returns:
            Identifiant du device ('cuda', 'privateuseone', ou 'cpu').
        """
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU détecté via ROCm/CUDA : {device_name}")
                return "cuda"
        except ImportError:
            pass

        try:
            import torch_directml

            device = torch_directml.device()
            logger.info(f"GPU détecté via DirectML : {device}")
            return str(device)
        except ImportError:
            pass

        logger.warning("Aucun GPU détecté, utilisation du CPU")
        return "cpu"
