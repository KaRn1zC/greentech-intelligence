"""Evaluation baseline d'un modele de base SANS fine-tuning.

Ce module mesure les metriques de reference d'un LLM pre-entraine sur le
Golden Dataset annote, avant toute adaptation specifique au domaine Green IT.
Le but est double :

1. **Etablir une reference permanente** avant chaque cycle de fine-tuning pour
   pouvoir quantifier le gain apporte par le LoRA K-fold.
2. **Detecter les regressions** : si apres entrainement le modele n'arrive pas
   a battre significativement la baseline, il y a un probleme (mauvaise
   convergence, data leakage, hyperparametres inadaptes, dataset pollue).

Strategie d'evaluation
----------------------

Le modele de base est charge avec une tete de classification fraichement
initialisee (`AutoModelForSequenceClassification`, num_labels=2). Les poids
de la tete de classification sont donc aleatoires : on mesure ici la
"chance + biais" du modele, pas sa comprehension reelle des articles
(c'est le role du fine-tuning LoRA).

Aucun prompt engineering n'est applique : on veut une reference brute et
reproductible, pas une optimisation par essais successifs. Le chargement
se fait en BF16 quand le GPU supporte, sinon en FP32 sur CPU (lent mais
fidele).

Portee d'evaluation
-------------------

L'integralite du Golden Dataset est utilisee (pas de split test), car :

- Le modele n'a pas ete entraine : aucun risque de data leakage.
- Une evaluation sur l'integralite donne une metrique beaucoup plus stable
  que sur un split 20% (surtout avec seulement ~22 Green IT dans le corpus).
- La baseline sert de reference permanente pour *tous* les K folds futurs.

Hardware cible
--------------

- RX 7900 XTX 24 Go via ROCm 7.2 : ~5-10 s/article en BF16 pour un 4B, soit
  ~8-15 min pour 5800 articles. Acceptable pour une execution occasionnelle.
- CPU : fallback automatique avec avertissement (peut prendre plusieurs
  heures sur un gros modele).

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from greentech.ai.models.classifier import LabelGreenIT
from greentech.config import get_settings

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class BaselineResult:
    """Resultat de l'evaluation baseline d'un modele.

    Attributes:
        model_name: Identifiant du modele evalue (ex. ``Qwen/Qwen3.5-4B``).
        metrics: Dictionnaire complet des metriques
            (mcc, f1, precision, recall, accuracy, matrice de confusion...).
        predictions: Liste des labels predits, dans l'ordre des articles.
        latencies_ms: Liste des latences d'inference en millisecondes.
        n_articles: Nombre d'articles evalues.
        duration_seconds: Duree totale de l'evaluation.
    """

    model_name: str
    metrics: dict[str, float | int]
    predictions: list[int]
    latencies_ms: list[float]
    n_articles: int
    duration_seconds: float


class BaselineEvaluator:
    """Evaluateur baseline generique pour modeles causaux pre-entraines.

    Charge un modele Hugging Face en mode classification sequence (avec tete
    de classification initialisee aleatoirement), puis itere sur l'integralite
    du Golden Dataset pour mesurer les metriques de reference.

    L'instance est jetable : un nouvel evaluateur par evaluation pour eviter
    les fuites de memoire GPU entre runs.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_length: int | None = None,
    ) -> None:
        """Initialise l'evaluateur.

        Args:
            model_name: Identifiant HF du modele a evaluer. Si ``None``,
                utilise ``settings.huggingface_model_baseline``.
            max_length: Longueur max des sequences tokenizees. Si ``None``,
                utilise ``settings.trainer_max_length`` pour rester aligne
                avec le setup d'entrainement.
        """
        settings = get_settings()
        self.model_name = model_name or settings.huggingface_model_baseline
        self.max_length = max_length or settings.trainer_max_length
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._device: str | None = None

    @staticmethod
    def _pick_device() -> str:
        """Detecte le meilleur device disponible (ROCm/CUDA > DirectML > CPU)."""
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detecte : {device_name}")
                return "cuda"
        except ImportError:
            pass

        try:
            import torch_directml

            device = torch_directml.device()
            logger.info(f"DirectML detecte : {device}")
            return str(device)
        except ImportError:
            pass

        logger.warning(
            "Aucun GPU detecte : evaluation baseline sur CPU (tres lent sur un modele 4B)"
        )
        return "cpu"

    def load(self) -> None:
        """Charge le modele et le tokenizer depuis Hugging Face.

        Le modele est instancie avec une tete de classification num_labels=2
        dont les poids sont aleatoires : c'est volontaire, on mesure la
        performance *sans* fine-tuning.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        settings = get_settings()
        hf_token = settings.huggingface_token or None

        self._device = self._pick_device()
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        logger.info(f"Chargement de {self.model_name} (device={self._device}, dtype={dtype})")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token,
            trust_remote_code=False,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            label2id={"Non Green IT": 0, "Green IT": 1},
            id2label={0: "Non Green IT", 1: "Green IT"},
            dtype=dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
            trust_remote_code=False,
        )
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.to(torch.device(self._device))
        self._model.eval()

        logger.info(f"Modele {self.model_name} charge et pret a l'inference")

    def predict(self, text: str) -> tuple[int, float]:
        """Predit le label d'un article unique.

        Args:
            text: Contenu de l'article (titre + corps) a classifier.

        Returns:
            Tuple ``(label, latence_ms)`` ou label est 0 (Non Green IT) ou
            1 (Green IT) et latence la duree en millisecondes.

        Raises:
            RuntimeError: Si ``load()`` n'a pas ete appele au prealable.
        """
        import torch

        if self._model is None or self._tokenizer is None or self._device is None:
            msg = "Modele non charge. Appelez load() avant predict()."
            raise RuntimeError(msg)

        start = time.perf_counter()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        inputs = {k: v.to(torch.device(self._device)) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        label = int(torch.softmax(outputs.logits, dim=-1).argmax(dim=-1).item())
        latency_ms = (time.perf_counter() - start) * 1000
        return label, latency_ms

    def evaluate(
        self,
        texts: list[str],
        labels: list[int],
        *,
        log_every: int = 500,
    ) -> BaselineResult:
        """Evalue le modele sur l'ensemble des articles fournis.

        Args:
            texts: Liste des textes a classifier.
            labels: Labels reels (0 = Non Green IT, 1 = Green IT).
            log_every: Frequence d'affichage de l'avancement (defaut : tous
                les 500 articles).

        Returns:
            Resultat complet avec metriques, predictions et latences.

        Raises:
            RuntimeError: Si ``load()`` n'a pas ete appele.
            ValueError: Si ``texts`` et ``labels`` ont des tailles differentes.
        """
        if self._model is None:
            self.load()

        if len(texts) != len(labels):
            msg = f"Tailles incompatibles : {len(texts)} textes, {len(labels)} labels"
            raise ValueError(msg)

        logger.info(
            f"Evaluation baseline : {len(texts)} articles, "
            f"modele={self.model_name}, max_length={self.max_length}"
        )

        start = time.perf_counter()
        predictions: list[int] = []
        latencies_ms: list[float] = []

        for idx, text in enumerate(texts, 1):
            if idx % log_every == 0:
                logger.info(f"  Avancement : {idx}/{len(texts)} articles")
            pred, latency = self.predict(text)
            predictions.append(pred)
            latencies_ms.append(latency)

        duration = time.perf_counter() - start

        metrics = compute_classification_metrics(labels, predictions, latencies_ms)
        logger.info(
            f"Baseline terminee en {duration:.1f}s : "
            f"MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )

        return BaselineResult(
            model_name=self.model_name,
            metrics=metrics,
            predictions=predictions,
            latencies_ms=latencies_ms,
            n_articles=len(texts),
            duration_seconds=duration,
        )

    def unload(self) -> None:
        """Libere la VRAM/RAM occupee par le modele."""
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(f"Echec liberation memoire : {exc}")
        logger.info(f"Modele {self.model_name} decharge")


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    latencies_ms: list[float] | None = None,
) -> dict[str, float | int]:
    """Calcule un jeu complet de metriques pour comparer les versions d'un modele.

    Produit les metriques standards pour un probleme de classification binaire
    fortement desequilibre : MCC (robuste au desequilibre), F1, accuracy,
    balanced accuracy, precision, recall, specificite, matrice de confusion
    complete (TP/TN/FP/FN), distribution reelle vs predite, et latence moyenne.

    Args:
        y_true: Labels reels (0 = Non Green IT, 1 = Green IT).
        y_pred: Labels predits par le modele.
        latencies_ms: Latences d'inference par article (optionnel).

    Returns:
        Dictionnaire pret a etre serialise en JSON ou compare entre versions.
    """
    import numpy as np
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

    nb_reels_green = int(sum(1 for lbl in y_true if lbl == LabelGreenIT.GREEN.value))
    nb_reels_non_green = int(sum(1 for lbl in y_true if lbl == LabelGreenIT.NON_GREEN.value))
    nb_pred_green = int(sum(1 for p in y_pred if p == LabelGreenIT.GREEN.value))
    nb_pred_non_green = int(sum(1 for p in y_pred if p == LabelGreenIT.NON_GREEN.value))

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
        "nb_reels_green_it": nb_reels_green,
        "nb_reels_non_green_it": nb_reels_non_green,
        "nb_predictions_green_it": nb_pred_green,
        "nb_predictions_non_green_it": nb_pred_non_green,
        "total_echantillons": len(y_true),
    }

    if latencies_ms:
        metrics["latence_moyenne_ms"] = float(np.mean(latencies_ms))
        metrics["latence_p95_ms"] = float(np.percentile(latencies_ms, 95))

    return metrics


def evaluate_baseline(
    model_name: str | None = None,
    dataset_path: Path | None = None,
    *,
    max_length: int | None = None,
) -> BaselineResult:
    """Evalue la baseline complete d'un modele sur le Golden Dataset.

    Point d'entree haut niveau qui orchestre le chargement du dataset,
    l'initialisation de l'evaluateur et le nettoyage memoire en fin
    d'execution.

    Args:
        model_name: Identifiant HF du modele a evaluer. Defaut : valeur de
            ``settings.huggingface_model_baseline``.
        dataset_path: Chemin du CSV golden dataset. Defaut :
            ``data/golden_dataset.csv``.
        max_length: Longueur max des sequences. Defaut : valeur de
            ``settings.trainer_max_length``.

    Returns:
        Resultat complet de la baseline (metriques, predictions, latences).

    Raises:
        FileNotFoundError: Si le golden dataset n'existe pas.
    """
    from greentech.ai.models.training import load_full_dataset

    texts, labels = load_full_dataset(dataset_path)
    evaluator = BaselineEvaluator(model_name=model_name, max_length=max_length)
    try:
        evaluator.load()
        result = evaluator.evaluate(texts, labels)
    finally:
        evaluator.unload()

    return result
