"""Calibration post-training : temperature scaling + threshold tuning.

Deux techniques complementaires appliquees APRES l'entrainement sur le
validation set de chaque fold, pour corriger la sur-confiance typique des
modeles fine-tunes sur dataset desequilibre et pour maximiser la metrique
principale (MCC) en ajustant le seuil de decision.

Temperature Scaling (Guo et al. 2017, arXiv:1706.04599)
-------------------------------------------------------
Parametre scalaire unique ``T`` qui divise les logits avant le softmax.
Optimise par LBFGS sur le val set pour minimiser la NLL. Reference 2024-2025
pour BERT/DeBERTa fine-tunes (cf. "Calibrating Language Models with Adaptive
Temperature Scaling", EMNLP 2024). Platt scaling et isotonic regression
ont ete ecartes : ils sur-apprennent avec moins de 1 000 positifs par fold.

Threshold Tuning
----------------
Scan de seuils sur ``[0.05, 0.95]`` par pas de 0.01, retenir celui qui
maximise la metrique cible (MCC par defaut). Gain typique +0.02-0.05 MCC
sans rien changer au modele lui-meme.

Les deux sorties (``T`` optimal et seuil optimal) sont persistees dans le
dossier du modele sous forme de JSON et consommees par ``inference.py``
au chargement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score, matthews_corrcoef
from torch import nn

Metric = Literal["mcc", "f1"]


@dataclass(frozen=True)
class TemperatureResult:
    """Resultat d'une calibration par temperature scaling.

    Attributes:
        temperature: Valeur scalaire T optimisee (>0). Un T>1 reduit la
            confiance, un T<1 l'augmente.
        nll_before: Negative Log-Likelihood sur le val set avant calibration.
        nll_after: NLL apres application de T. Doit etre <= nll_before.
        n_iterations: Nombre d'iterations LBFGS effectuees avant convergence.
    """

    temperature: float
    nll_before: float
    nll_after: float
    n_iterations: int


@dataclass(frozen=True)
class ThresholdResult:
    """Resultat d'un scan de seuils de decision.

    Attributes:
        threshold: Seuil optimal dans [0.05, 0.95].
        metric: Nom de la metrique optimisee ("mcc" ou "f1").
        value: Valeur de la metrique au seuil optimal.
        grid_values: Valeurs de la metrique pour chaque seuil scanne
            (utile pour visualiser la sensibilite, optionnel).
    """

    threshold: float
    metric: Metric
    value: float
    grid_values: list[tuple[float, float]] = field(default_factory=list)


class TemperatureScaler(nn.Module):
    """Calibrateur par temperature scaling optimisable via LBFGS.

    Utilisation typique apres chaque fold :

        scaler = TemperatureScaler()
        result = scaler.fit(val_logits, val_labels)
        calibrated_probs = scaler.transform(new_logits)

    Le module est un ``nn.Module`` pour pouvoir etre sauvegarde/charge
    comme n'importe quel sous-modele torch si besoin, mais en pratique on
    persiste juste la valeur scalaire ``T`` dans un JSON.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialisation a 1.0 (pas de calibration) en log-space pour
        # garantir que T reste strictement positif sans contrainte manuelle.
        # Un T effectif de 1.0 correspond a log(T)=0.
        self._log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        """Retourne la valeur scalaire T (toujours > 0)."""
        return float(torch.exp(self._log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Divise les logits par la temperature apprise."""
        return logits / torch.exp(self._log_temperature)

    def transform(self, logits: np.ndarray | torch.Tensor) -> np.ndarray:
        """Applique T puis softmax pour obtenir des probabilites calibrees.

        Args:
            logits: Logits bruts du modele, shape ``(n_samples, n_classes)``.

        Returns:
            Probabilites calibrees, shape ``(n_samples, n_classes)``.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        with torch.no_grad():
            scaled = self.forward(logits)
            probs = torch.softmax(scaled, dim=-1)
        return probs.cpu().numpy()

    def fit(
        self,
        val_logits: np.ndarray | torch.Tensor,
        val_labels: np.ndarray | torch.Tensor,
        *,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> TemperatureResult:
        """Optimise T pour minimiser la NLL sur le validation set.

        Args:
            val_logits: Logits du modele sur le val set,
                shape ``(n_samples, n_classes)``.
            val_labels: Labels ground truth, shape ``(n_samples,)``.
            max_iter: Iterations max de LBFGS (converge generalement en < 30).
            lr: Learning rate initial pour LBFGS.

        Returns:
            Details de la calibration (T optimal, NLL avant/apres, iterations).
        """
        if isinstance(val_logits, np.ndarray):
            val_logits = torch.from_numpy(val_logits).float()
        if isinstance(val_labels, np.ndarray):
            val_labels = torch.from_numpy(val_labels).long()

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            nll_before = float(criterion(val_logits, val_labels).item())

        optimizer = torch.optim.LBFGS([self._log_temperature], lr=lr, max_iter=max_iter)

        iteration_counter = {"n": 0}

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            iteration_counter["n"] += 1
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            nll_after = float(criterion(self.forward(val_logits), val_labels).item())

        t_value = self.temperature
        logger.info(
            f"Temperature scaling : T={t_value:.4f}, "
            f"NLL {nll_before:.4f} -> {nll_after:.4f} "
            f"(gain {nll_before - nll_after:+.4f}, {iteration_counter['n']} iterations)"
        )
        return TemperatureResult(
            temperature=t_value,
            nll_before=nll_before,
            nll_after=nll_after,
            n_iterations=iteration_counter["n"],
        )


def _score_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    metric: Metric,
) -> float:
    """Calcule la metrique cible pour un seuil donne."""
    y_pred = (y_proba >= threshold).astype(int)
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    return float(f1_score(y_true, y_pred, zero_division=0))


def find_optimal_threshold(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    *,
    metric: Metric = "mcc",
    grid_start: float = 0.05,
    grid_stop: float = 0.95,
    grid_step: float = 0.01,
) -> ThresholdResult:
    """Scanne la grille de seuils et retourne celui qui maximise ``metric``.

    Le seuil par defaut d'un classifieur binaire est 0.5, mais pour un
    dataset desequilibre (1:10.5) ce seuil est rarement optimal. Scanner
    ``[0.05, 0.95]`` permet d'exploiter tout l'espace de trade-off entre
    precision et recall.

    Args:
        y_true: Labels ground truth (0/1).
        y_proba: Probabilites predites pour la classe positive (index 1).
        metric: Metrique a maximiser, "mcc" ou "f1".
        grid_start: Seuil minimum scanne (inclus).
        grid_stop: Seuil maximum scanne (inclus).
        grid_step: Pas de la grille.

    Returns:
        Seuil optimal et valeur de la metrique associee.

    Raises:
        ValueError: Si ``y_true`` et ``y_proba`` n'ont pas la meme taille
            ou si la grille est invalide.
    """
    y_true_arr = np.asarray(y_true).flatten()
    y_proba_arr = np.asarray(y_proba).flatten()

    if y_true_arr.shape != y_proba_arr.shape:
        msg = (
            f"y_true et y_proba doivent avoir la meme shape "
            f"(got {y_true_arr.shape} vs {y_proba_arr.shape})"
        )
        raise ValueError(msg)

    if grid_stop <= grid_start or grid_step <= 0:
        msg = (
            f"Grille de seuils invalide : [{grid_start}, {grid_stop}] "
            f"pas {grid_step}"
        )
        raise ValueError(msg)

    thresholds = np.arange(grid_start, grid_stop + grid_step / 2, grid_step)
    scores = [
        (float(t), _score_threshold(y_true_arr, y_proba_arr, float(t), metric))
        for t in thresholds
    ]

    best_idx = int(np.argmax([s for _, s in scores]))
    best_threshold, best_value = scores[best_idx]

    logger.info(
        f"Threshold tuning ({metric}) : seuil optimal = {best_threshold:.2f}, "
        f"{metric.upper()}={best_value:.4f} "
        f"(scan sur {len(scores)} valeurs)"
    )

    return ThresholdResult(
        threshold=best_threshold,
        metric=metric,
        value=best_value,
        grid_values=scores,
    )


def save_calibration(
    output_dir: Path,
    temperature: TemperatureResult | None = None,
    threshold: ThresholdResult | None = None,
) -> None:
    """Persiste les resultats de calibration dans le dossier du modele.

    Cree deux fichiers independants pour pouvoir les recharger
    separement cote inference (un modele peut avoir un T sans seuil
    optimise, ou inversement).

    Args:
        output_dir: Dossier du modele (ex: ``models/qwen3/``).
        temperature: Resultat du temperature scaling, si applicable.
        threshold: Resultat du threshold tuning, si applicable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if temperature is not None:
        temp_path = output_dir / "temperature.json"
        payload = {
            "temperature": temperature.temperature,
            "nll_before": temperature.nll_before,
            "nll_after": temperature.nll_after,
            "n_iterations": temperature.n_iterations,
        }
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Temperature calibration sauvegardee : {temp_path}")

    if threshold is not None:
        thresh_path = output_dir / "optimal_threshold.json"
        payload_thresh = {
            "threshold": threshold.threshold,
            "metric": threshold.metric,
            "value": threshold.value,
        }
        thresh_path.write_text(json.dumps(payload_thresh, indent=2), encoding="utf-8")
        logger.info(f"Threshold optimal sauvegarde : {thresh_path}")


def load_calibration(
    model_dir: Path,
) -> tuple[float | None, float | None]:
    """Charge la temperature et le seuil optimal si disponibles.

    Args:
        model_dir: Dossier du modele.

    Returns:
        Tuple ``(temperature, threshold)`` avec ``None`` pour chaque valeur
        si le JSON correspondant n'existe pas. Permet un fallback propre
        a ``T=1.0`` et seuil 0.5 dans le code appelant.
    """
    temp_path = model_dir / "temperature.json"
    thresh_path = model_dir / "optimal_threshold.json"

    temperature: float | None = None
    threshold: float | None = None

    if temp_path.exists():
        data = json.loads(temp_path.read_text(encoding="utf-8"))
        temperature = float(data["temperature"])

    if thresh_path.exists():
        data = json.loads(thresh_path.read_text(encoding="utf-8"))
        threshold = float(data["threshold"])

    return temperature, threshold


def apply_calibration(
    logits: np.ndarray,
    temperature: float | None = None,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique la calibration complete (T puis seuil) sur des logits bruts.

    Fonction utilitaire pour l'inference : prend des logits de sortie et
    applique dans l'ordre (1) le temperature scaling s'il est disponible,
    (2) le seuil optimal pour la decision binaire.

    Args:
        logits: Logits bruts du modele, shape ``(n_samples, 2)`` pour du
            binaire, ou ``(n_samples,)`` si on ne transmet que le logit
            de la classe positive.
        temperature: Valeur T a appliquer. ``None`` ou 1.0 = pas de calibration.
        threshold: Seuil binaire. ``None`` = 0.5 par defaut.

    Returns:
        Tuple ``(probabilites_positives, predictions)`` ou
        ``probabilites_positives`` est la probabilite de la classe 1
        (shape ``(n_samples,)``) et ``predictions`` les labels binaires.
    """
    logits_arr = np.asarray(logits, dtype=np.float32)

    # Appliquer T (division des logits avant softmax)
    if temperature is not None and temperature != 1.0:
        logits_arr = logits_arr / temperature

    # Calculer les probabilites de la classe positive
    if logits_arr.ndim == 2 and logits_arr.shape[1] == 2:
        # Shape (n, 2) -> softmax puis recuperer la proba de classe 1
        exp_logits = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        pos_probs = probs[:, 1]
    else:
        # Shape (n,) -> sigmoid (suppose que c'est le logit de la classe positive)
        pos_probs = 1.0 / (1.0 + np.exp(-logits_arr))

    effective_threshold = threshold if threshold is not None else 0.5
    predictions = (pos_probs >= effective_threshold).astype(np.int64)

    return pos_probs, predictions
