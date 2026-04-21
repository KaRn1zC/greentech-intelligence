"""Tests unitaires de la calibration post-training (B3.4).

Couvre :

1. ``TemperatureScaler`` : initialisation a T=1, correction sur logits
   sur-confiants (T > 1 attendu), preservation des probas deja calibrees
   (T proche de 1), invariance de l'argmax (T ne change que la confiance,
   pas la decision).
2. ``find_optimal_threshold`` : argmax MCC sur une grille connue, gestion
   des inputs incoherents (shapes, grilles invalides), resultats stables
   sur un dataset synthetique.
3. ``save_calibration`` + ``load_calibration`` : round-trip des JSON,
   absence silencieuse des fichiers (fallback None), resilience aux
   appels partiels (temperature seule ou threshold seul).

Tests 100 % CPU, pas de modele HuggingFace charge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from greentech.ai.mlops.calibration import (
    TemperatureResult,
    TemperatureScaler,
    ThresholdResult,
    apply_calibration,
    find_optimal_threshold,
    load_calibration,
    save_calibration,
)


class TestTemperatureScaler:
    """Temperature scaling via LBFGS sur la NLL."""

    def test_initial_temperature_is_one(self) -> None:
        """Avant tout fit, T = exp(log(0)) = 1.0 (identite)."""
        scaler = TemperatureScaler()
        assert scaler.temperature == pytest.approx(1.0)

    def test_transform_with_t_one_equals_standard_softmax(self) -> None:
        """A T=1, ``transform`` doit retourner le softmax standard."""
        scaler = TemperatureScaler()
        logits = np.array([[2.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        probs = scaler.transform(logits)

        expected = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        np.testing.assert_allclose(probs, expected, atol=1e-6)

    def test_fit_reduces_nll_on_overconfident_logits(self) -> None:
        """Sur des logits sur-confiants (magnitude elevee), T doit augmenter (>1).

        Intuition : un modele qui sort [5, -5] au lieu de [1, -1] est
        sur-confiant ; le temperature scaling apprend T > 1 pour aplatir
        la distribution et mieux refleter la vraie incertitude.
        """
        # 50 % positifs, 50 % negatifs, mais logits tres tranchants
        n = 100
        rng = np.random.default_rng(seed=42)
        labels = rng.integers(0, 2, size=n)
        logits = np.stack(
            [
                -10.0 * labels + 5.0 * (1 - labels),  # classe 0 -> logit eleve
                10.0 * labels - 5.0 * (1 - labels),  # classe 1 -> logit eleve
            ],
            axis=1,
        ).astype(np.float32)
        # Injecter 20 % de bruit : le modele sur-confiant se trompe parfois
        flip_idx = rng.choice(n, size=20, replace=False)
        logits[flip_idx] = -logits[flip_idx]

        scaler = TemperatureScaler()
        result = scaler.fit(logits, labels, max_iter=50)

        # T doit etre > 1 pour corriger la sur-confiance
        assert result.temperature > 1.0
        # NLL doit avoir strictement diminue
        assert result.nll_after < result.nll_before
        # Metadata non vide
        assert result.n_iterations > 0

    def test_fit_preserves_argmax(self) -> None:
        """Le temperature scaling ne change que la confiance, pas l'argmax.

        Propriete fondamentale : diviser les logits par une constante
        positive ne change pas leur ordre. Donc ``argmax(T * logits) ==
        argmax(logits)`` pour tout T > 0.
        """
        scaler = TemperatureScaler()
        logits = np.array([[3.0, -1.0], [-2.0, 1.5], [0.1, 0.2]], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int64)
        scaler.fit(logits, labels, max_iter=10)

        probs_calibrated = scaler.transform(logits)
        argmax_orig = np.argmax(logits, axis=1)
        argmax_calib = np.argmax(probs_calibrated, axis=1)
        np.testing.assert_array_equal(argmax_orig, argmax_calib)

    def test_fit_accepts_torch_tensors_and_numpy_arrays(self) -> None:
        """Les inputs peuvent etre des numpy arrays OU des torch tensors."""
        logits = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
        labels = np.array([0, 1] * 10, dtype=np.int64)

        scaler_np = TemperatureScaler()
        result_np = scaler_np.fit(logits, labels, max_iter=10)

        scaler_torch = TemperatureScaler()
        result_torch = scaler_torch.fit(
            torch.from_numpy(logits), torch.from_numpy(labels), max_iter=10
        )

        # Les T optimaux doivent etre tres proches (meme algorithme)
        assert result_np.temperature == pytest.approx(result_torch.temperature, rel=1e-4)


class TestFindOptimalThreshold:
    """Scan de grille de seuils pour maximiser MCC ou F1."""

    def test_returns_threshold_in_grid_range(self) -> None:
        """Le seuil retourne doit etre dans ``[grid_start, grid_stop]``."""
        rng = np.random.default_rng(seed=1)
        y_true = rng.integers(0, 2, size=200)
        y_proba = rng.uniform(0, 1, size=200)

        result = find_optimal_threshold(
            y_true, y_proba, metric="mcc", grid_start=0.05, grid_stop=0.95
        )
        assert 0.05 <= result.threshold <= 0.95

    def test_perfectly_separable_data_yields_mcc_one(self) -> None:
        """Sur un dataset parfaitement separable (probas 0 ou 1), MCC = 1."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = find_optimal_threshold(y_true, y_proba, metric="mcc")
        assert result.value == pytest.approx(1.0)
        # Le seuil optimal doit etre entre 0.3 et 0.7
        assert 0.3 <= result.threshold <= 0.7

    def test_f1_metric_variant_also_supported(self) -> None:
        """``metric="f1"`` doit fonctionner et retourner une valeur dans [0, 1]."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_proba = np.array([0.2, 0.7, 0.3, 0.8, 0.6])
        result = find_optimal_threshold(y_true, y_proba, metric="f1")
        assert result.metric == "f1"
        assert 0.0 <= result.value <= 1.0

    def test_mismatched_shapes_raises_value_error(self) -> None:
        """Si ``y_true`` et ``y_proba`` n'ont pas la meme taille : ``ValueError``."""
        with pytest.raises(ValueError, match="meme shape"):
            find_optimal_threshold(np.array([0, 1]), np.array([0.5]))

    def test_invalid_grid_raises_value_error(self) -> None:
        """Grille decroissante ou pas nul : ``ValueError``."""
        y_true = np.array([0, 1])
        y_proba = np.array([0.3, 0.7])

        with pytest.raises(ValueError, match="Grille"):
            find_optimal_threshold(y_true, y_proba, grid_start=0.9, grid_stop=0.1)

        with pytest.raises(ValueError, match="Grille"):
            find_optimal_threshold(y_true, y_proba, grid_step=0.0)

    def test_grid_values_stored_for_plotting(self) -> None:
        """``result.grid_values`` doit contenir tous les (seuil, score) scannes."""
        y_true = np.array([0, 1] * 20)
        y_proba = np.array([0.3, 0.7] * 20)
        result = find_optimal_threshold(
            y_true, y_proba, grid_start=0.1, grid_stop=0.9, grid_step=0.1
        )
        # 9 seuils : 0.1, 0.2, ..., 0.9
        assert result.grid_values is not None
        assert len(result.grid_values) == 9


class TestSaveLoadCalibration:
    """Round-trip JSON temperature.json + optimal_threshold.json."""

    def test_save_and_load_temperature_round_trip(self, tmp_path: Path) -> None:
        """Apres save puis load, la valeur T doit etre preservee exactement."""
        temp_result = TemperatureResult(
            temperature=1.87,
            nll_before=0.45,
            nll_after=0.32,
            n_iterations=12,
        )
        save_calibration(tmp_path, temperature=temp_result)

        loaded_t, loaded_thresh = load_calibration(tmp_path)
        assert loaded_t == pytest.approx(1.87)
        # Pas de threshold.json sauvegarde => None
        assert loaded_thresh is None

    def test_save_and_load_threshold_round_trip(self, tmp_path: Path) -> None:
        """Apres save puis load, le seuil doit etre preserve exactement."""
        thresh_result = ThresholdResult(
            threshold=0.42,
            metric="mcc",
            value=0.78,
        )
        save_calibration(tmp_path, threshold=thresh_result)

        loaded_t, loaded_thresh = load_calibration(tmp_path)
        assert loaded_t is None
        assert loaded_thresh == pytest.approx(0.42)

    def test_save_both_load_both(self, tmp_path: Path) -> None:
        """Temperature + threshold peuvent etre sauvegardes/recharges ensemble."""
        save_calibration(
            tmp_path,
            temperature=TemperatureResult(1.5, 0.4, 0.3, 10),
            threshold=ThresholdResult(0.35, "mcc", 0.75),
        )
        t, thresh = load_calibration(tmp_path)
        assert t == pytest.approx(1.5)
        assert thresh == pytest.approx(0.35)

    def test_load_from_empty_dir_returns_none_none(self, tmp_path: Path) -> None:
        """Un dossier sans calibration doit retourner ``(None, None)`` sans crasher.

        Propriete critique : ``inference.py`` compte sur ce fallback
        silencieux pour pouvoir charger des modeles anterieurs au protocole B3.
        """
        t, thresh = load_calibration(tmp_path)
        assert t is None
        assert thresh is None


class TestApplyCalibration:
    """Transformation combinee logits -> (pos_probs, predictions).

    ``apply_calibration`` retourne un tuple ``(probas_classe_positive,
    labels_binaires)`` apres application de T puis du seuil optimal.
    """

    def test_apply_returns_tuple_of_two_arrays(self) -> None:
        """Retour : ``(pos_probs, predictions)``, tous deux de shape (n,)."""
        logits = np.array([[1.0, -1.0], [-2.0, 2.0]], dtype=np.float32)
        pos_probs, preds = apply_calibration(logits)

        assert pos_probs.shape == (2,)
        assert preds.shape == (2,)
        assert pos_probs.dtype == np.float32
        assert preds.dtype == np.int64

    def test_apply_with_t_one_equals_softmax_class_1(self) -> None:
        """A T=1, ``pos_probs`` = softmax[:, 1] (classe positive)."""
        logits = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)
        pos_probs, _ = apply_calibration(logits, temperature=1.0)

        expected = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1]
        np.testing.assert_allclose(pos_probs, expected, atol=1e-5)

    def test_apply_without_temperature_behaves_like_t_one(self) -> None:
        """``temperature=None`` doit donner le meme resultat que T=1.0."""
        logits = np.array([[1.0, -1.0]], dtype=np.float32)
        probs_none, _ = apply_calibration(logits, temperature=None)
        probs_one, _ = apply_calibration(logits, temperature=1.0)
        np.testing.assert_allclose(probs_none, probs_one, atol=1e-6)

    def test_apply_with_high_t_flattens_distribution(self) -> None:
        """Un T eleve rapproche ``pos_probs`` de 0.5 (distribution plus plate)."""
        logits = np.array([[5.0, -5.0]], dtype=np.float32)
        probs_cold, _ = apply_calibration(logits, temperature=1.0)
        probs_hot, _ = apply_calibration(logits, temperature=10.0)

        # pos_probs negative (logit de classe 1 < 0) : plus T est eleve,
        # plus la proba se rapproche de 0.5 depuis le bas.
        assert probs_cold[0] < probs_hot[0] < 0.5

    def test_threshold_affects_predictions_not_probas(self) -> None:
        """Le seuil influence uniquement ``predictions``, pas ``pos_probs``."""
        logits = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        probs_a, preds_a = apply_calibration(logits, threshold=0.3)
        probs_b, preds_b = apply_calibration(logits, threshold=0.7)

        # Les probas sont identiques
        np.testing.assert_allclose(probs_a, probs_b, atol=1e-6)
        # Les predictions peuvent differer selon le seuil
        # Avec seuil 0.3, plus de positifs ; avec seuil 0.7, moins.
        assert preds_a.sum() >= preds_b.sum()

    def test_threshold_default_is_half(self) -> None:
        """Si ``threshold=None``, seuil = 0.5 (comportement standard)."""
        logits = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
        _, preds_none = apply_calibration(logits, threshold=None)
        _, preds_half = apply_calibration(logits, threshold=0.5)
        np.testing.assert_array_equal(preds_none, preds_half)
