"""Tests unitaires de la stratification croisee (langue x label) B3.1.

Verifie que les helpers de verification des folds se comportent correctement :

1. ``_STRATIFICATION_TARGET_RATIOS`` est coherent avec le dataset de production
   (EN 74.75 %, FR 25.25 %, Green 8.73 % global, 4.80 % parmi EN, 20.37 % FR).
2. ``_check_fold_stratification`` detecte les deviations au-dela de la
   tolerance de 2 pp, logge un warning en mode souple et leve
   ``AssertionError`` en mode strict.
3. ``_log_fold_split_stats`` retourne bien les ratios observes calcules
   correctement sur un splits fictif.

Les tests n'appellent pas PyTorch, ne touchent pas au GPU et ne chargent
aucun modele : ils valident la logique pure des helpers.
"""

from __future__ import annotations

import pytest

from greentech.ai.models.training import (
    _STRATIFICATION_TARGET_RATIOS,
    _STRATIFICATION_TOLERANCE_PP,
    _check_fold_stratification,
    _log_fold_split_stats,
)


class TestStratificationTargets:
    """Coherence des constantes cibles avec le dataset de production."""

    def test_tolerance_is_two_percentage_points(self) -> None:
        """La tolerance documentee (checklist B3.1) doit etre 2 pp exactement."""
        assert pytest.approx(0.02) == _STRATIFICATION_TOLERANCE_PP

    def test_target_ratios_sum_coherent(self) -> None:
        """Les ratios langue (EN+FR) doivent sommer a 100 % du corpus bilingue."""
        total = (
            _STRATIFICATION_TARGET_RATIOS["ratio_en"] + _STRATIFICATION_TARGET_RATIOS["ratio_fr"]
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_green_global_between_en_and_fr(self) -> None:
        """Le ratio Green global doit etre entre les ratios Green par langue.

        Propriete mathematique : si FR est plus dense en Green (20 %) et EN
        moins dense (5 %), le global (8.7 %) doit se situer entre les deux.
        """
        target = _STRATIFICATION_TARGET_RATIOS
        assert target["ratio_green_en"] < target["ratio_green_global"]
        assert target["ratio_green_global"] < target["ratio_green_fr"]

    def test_all_expected_keys_present(self) -> None:
        """Les 5 cles verifiees par ``_check_fold_stratification`` doivent exister."""
        expected = {
            "ratio_en",
            "ratio_fr",
            "ratio_green_global",
            "ratio_green_en",
            "ratio_green_fr",
        }
        assert set(_STRATIFICATION_TARGET_RATIOS.keys()) == expected


class TestCheckFoldStratification:
    """Logique de detection de deviation par fold."""

    def _conforming_ratios(self) -> dict[str, float]:
        """Retourne un set de ratios observes parfaitement sur la cible."""
        return dict(_STRATIFICATION_TARGET_RATIOS)

    def test_conforming_fold_returns_no_deviation(self) -> None:
        """Un fold strictement sur la cible ne doit remonter aucune deviation."""
        deviations = _check_fold_stratification(
            fold_idx=1,
            observed=self._conforming_ratios(),
            strict=False,
        )
        assert deviations == []

    def test_fold_just_below_tolerance_accepts(self) -> None:
        """Un ecart strictement inferieur a la tolerance reste accepte.

        Le code utilise ``diff > tolerance`` (strict), donc un ecart de
        1.9 pp (< 2 pp) est toujours accepte sans warning. On evite de
        tester pile 2 pp qui est ambigu a cause de l'imprecision flottante.
        """
        observed = self._conforming_ratios()
        observed["ratio_en"] = observed["ratio_en"] + 0.019
        deviations = _check_fold_stratification(
            fold_idx=1,
            observed=observed,
            strict=False,
        )
        assert deviations == []

    def test_fold_slightly_beyond_tolerance_emits_warning(self) -> None:
        """Un ecart de 2.1 pp en mode souple genere une deviation loguee."""
        observed = self._conforming_ratios()
        observed["ratio_en"] = observed["ratio_en"] + 0.021
        deviations = _check_fold_stratification(
            fold_idx=2,
            observed=observed,
            strict=False,
        )
        assert len(deviations) == 1
        assert "ratio_en" in deviations[0]

    def test_multiple_deviations_all_reported(self) -> None:
        """Plusieurs ratios devies sont tous reportes (pas de short-circuit)."""
        observed = {
            # EN s'envole, FR chute, les deux Green aussi => au moins 4 deviations
            "ratio_en": 0.90,
            "ratio_fr": 0.10,
            "ratio_green_global": 0.30,
            "ratio_green_en": 0.25,
            "ratio_green_fr": 0.05,
        }
        deviations = _check_fold_stratification(
            fold_idx=3,
            observed=observed,
            strict=False,
        )
        assert len(deviations) >= 4

    def test_strict_mode_raises_on_first_deviation(self) -> None:
        """En mode strict, la premiere deviation leve ``AssertionError``."""
        observed = self._conforming_ratios()
        observed["ratio_en"] = 0.99  # ecart massif

        with pytest.raises(AssertionError) as exc_info:
            _check_fold_stratification(
                fold_idx=4,
                observed=observed,
                strict=True,
            )
        # Le message doit mentionner la cle devie et la tolerance
        message = str(exc_info.value)
        assert "ratio_en" in message
        assert "strict_stratification" in message

    def test_missing_observed_key_is_skipped_silently(self) -> None:
        """Une cle absente du dict observe n'est simplement pas verifiee.

        Utile pour les cas ou un fold degenere n'a pas de positifs d'une
        langue donnee (``ratio_green_fr`` indefini), on ne doit pas crasher.
        """
        observed = {
            "ratio_en": _STRATIFICATION_TARGET_RATIOS["ratio_en"],
            "ratio_fr": _STRATIFICATION_TARGET_RATIOS["ratio_fr"],
            # ratio_green_* volontairement absents
        }
        deviations = _check_fold_stratification(
            fold_idx=5,
            observed=observed,
            strict=True,  # meme en strict, les cles absentes passent
        )
        assert deviations == []


class TestLogFoldSplitStats:
    """Calcul des ratios observes par ``_log_fold_split_stats``."""

    def test_returns_dict_with_five_expected_keys(self) -> None:
        """Le retour doit contenir exactement les 5 ratios attendus."""
        ratios = _log_fold_split_stats(
            fold_idx=1,
            n_splits=5,
            train_texts_fold=["t1"],
            train_labels_fold=[0],
            val_texts_fold=["v1", "v2"],
            val_labels_fold=[0, 1],
            val_langues_fold=["en", "fr"],
            n_augmented_in_train=0,
        )
        assert set(ratios.keys()) == set(_STRATIFICATION_TARGET_RATIOS.keys())

    def test_ratios_match_manual_computation(self) -> None:
        """Les ratios calcules doivent matcher le calcul manuel attendu.

        Fold synthetique : 10 val articles, 7 EN dont 1 Green, 3 FR dont 2 Green.
        => ratio_en=0.7, ratio_fr=0.3, ratio_green_global=3/10=0.3,
           ratio_green_en=1/7, ratio_green_fr=2/3.
        """
        val_langues = ["en"] * 7 + ["fr"] * 3
        val_labels = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0]

        ratios = _log_fold_split_stats(
            fold_idx=2,
            n_splits=5,
            train_texts_fold=["t"] * 40,
            train_labels_fold=[0] * 40,
            val_texts_fold=["v"] * 10,
            val_labels_fold=val_labels,
            val_langues_fold=val_langues,
            n_augmented_in_train=5,
        )

        assert ratios["ratio_en"] == pytest.approx(0.7)
        assert ratios["ratio_fr"] == pytest.approx(0.3)
        assert ratios["ratio_green_global"] == pytest.approx(0.3)
        assert ratios["ratio_green_en"] == pytest.approx(1 / 7)
        assert ratios["ratio_green_fr"] == pytest.approx(2 / 3)

    def test_empty_val_handled_gracefully(self) -> None:
        """Un val vide (degenere) doit retourner des zeros, pas crasher."""
        ratios = _log_fold_split_stats(
            fold_idx=1,
            n_splits=5,
            train_texts_fold=["t"],
            train_labels_fold=[0],
            val_texts_fold=[],
            val_labels_fold=[],
            val_langues_fold=[],
            n_augmented_in_train=0,
        )
        for value in ratios.values():
            assert value == 0.0

    def test_strict_mode_propagated_to_check(self) -> None:
        """``strict_stratification=True`` dans le logger doit lever si devie."""
        # Fold 100 % EN (ratio_en=1.0), largement au-dela de la tolerance
        with pytest.raises(AssertionError):
            _log_fold_split_stats(
                fold_idx=1,
                n_splits=5,
                train_texts_fold=["t"],
                train_labels_fold=[0],
                val_texts_fold=["v"] * 10,
                val_labels_fold=[0] * 10,
                val_langues_fold=["en"] * 10,
                n_augmented_in_train=0,
                strict_stratification=True,
            )
