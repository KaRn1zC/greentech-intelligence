"""Tests unitaires de l'ensembling K-fold (B3.5).

Couvre :

1. ``_select_best_seed_per_fold`` : selection de la meilleure seed par
   fold (max MCC, F1 en tie-break).
2. ``_build_ensemble`` : generation de ``ensemble_config.json`` pour la
   strategie ``logit_average`` (mDeBERTa). La strategie ``merge_lora``
   (Qwen3) n'est pas testee ici car elle necessite le chargement d'un
   modele Qwen3-4B (7 Go VRAM).
3. ``EnsembleClassifier`` : chargement de K membres via factory,
   moyenne des ``proba_positive`` a l'inference, gestion du cas
   ``proba_positive=None`` (fallback score_confiance).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from greentech.ai.models.classifier import (
    BaseClassifier,
    LabelGreenIT,
    PredictionResult,
    TrainingConfig,
)
from greentech.ai.models.inference import EnsembleClassifier
from greentech.ai.models.training import _build_ensemble, _select_best_seed_per_fold


class _StubClassifier(BaseClassifier):
    """Classifier stub qui retourne une proba fixe pour les tests ensemble."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        proba_positive: float | None,
        score_confiance: float = 0.6,
        est_green_it_forced: bool | None = None,
    ) -> None:
        super().__init__(config)
        self._proba = proba_positive
        self._score = score_confiance
        self._forced_green = est_green_it_forced

    async def train(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        return {}

    def save(self, output_dir: Path | None = None) -> Path:
        return output_dir or Path("/tmp")

    def load(self, model_path: Path) -> None:
        pass

    async def predict(self, text: str) -> PredictionResult:
        _ = text  # non utilise
        if self._forced_green is not None:
            label = LabelGreenIT.GREEN if self._forced_green else LabelGreenIT.NON_GREEN
        elif self._proba is None:
            label = LabelGreenIT.GREEN
        else:
            label = LabelGreenIT.GREEN if self._proba >= 0.5 else LabelGreenIT.NON_GREEN
        return PredictionResult(
            label=label,
            score_confiance=float(self._score),
            temps_ms=1,
            modele="stub",
            proba_positive=self._proba,
        )


class TestSelectBestSeedPerFold:
    """Selection de la meilleure seed pour chaque fold K-fold."""

    def test_selects_highest_mcc_per_fold(self) -> None:
        """Parmi 3 seeds d'un fold, celle avec le MCC le plus eleve gagne."""
        run_metrics = [
            {"fold": 1, "seed_idx": 1, "seed": 42, "mcc": 0.70, "f1": 0.65},
            {"fold": 1, "seed_idx": 2, "seed": 43, "mcc": 0.82, "f1": 0.72},
            {"fold": 1, "seed_idx": 3, "seed": 44, "mcc": 0.78, "f1": 0.74},
        ]
        best = _select_best_seed_per_fold(run_metrics)
        assert len(best) == 1
        assert best[0]["seed_idx"] == 2
        assert best[0]["mcc"] == 0.82

    def test_f1_is_tiebreaker_when_mcc_equal(self) -> None:
        """En cas d'egalite MCC, le F1 le plus eleve departage."""
        run_metrics = [
            {"fold": 2, "seed_idx": 1, "seed": 42, "mcc": 0.75, "f1": 0.70},
            {"fold": 2, "seed_idx": 2, "seed": 43, "mcc": 0.75, "f1": 0.80},
            {"fold": 2, "seed_idx": 3, "seed": 44, "mcc": 0.75, "f1": 0.72},
        ]
        best = _select_best_seed_per_fold(run_metrics)
        assert best[0]["seed_idx"] == 2
        assert best[0]["f1"] == 0.80

    def test_returns_one_entry_per_fold(self) -> None:
        """K folds => K entrees retournees, dans l'ordre des fold_idx."""
        run_metrics = [
            {"fold": 1, "seed_idx": 1, "seed": 42, "mcc": 0.70, "f1": 0.65},
            {"fold": 1, "seed_idx": 2, "seed": 43, "mcc": 0.80, "f1": 0.72},
            {"fold": 2, "seed_idx": 1, "seed": 42, "mcc": 0.60, "f1": 0.55},
            {"fold": 2, "seed_idx": 2, "seed": 43, "mcc": 0.65, "f1": 0.62},
            {"fold": 3, "seed_idx": 1, "seed": 42, "mcc": 0.90, "f1": 0.85},
        ]
        best = _select_best_seed_per_fold(run_metrics)
        assert len(best) == 3
        folds = [b["fold"] for b in best]
        assert folds == [1, 2, 3]

    def test_single_seed_fold_returned_as_is(self) -> None:
        """Un fold avec une seule seed doit etre retourne tel quel."""
        run_metrics = [
            {"fold": 1, "seed_idx": 1, "seed": 42, "mcc": 0.70, "f1": 0.65},
        ]
        best = _select_best_seed_per_fold(run_metrics)
        assert len(best) == 1
        assert best[0]["seed_idx"] == 1


class TestBuildEnsembleMDeBERTa:
    """Generation de ``ensemble_config.json`` pour mDeBERTa (strategy=logit_average).

    On ne teste pas la variante Qwen3 (``merge_lora``) car elle necessite
    le chargement d'un Qwen3-4B (7 Go VRAM + ~30s de download).
    """

    def _make_run_metrics(self) -> list[dict]:
        """Jeu de donnees synthetique : 2 folds x 2 seeds chacun."""
        return [
            {
                "fold": 1,
                "seed_idx": 1,
                "seed": 42,
                "mcc": 0.70,
                "f1": 0.65,
                "recall": 0.60,
                "precision": 0.72,
                "temperature": 1.2,
                "threshold": 0.42,
            },
            {
                "fold": 1,
                "seed_idx": 2,
                "seed": 43,
                "mcc": 0.80,  # best fold 1
                "f1": 0.72,
                "recall": 0.68,
                "precision": 0.78,
                "temperature": 1.3,
                "threshold": 0.40,
            },
            {
                "fold": 2,
                "seed_idx": 1,
                "seed": 42,
                "mcc": 0.75,  # best fold 2
                "f1": 0.70,
                "recall": 0.65,
                "precision": 0.78,
                "temperature": 1.15,
                "threshold": 0.44,
            },
            {
                "fold": 2,
                "seed_idx": 2,
                "seed": 43,
                "mcc": 0.72,
                "f1": 0.69,
                "recall": 0.60,
                "precision": 0.80,
                "temperature": 1.2,
                "threshold": 0.43,
            },
        ]

    def _setup_folds_root(self, tmp_path: Path) -> Path:
        """Cree les dossiers fold_X_seed_Y vides (simulation de checkpoints)."""
        folds_root = tmp_path / "folds"
        folds_root.mkdir()
        for fold in [1, 2]:
            for seed in [1, 2]:
                (folds_root / f"fold_{fold}_seed_{seed}").mkdir()
        return folds_root

    def test_writes_ensemble_config_json(self, tmp_path: Path) -> None:
        """``_build_ensemble`` doit creer ``ensemble_config.json`` a la racine."""
        folds_root = self._setup_folds_root(tmp_path)
        info = _build_ensemble(
            model_type="mdeberta",
            model_output_root=tmp_path,
            run_metrics=self._make_run_metrics(),
            folds_root=folds_root,
            mean_temperature=1.22,
            mean_threshold=0.42,
            aggregated={"mcc": {"mean": 0.775, "std": 0.04}},
        )

        cfg_path = tmp_path / "ensemble_config.json"
        assert cfg_path.exists()

        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert data["strategy"] == "logit_average"
        assert data["model_type"] == "mdeberta"
        assert info["strategy"] == "logit_average"

    def test_selects_best_seed_in_config(self, tmp_path: Path) -> None:
        """Les folds listes doivent utiliser la meilleure seed selon MCC."""
        folds_root = self._setup_folds_root(tmp_path)
        info = _build_ensemble(
            model_type="mdeberta",
            model_output_root=tmp_path,
            run_metrics=self._make_run_metrics(),
            folds_root=folds_root,
            mean_temperature=1.22,
            mean_threshold=0.42,
            aggregated={"mcc": {"mean": 0.775, "std": 0.04}},
        )

        folds = info["folds"]
        assert len(folds) == 2
        # Fold 1 meilleure seed = 2 (MCC 0.80)
        assert folds[0]["fold"] == 1
        assert folds[0]["seed_idx"] == 2
        # Fold 2 meilleure seed = 1 (MCC 0.75)
        assert folds[1]["fold"] == 2
        assert folds[1]["seed_idx"] == 1

    def test_inference_paths_list_for_mdeberta(self, tmp_path: Path) -> None:
        """Pour mDeBERTa, ``inference_model_paths`` contient les K chemins."""
        folds_root = self._setup_folds_root(tmp_path)
        info = _build_ensemble(
            model_type="mdeberta",
            model_output_root=tmp_path,
            run_metrics=self._make_run_metrics(),
            folds_root=folds_root,
            mean_temperature=1.22,
            mean_threshold=0.42,
            aggregated={"mcc": {"mean": 0.775, "std": 0.04}},
        )

        assert "inference_model_paths" in info
        paths = info["inference_model_paths"]
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()

    def test_metadata_includes_cv_stats(self, tmp_path: Path) -> None:
        """Metadata doit preserver MCC mean/std pour la tracabilite."""
        folds_root = self._setup_folds_root(tmp_path)
        info = _build_ensemble(
            model_type="mdeberta",
            model_output_root=tmp_path,
            run_metrics=self._make_run_metrics(),
            folds_root=folds_root,
            mean_temperature=1.22,
            mean_threshold=0.42,
            aggregated={"mcc": {"mean": 0.775, "std": 0.04}},
        )
        meta = info["metadata"]
        assert meta["cv_mcc_mean"] == pytest.approx(0.775)
        assert meta["cv_mcc_std"] == pytest.approx(0.04)
        assert meta["n_folds"] == 2
        assert "built_at" in meta

    def test_unknown_model_type_raises(self, tmp_path: Path) -> None:
        """Un ``model_type`` inconnu doit lever ``ValueError``."""
        folds_root = self._setup_folds_root(tmp_path)
        with pytest.raises(ValueError, match="Ensemble non supporte"):
            _build_ensemble(
                model_type="unknown-model",
                model_output_root=tmp_path,
                run_metrics=self._make_run_metrics(),
                folds_root=folds_root,
                mean_temperature=1.0,
                mean_threshold=0.5,
                aggregated={"mcc": {"mean": 0.7, "std": 0.05}},
            )


class TestEnsembleClassifier:
    """Moyenne des logits a l'inference (strategy logit_average)."""

    def _make_ensemble(
        self,
        probas: list[float | None],
        tmp_path: Path,
    ) -> EnsembleClassifier:
        """Construit un ensemble avec ``len(probas)`` membres stubs."""
        fold_paths = [tmp_path / f"fold_{i}" for i in range(1, len(probas) + 1)]
        for p in fold_paths:
            p.mkdir()

        probas_iter = iter(probas)

        def factory(path: Path) -> BaseClassifier:
            return _StubClassifier(
                TrainingConfig(nom_modele=str(path)),
                proba_positive=next(probas_iter),
            )

        ensemble = EnsembleClassifier(
            fold_paths=fold_paths,
            classifier_factory=factory,
        )
        ensemble.load(tmp_path)
        return ensemble

    def test_empty_fold_paths_raises(self) -> None:
        """Instancier un ensemble sans aucun membre doit lever."""
        with pytest.raises(ValueError, match="au moins 1 fold_path"):
            EnsembleClassifier(
                fold_paths=[],
                classifier_factory=lambda _p: _StubClassifier(
                    TrainingConfig(nom_modele="x"),
                    proba_positive=0.5,
                ),
            )

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path: Path) -> None:
        """Si un checkpoint n'existe pas sur disque : ``FileNotFoundError``."""
        ensemble = EnsembleClassifier(
            fold_paths=[tmp_path / "inexistant"],
            classifier_factory=lambda p: _StubClassifier(
                TrainingConfig(nom_modele=str(p)),
                proba_positive=0.5,
            ),
        )
        with pytest.raises(FileNotFoundError):
            ensemble.load(tmp_path)

    def test_predict_averages_probas(self, tmp_path: Path) -> None:
        """3 membres avec probas [0.3, 0.6, 0.8] => moyenne 0.5667 => GREEN."""
        ensemble = self._make_ensemble([0.3, 0.6, 0.8], tmp_path)
        result = asyncio.run(ensemble.predict("texte"))

        avg_expected = (0.3 + 0.6 + 0.8) / 3
        assert result.proba_positive == pytest.approx(avg_expected)
        assert result.label == LabelGreenIT.GREEN
        assert "ensemble" in result.modele
        assert "3x" in result.modele

    def test_predict_mean_below_half_is_non_green(self, tmp_path: Path) -> None:
        """Moyenne des probas < 0.5 => label NON_GREEN."""
        ensemble = self._make_ensemble([0.1, 0.3, 0.2], tmp_path)
        result = asyncio.run(ensemble.predict("texte"))

        assert result.proba_positive == pytest.approx(0.2)
        assert result.label == LabelGreenIT.NON_GREEN

    def test_predict_before_load_raises(self, tmp_path: Path) -> None:
        """Predict avant load doit lever ``RuntimeError`` explicite."""
        ensemble = EnsembleClassifier(
            fold_paths=[tmp_path],
            classifier_factory=lambda p: _StubClassifier(
                TrainingConfig(nom_modele=str(p)),
                proba_positive=0.5,
            ),
        )
        with pytest.raises(RuntimeError, match="avant load"):
            asyncio.run(ensemble.predict("texte"))

    def test_fallback_when_member_returns_none_proba(self, tmp_path: Path) -> None:
        """Si un membre ne fournit pas proba_positive, on derive depuis score."""
        # Member 1 : proba=0.8 (GREEN). Member 2 : proba=None, force GREEN,
        # score_confiance=0.7 => pseudo-proba=0.7 (car label GREEN).
        fold_paths = [tmp_path / "m1", tmp_path / "m2"]
        for p in fold_paths:
            p.mkdir()

        members_config = [
            {"proba": 0.8, "score": 0.9, "forced_green": True},
            {"proba": None, "score": 0.7, "forced_green": True},
        ]
        cfg_iter = iter(members_config)

        def factory(path: Path) -> BaseClassifier:
            cfg = next(cfg_iter)
            return _StubClassifier(
                TrainingConfig(nom_modele=str(path)),
                proba_positive=cfg["proba"],
                score_confiance=cfg["score"],
                est_green_it_forced=cfg["forced_green"],
            )

        ensemble = EnsembleClassifier(fold_paths=fold_paths, classifier_factory=factory)
        ensemble.load(tmp_path)

        result = asyncio.run(ensemble.predict("texte"))
        # moyenne = (0.8 + 0.7) / 2 = 0.75
        assert result.proba_positive == pytest.approx(0.75)
        assert result.label == LabelGreenIT.GREEN

    def test_train_raises_not_implemented(self, tmp_path: Path) -> None:
        """L'ensemble n'est pas entrainable (reassemblage uniquement)."""
        ensemble = self._make_ensemble([0.5], tmp_path)
        with pytest.raises(NotImplementedError, match="n'implemente pas train"):
            asyncio.run(ensemble.train([], [], [], []))
