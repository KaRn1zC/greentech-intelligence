"""Tests de drift de distribution entre les folds K-fold (B3.6).

Une stratification correcte doit garantir que les K folds du K-fold sont
distributionnellement similaires sur les dimensions cles :

1. **Langue** : repartition EN/FR doit etre stable entre folds (cible 75/25)
2. **Label** : taux Green IT doit etre stable entre folds (cible ~8.73 %)
3. **Longueur de texte** : statistiques (moyenne, ecart-type) doivent rester
   coherentes entre folds

Si la stratification est defaillante (par ex. dataset trop petit, ratio
extreme, mauvais stratifier), la variance MCC inter-fold explose - c'est
le syndrome documente sur le run v20260415 (sigma MCC = 0.25). L'objectif
de B3 est de ramener sigma < 0.10 grace a la stratification croisee
``(langue x label)`` via ``MultilabelStratifiedKFold`` du package
``iterative-stratification``.

Ce module verifie l'invariant directement sur des K-folds synthetiques
avec distribution proche du dataset reel de production (B2.9 / B3.1).
"""

from __future__ import annotations

import numpy as np
import pytest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from greentech.ai.models.training import (
    _STRATIFICATION_TARGET_RATIOS,
    _STRATIFICATION_TOLERANCE_PP,
)

# === Helpers ===


def _build_realistic_dataset(
    n_total: int = 1000,
    *,
    ratio_en: float = 0.7475,
    ratio_green_en: float = 0.0480,
    ratio_green_fr: float = 0.2037,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genere un dataset synthetique avec distribution proche du dataset reel.

    Args:
        n_total: Nombre total d'articles.
        ratio_en: Proportion d'articles en anglais.
        ratio_green_en: Proportion de Green IT parmi les articles EN.
        ratio_green_fr: Proportion de Green IT parmi les articles FR.
        seed: Graine RNG pour la reproductibilite.

    Returns:
        Tuple ``(texts, labels, langues)`` ou ``texts`` contient des chaines
        de longueur aleatoire (entre 80 et 350 caracteres), et ``labels``,
        ``langues`` sont les vecteurs correspondants.
    """
    rng = np.random.default_rng(seed)
    n_en = int(n_total * ratio_en)
    n_fr = n_total - n_en

    n_green_en = int(n_en * ratio_green_en)
    n_green_fr = int(n_fr * ratio_green_fr)

    labels_en = np.array([1] * n_green_en + [0] * (n_en - n_green_en))
    labels_fr = np.array([1] * n_green_fr + [0] * (n_fr - n_green_fr))
    rng.shuffle(labels_en)
    rng.shuffle(labels_fr)

    labels_arr = np.concatenate([labels_en, labels_fr])
    langues_arr = np.array(["en"] * n_en + ["fr"] * n_fr, dtype=object)

    # Longueurs realistes : 80-350 chars (titre 5-15 mots + resume 150-220 mots,
    # 1 mot ~= 5 chars en moyenne)
    lengths = rng.integers(low=80, high=350, size=n_total)
    texts_arr = np.array(
        [
            f"Titre article {i}\n\n" + ("contenu " * (length // 8))
            for i, length in enumerate(lengths)
        ],
        dtype=object,
    )

    # Re-permutation globale pour eviter le bloc "EN puis FR"
    perm = rng.permutation(n_total)
    return texts_arr[perm], labels_arr[perm], langues_arr[perm]


def _compute_fold_stats(
    labels_fold: np.ndarray, langues_fold: np.ndarray, texts_fold: np.ndarray
) -> dict[str, float]:
    """Calcule les ratios et statistiques de longueur sur un fold."""
    n = len(labels_fold)
    if n == 0:
        return {
            "ratio_en": 0.0,
            "ratio_fr": 0.0,
            "ratio_green": 0.0,
            "ratio_green_en": 0.0,
            "ratio_green_fr": 0.0,
            "length_mean": 0.0,
            "length_std": 0.0,
        }

    en_mask = langues_fold == "en"
    fr_mask = langues_fold == "fr"
    n_en = int(en_mask.sum())
    n_fr = int(fr_mask.sum())

    text_lengths = np.array([len(str(t)) for t in texts_fold])

    return {
        "ratio_en": n_en / n,
        "ratio_fr": n_fr / n,
        "ratio_green": float(labels_fold.sum()) / n,
        "ratio_green_en": float(labels_fold[en_mask].sum()) / n_en if n_en else 0.0,
        "ratio_green_fr": float(labels_fold[fr_mask].sum()) / n_fr if n_fr else 0.0,
        "length_mean": float(text_lengths.mean()),
        "length_std": float(text_lengths.std()),
    }


def _run_kfold_and_collect_stats(
    texts_arr: np.ndarray,
    labels_arr: np.ndarray,
    langues_arr: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Lance un K-fold MultilabelStratifiedKFold + retourne les stats val par fold."""
    lang_en = (langues_arr == "en").astype(int)
    lang_fr = (langues_arr == "fr").astype(int)
    strat_labels = np.stack([lang_en, lang_fr, labels_arr], axis=1)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_stats: list[dict[str, float]] = []
    for _train_idx, val_idx in mskf.split(strat_labels, strat_labels):
        stats = _compute_fold_stats(labels_arr[val_idx], langues_arr[val_idx], texts_arr[val_idx])
        folds_stats.append(stats)
    return folds_stats


# === Tests de drift langue ===


class TestLangueDistributionDrift:
    """Verifie que la repartition EN/FR est stable entre folds."""

    @pytest.mark.parametrize(
        "n_total,n_splits,seed",
        [
            (1000, 5, 42),
            (2000, 5, 123),
            (1000, 3, 999),
            (5000, 5, 7),
        ],
    )
    def test_max_min_ratio_en_inferieur_a_tolerance(
        self, n_total: int, n_splits: int, seed: int
    ) -> None:
        """L'ecart max-min du ratio EN entre folds doit rester sous 2*tolerance.

        Note : on autorise 2*tolerance (~4pp) car la tolerance documentee est
        pour l'ecart fold-vs-cible, pas fold-vs-fold. Deux folds extremes
        peuvent legitimement etre a -2pp et +2pp.
        """
        texts, labels, langues = _build_realistic_dataset(n_total=n_total, seed=seed)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits, seed)

        ratios_en = [f["ratio_en"] for f in folds]
        drift = max(ratios_en) - min(ratios_en)
        assert drift < 2 * _STRATIFICATION_TOLERANCE_PP, (
            f"Drift langue EN excessif : max-min={drift * 100:.2f}pp "
            f"(seuil={2 * _STRATIFICATION_TOLERANCE_PP * 100:.0f}pp). "
            f"Ratios par fold : {[f'{r:.4f}' for r in ratios_en]}"
        )

    def test_chaque_fold_proche_de_la_cible_globale(self) -> None:
        """Chaque fold individuel doit etre proche de la cible globale 75/25."""
        texts, labels, langues = _build_realistic_dataset(n_total=2000, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        target_en = _STRATIFICATION_TARGET_RATIOS["ratio_en"]
        for i, fold in enumerate(folds, 1):
            diff = abs(fold["ratio_en"] - target_en)
            assert diff <= _STRATIFICATION_TOLERANCE_PP, (
                f"Fold {i} ratio_en={fold['ratio_en']:.4f} devie de "
                f"{diff * 100:.2f}pp de la cible {target_en:.4f}"
            )


# === Tests de drift label ===


class TestLabelDistributionDrift:
    """Verifie que la repartition Green IT / Non Green IT est stable."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_max_min_ratio_green_inferieur_a_tolerance(self, seed: int) -> None:
        """L'ecart max-min du ratio Green entre folds < 2*tolerance."""
        texts, labels, langues = _build_realistic_dataset(n_total=2000, seed=seed)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=seed)

        ratios_green = [f["ratio_green"] for f in folds]
        drift = max(ratios_green) - min(ratios_green)
        assert drift < 2 * _STRATIFICATION_TOLERANCE_PP, (
            f"Drift label Green excessif : max-min={drift * 100:.2f}pp "
            f"(seuil={2 * _STRATIFICATION_TOLERANCE_PP * 100:.0f}pp). "
            f"Ratios par fold : {[f'{r:.4f}' for r in ratios_green]}"
        )

    def test_aucun_fold_sans_positif(self) -> None:
        """Avec stratification correcte, chaque fold val doit avoir au moins
        un positif (sauf cas degenere ou n_total tres petit)."""
        texts, labels, langues = _build_realistic_dataset(n_total=2000, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        for i, fold in enumerate(folds, 1):
            assert fold["ratio_green"] > 0, (
                f"Fold {i} ne contient aucun positif (ratio_green=0). Stratification defaillante."
            )


# === Tests de drift longueur de texte ===


class TestLengthDistributionDrift:
    """Verifie que les statistiques de longueur de texte sont stables."""

    def test_moyenne_longueur_proche_entre_folds(self) -> None:
        """L'ecart relatif de la moyenne de longueur entre folds doit rester
        sous 10 % (ce drift n'est pas directement controle par la stratification
        mais devrait emerger naturellement avec un dataset de taille raisonnable)."""
        texts, labels, langues = _build_realistic_dataset(n_total=2000, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        means = [f["length_mean"] for f in folds]
        global_mean = sum(means) / len(means)
        max_deviation = max(abs(m - global_mean) for m in means) / global_mean

        assert max_deviation < 0.10, (
            f"Drift longueur excessif : max deviation relative = "
            f"{max_deviation * 100:.2f}% (seuil 10%). "
            f"Moyennes par fold : {[f'{m:.1f}' for m in means]}"
        )


# === Tests croisees (langue x label) ===


class TestCrossDimensionDrift:
    """Verifie la stratification croisee langue x label simultanement."""

    def test_ratio_green_fr_stable_entre_folds(self) -> None:
        """La densite Green parmi les articles FR doit rester stable entre folds.
        FR est 4.2x plus dense en Green que EN (B2.9), c'est la dimension la plus
        sensible au drift."""
        texts, labels, langues = _build_realistic_dataset(n_total=3000, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        ratios_green_fr = [f["ratio_green_fr"] for f in folds]
        drift = max(ratios_green_fr) - min(ratios_green_fr)
        # Tolerance plus large car ratio_green_fr est calcule sur le sous-ensemble FR
        # du fold (effectif plus petit donc plus de variance acceptable)
        assert drift < 0.10, (
            f"Drift ratio_green_fr excessif : max-min={drift * 100:.2f}pp "
            f"(seuil 10pp). Ratios : {[f'{r:.4f}' for r in ratios_green_fr]}"
        )

    def test_aucun_fold_sans_positif_fr(self) -> None:
        """Avec stratification croisee, aucun fold ne doit avoir ratio_green_fr=0."""
        texts, labels, langues = _build_realistic_dataset(n_total=2000, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        for i, fold in enumerate(folds, 1):
            assert fold["ratio_green_fr"] > 0, (
                f"Fold {i} val n'a aucun positif FR alors qu'il devrait "
                f"en avoir (Green IT FR au taux global de "
                f"{_STRATIFICATION_TARGET_RATIOS['ratio_green_fr']:.2%})"
            )

    def test_stratification_robuste_a_petits_n_total(self) -> None:
        """La stratification doit tenir meme sur petit dataset (n=500)
        - les ratios doivent rester proches de la cible meme si la variance
        est mecaniquement plus grande."""
        texts, labels, langues = _build_realistic_dataset(n_total=500, seed=42)
        folds = _run_kfold_and_collect_stats(texts, labels, langues, n_splits=5, seed=42)

        # Tolerance relaxee a 5pp pour n=500 (vs 2pp pour n=2000)
        target_en = _STRATIFICATION_TARGET_RATIOS["ratio_en"]
        for i, fold in enumerate(folds, 1):
            diff = abs(fold["ratio_en"] - target_en)
            assert diff <= 0.05, (
                f"Fold {i} (n=500) ratio_en={fold['ratio_en']:.4f} devie "
                f"de {diff * 100:.2f}pp de la cible {target_en:.4f}"
            )
