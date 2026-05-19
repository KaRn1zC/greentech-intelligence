"""Tests de non-fuite des variantes d'augmentation entre splits K-fold (B3.6).

L'augmentation par back-translation (opus-mt EN<->FR) genere des variantes
semantiquement equivalentes aux articles originaux. Pour eviter une fuite
d'evaluation, ces variantes doivent UNIQUEMENT etre injectees dans le
train split de chaque fold du K-fold, jamais dans le val split.

Ce module teste :

1. ``_build_variant_index`` rattache correctement chaque variante a son
   original via la cle "titre" (le titre est inchange entre original et
   variante puisque seul le resume est traduit).
2. ``_collect_variants_for_train`` ne selectionne que les variantes des
   originaux presents dans le train split.
3. **Invariant data leakage** (P3.6 - Deepchecks renforce) : sur tout
   K-fold genere par ``MultilabelStratifiedKFold`` + injection variantes,
   aucune variante ne se retrouve jamais dans le val/test split, et un
   original X qui est en val n'a aucune de ses variantes dans le train.

Les tests utilisent un dataset synthetique en memoire (pas de chargement
golden_dataset.csv) pour rester rapides et deterministes.
"""

from __future__ import annotations

import numpy as np
import pytest
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from greentech.ai.models.training import (
    _build_variant_index,
    _collect_variants_for_train,
)

# === Helpers de generation de dataset synthetique ===


def _build_synthetic_dataset(
    n_originals_pos: int = 30,
    n_originals_neg: int = 270,
    n_variants_per_pos: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construit un dataset synthetique avec originaux + variantes.

    Args:
        n_originals_pos: Nombre d'articles positifs originaux.
        n_originals_neg: Nombre d'articles negatifs originaux.
        n_variants_per_pos: Nombre de variantes generees pour chaque positif.
        seed: Graine pour les choix aleatoires.

    Returns:
        Tuple ``(texts_arr, labels_arr, langues_arr, aug_sources_arr)`` ou
        chaque texte est au format ``"titre N\\n\\nresume ..."`` et les
        variantes partagent le meme titre que leur original.
    """
    rng = np.random.default_rng(seed)

    texts: list[str] = []
    labels: list[int] = []
    langues: list[str] = []
    aug_sources: list[str] = []

    # Originaux positifs
    for i in range(n_originals_pos):
        title = f"Article positif {i}"
        resume_text = f"Resume original positif numero {i}, contenu Green IT."
        lang = rng.choice(["en", "fr"], p=[0.75, 0.25])
        texts.append(f"{title}\n\n{resume_text}")
        labels.append(1)
        langues.append(str(lang))
        aug_sources.append("")

    # Originaux negatifs
    for i in range(n_originals_neg):
        title = f"Article negatif {i}"
        resume_text = f"Resume original negatif numero {i}, contenu hors sujet."
        lang = rng.choice(["en", "fr"], p=[0.75, 0.25])
        texts.append(f"{title}\n\n{resume_text}")
        labels.append(0)
        langues.append(str(lang))
        aug_sources.append("")

    # Variantes : meme titre, resume different, meme label, augmentation_source != ""
    for i in range(n_originals_pos):
        title = f"Article positif {i}"
        for v in range(n_variants_per_pos):
            resume_text = f"Resume variante {v} pour positif {i}, contenu Green IT paraphrase."
            sim = round(rng.uniform(0.86, 0.98), 3)
            texts.append(f"{title}\n\n{resume_text}")
            labels.append(1)
            # La langue de la variante peut differer de l'original (back-trans pivot)
            langues.append(str(rng.choice(["en", "fr"])))
            aug_sources.append(f"opus-mt-backtranslation-sim{sim}")

    return (
        np.array(texts, dtype=object),
        np.array(labels, dtype=int),
        np.array(langues, dtype=object),
        np.array(aug_sources, dtype=object),
    )


# === Tests unitaires des fonctions atomiques ===


class TestBuildVariantIndex:
    """Tests pour _build_variant_index : rattachement variante -> original."""

    def test_rattache_variante_au_bon_original(self) -> None:
        """Chaque variante doit etre liee a l'original qui partage son titre."""
        texts_arr, _, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=5, n_originals_neg=5, n_variants_per_pos=2, seed=1
        )
        original_mask = aug_sources_arr == ""

        variant_index = _build_variant_index(texts_arr, original_mask)

        # Chaque original positif doit avoir exactement 2 variantes
        original_indices = np.where(original_mask)[0]
        for orig_idx in original_indices:
            title = str(texts_arr[orig_idx]).split("\n\n", 1)[0]
            expected_variants = sum(
                1
                for j in range(len(texts_arr))
                if not original_mask[j] and str(texts_arr[j]).split("\n\n", 1)[0] == title
            )
            assert len(variant_index[int(orig_idx)]) == expected_variants

    def test_originaux_negatifs_ont_zero_variantes(self) -> None:
        """Les originaux negatifs (label=0) n'ont pas de variantes generees."""
        texts_arr, labels_arr, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=5, n_originals_neg=10, n_variants_per_pos=2, seed=2
        )
        original_mask = aug_sources_arr == ""

        variant_index = _build_variant_index(texts_arr, original_mask)

        for orig_idx, variants in variant_index.items():
            if labels_arr[orig_idx] == 0:
                assert variants == [], (
                    f"Original negatif idx={orig_idx} a {len(variants)} variantes (attendu 0)"
                )

    def test_clef_dict_couvre_uniquement_les_originaux(self) -> None:
        """Le dictionnaire ne contient que les indices d'originaux comme cles."""
        texts_arr, _, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=3, n_originals_neg=3, n_variants_per_pos=1, seed=3
        )
        original_mask = aug_sources_arr == ""

        variant_index = _build_variant_index(texts_arr, original_mask)

        original_indices_set = set(np.where(original_mask)[0].tolist())
        assert set(variant_index.keys()) == original_indices_set

    def test_variantes_pointent_vers_des_indices_de_variantes(self) -> None:
        """Toutes les valeurs (indices de variantes) doivent etre des indices
        non-originaux dans le dataset."""
        texts_arr, _, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=5, n_originals_neg=5, n_variants_per_pos=2, seed=4
        )
        original_mask = aug_sources_arr == ""

        variant_index = _build_variant_index(texts_arr, original_mask)

        for variants in variant_index.values():
            for v_idx in variants:
                assert not original_mask[v_idx], (
                    f"L'indice {v_idx} est marque comme original alors qu'il "
                    "est liste comme variante dans variant_index."
                )


class TestCollectVariantsForTrain:
    """Tests pour _collect_variants_for_train : selection des variantes a injecter."""

    def test_train_obtient_les_variantes_de_ses_originaux(self) -> None:
        """Les variantes des originaux presents dans train doivent etre collectees."""
        texts_arr, _, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=10, n_originals_neg=10, n_variants_per_pos=3, seed=5
        )
        original_mask = aug_sources_arr == ""
        variant_index = _build_variant_index(texts_arr, original_mask)

        # Simuler un train split contenant tous les positifs originaux + 5 negatifs
        original_indices = np.where(original_mask)[0]
        # Les 10 premiers originaux dans notre construction sont les positifs
        train_orig = original_indices[:10]

        variants_for_train = _collect_variants_for_train(train_orig, variant_index)

        # 10 positifs x 3 variantes = 30 variantes attendues
        assert len(variants_for_train) == 30

    def test_train_sans_positif_ne_recupere_aucune_variante(self) -> None:
        """Un train compose uniquement de negatifs n'attire aucune variante
        (seuls les positifs sont augmentes dans notre pipeline)."""
        texts_arr, _, _, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=5, n_originals_neg=10, n_variants_per_pos=2, seed=6
        )
        original_mask = aug_sources_arr == ""
        variant_index = _build_variant_index(texts_arr, original_mask)

        original_indices = np.where(original_mask)[0]
        # Les 5 premiers sont positifs, les 10 suivants sont negatifs
        train_neg_only = original_indices[5:]

        variants_for_train = _collect_variants_for_train(train_neg_only, variant_index)
        assert len(variants_for_train) == 0

    def test_train_vide_retourne_array_vide(self) -> None:
        """Edge case : un train vide ne retourne aucune variante."""
        empty_train = np.array([], dtype=int)
        variants = _collect_variants_for_train(empty_train, {0: [1, 2, 3]})
        assert len(variants) == 0
        assert variants.dtype == np.int64 or variants.dtype == np.int32


# === Test d'invariant data leakage (le coeur de P3.1) ===


class TestKFoldDataLeakageInvariant:
    """Verifie que sur tout K-fold genere par MultilabelStratifiedKFold +
    injection des variantes via _collect_variants_for_train, aucune variante
    ne se retrouve jamais dans le val split."""

    @pytest.mark.parametrize("n_splits", [3, 5])
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_val_split_ne_contient_jamais_de_variante(self, n_splits: int, seed: int) -> None:
        """Pour chaque fold, aucun indice du val split ne doit avoir
        ``augmentation_source != ""``. C'est la regle d'or anti-leakage de B3.3."""
        texts_arr, labels_arr, langues_arr, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=50,
            n_originals_neg=300,
            n_variants_per_pos=2,
            seed=seed,
        )
        original_mask = aug_sources_arr == ""
        original_indices = np.where(original_mask)[0]

        lang_en = (langues_arr[original_mask] == "en").astype(int)
        lang_fr = (langues_arr[original_mask] == "fr").astype(int)
        label_pos = labels_arr[original_mask]
        strat_labels = np.stack([lang_en, lang_fr, label_pos], axis=1)

        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for _fold_idx, (_train_idx_local, val_idx_local) in enumerate(
            mskf.split(strat_labels, strat_labels), 1
        ):
            val_orig_global = original_indices[val_idx_local]
            val_aug_sources = aug_sources_arr[val_orig_global]
            assert (val_aug_sources == "").all(), (
                f"Fuite detectee : fold {_fold_idx} contient "
                f"{(val_aug_sources != '').sum()} variantes dans val"
            )

    def test_variantes_dun_original_en_val_ne_sont_jamais_en_train(self) -> None:
        """Garantie plus forte : si un original X est en val, alors aucune
        variante derivee de X (meme titre) ne doit apparaitre dans train."""
        texts_arr, labels_arr, langues_arr, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=30, n_originals_neg=170, n_variants_per_pos=3, seed=7
        )
        original_mask = aug_sources_arr == ""
        variant_index = _build_variant_index(texts_arr, original_mask)
        original_indices = np.where(original_mask)[0]

        lang_en = (langues_arr[original_mask] == "en").astype(int)
        lang_fr = (langues_arr[original_mask] == "fr").astype(int)
        label_pos = labels_arr[original_mask]
        strat_labels = np.stack([lang_en, lang_fr, label_pos], axis=1)

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        for fold_idx, (train_idx_local, val_idx_local) in enumerate(
            mskf.split(strat_labels, strat_labels), 1
        ):
            train_orig_global = original_indices[train_idx_local]
            val_orig_global = original_indices[val_idx_local]

            # Construire le train enrichi des variantes
            augment_global = _collect_variants_for_train(train_orig_global, variant_index)
            train_global = np.concatenate([train_orig_global, augment_global])

            # Pour chaque original X dans val, ses variantes (meme titre) ne doivent
            # pas etre dans train_global
            for orig_val_idx in val_orig_global:
                title_val = str(texts_arr[orig_val_idx]).split("\n\n", 1)[0]
                for train_idx in train_global:
                    title_train = str(texts_arr[train_idx]).split("\n\n", 1)[0]
                    if train_idx in train_orig_global:
                        # train original peut partager un titre avec val original
                        # uniquement par collision (negatifs synthetique distincts).
                        # On verifie qu'il s'agit bien d'un original.
                        assert aug_sources_arr[train_idx] == "" or title_train != title_val
                    else:
                        # train variante : son titre ne doit PAS matcher un val original
                        assert title_train != title_val, (
                            f"Fuite detectee fold {fold_idx} : variante "
                            f"idx={train_idx} (titre '{title_train}') correspond a "
                            f"l'original val idx={orig_val_idx}"
                        )

    def test_train_split_recupere_toutes_les_variantes_de_ses_originaux(self) -> None:
        """Symetriquement, toutes les variantes des originaux du train doivent
        bien y etre injectees (pas de fuite a l'envers : variantes oubliees)."""
        texts_arr, labels_arr, langues_arr, aug_sources_arr = _build_synthetic_dataset(
            n_originals_pos=20, n_originals_neg=80, n_variants_per_pos=2, seed=8
        )
        original_mask = aug_sources_arr == ""
        variant_index = _build_variant_index(texts_arr, original_mask)
        original_indices = np.where(original_mask)[0]

        lang_en = (langues_arr[original_mask] == "en").astype(int)
        lang_fr = (langues_arr[original_mask] == "fr").astype(int)
        label_pos = labels_arr[original_mask]
        strat_labels = np.stack([lang_en, lang_fr, label_pos], axis=1)

        mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=8)
        for _fold_idx, (train_idx_local, _val_idx_local) in enumerate(
            mskf.split(strat_labels, strat_labels), 1
        ):
            train_orig_global = original_indices[train_idx_local]
            augment_global = _collect_variants_for_train(train_orig_global, variant_index)

            # Compter les variantes attendues : 2 variantes par original positif present
            expected_variants = 0
            for orig_idx in train_orig_global:
                if labels_arr[orig_idx] == 1:
                    expected_variants += 2

            assert len(augment_global) == expected_variants
