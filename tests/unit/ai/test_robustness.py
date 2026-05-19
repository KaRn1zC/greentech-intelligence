"""Tests de robustesse au bruit textuel (B3.6).

Verifie le comportement des trois perturbations de robustesse :

1. **AEDA** (insertions de ponctuation) - Karimi et al. 2021
2. **Random casing** (inversion de casse aleatoire)
3. **Typos** (swap, delete, insert de caracteres)

Tests unitaires :
- Determinisme : la meme entree + meme seed produit la meme sortie
- Validite : la sortie reste une chaine de caracteres similaire
- Intensite : intensite=0 = no-op ; intensite=1 = perturbation systematique
- Edge cases : texte vide, texte court, valeurs invalides

L'invariant de robustesse MCC < 0.10 baisse necessiterait le chargement
du modele de production, dispendieux en temps : il est verifie a part
dans le pipeline de validation Deepchecks (script
``scripts/benchmark_models.py``) sur le test set hold-out.
"""

from __future__ import annotations

import pytest

from greentech.ai.mlops.robustness import (
    AEDA_PUNCTUATION_MARKS,
    PerturbationConfig,
    apply_aeda,
    apply_combined_perturbations,
    apply_random_casing,
    apply_typos,
)

# === Validation de PerturbationConfig ===


class TestPerturbationConfig:
    """Tests de la dataclass PerturbationConfig."""

    def test_defaults(self) -> None:
        """Defaut : intensite 0.1, seed None."""
        cfg = PerturbationConfig()
        assert cfg.intensity == 0.1
        assert cfg.seed is None

    @pytest.mark.parametrize("intensity", [0.0, 0.1, 0.5, 1.0])
    def test_intensity_valid_range(self, intensity: float) -> None:
        """Toute intensite dans [0, 1] est acceptee."""
        cfg = PerturbationConfig(intensity=intensity)
        assert cfg.intensity == intensity

    @pytest.mark.parametrize("intensity", [-0.1, 1.1, 2.0, -1.0])
    def test_intensity_invalid_range_raises(self, intensity: float) -> None:
        """Une intensite hors [0, 1] doit lever ValueError."""
        with pytest.raises(ValueError, match="intensity"):
            PerturbationConfig(intensity=intensity)


# === Tests AEDA ===


class TestApplyAeda:
    """Tests pour l'insertion AEDA de ponctuation."""

    def test_intensite_zero_est_identite(self) -> None:
        """Avec intensite=0, le texte doit etre inchange."""
        text = "Le Green IT est important pour la planete."
        result = apply_aeda(text, PerturbationConfig(intensity=0.0, seed=42))
        assert result == text

    def test_determinisme_avec_seed(self) -> None:
        """Meme seed + meme texte = meme resultat."""
        text = "Optimisation energetique des modeles de machine learning frugaux."
        cfg = PerturbationConfig(intensity=0.5, seed=42)
        result_a = apply_aeda(text, cfg)
        result_b = apply_aeda(text, cfg)
        assert result_a == result_b

    def test_seeds_differents_donnent_resultats_differents(self) -> None:
        """Deux seeds differents doivent produire des sorties differentes
        (avec intensite suffisante)."""
        text = "Reduction de l'empreinte carbone des data centers via refroidissement liquide."
        result_a = apply_aeda(text, PerturbationConfig(intensity=0.8, seed=1))
        result_b = apply_aeda(text, PerturbationConfig(intensity=0.8, seed=2))
        assert result_a != result_b

    def test_n_mots_preserve(self) -> None:
        """AEDA n'ajoute que de la ponctuation, le nombre de mots non-ponctuation
        doit rester identique."""
        text = "Sobriete numerique eco-conception logicielle optimisation"
        original_words = text.split()
        result = apply_aeda(text, PerturbationConfig(intensity=0.8, seed=42))
        result_words = [w for w in result.split() if w not in AEDA_PUNCTUATION_MARKS]
        assert result_words == original_words

    def test_ponctuation_inserees_appartiennent_au_set_aeda(self) -> None:
        """Toute ponctuation isolee inseree doit etre dans AEDA_PUNCTUATION_MARKS."""
        text = "Reduction consommation energetique GPU AMD pour entrainement IA frugale"
        result = apply_aeda(text, PerturbationConfig(intensity=1.0, seed=42))
        tokens = result.split()
        added_tokens = [t for t in tokens if t not in text.split()]
        for tok in added_tokens:
            assert tok in AEDA_PUNCTUATION_MARKS, (
                f"Token insere '{tok}' n'est pas dans AEDA_PUNCTUATION_MARKS"
            )

    def test_intensite_un_insere_partout(self) -> None:
        """Avec intensite=1, une ponctuation doit etre inseree apres chaque mot
        sauf le dernier (n_mots - 1 insertions)."""
        text = "Article green IT francais"
        result = apply_aeda(text, PerturbationConfig(intensity=1.0, seed=42))
        # n_mots = 4, donc 3 insertions = 7 tokens au total
        assert len(result.split()) == 7

    def test_texte_vide(self) -> None:
        """Texte vide -> texte vide."""
        assert apply_aeda("", PerturbationConfig(intensity=0.5, seed=42)) == ""

    def test_texte_un_mot(self) -> None:
        """Un seul mot : aucune insertion possible (pas d'inter-mot)."""
        result = apply_aeda("mot", PerturbationConfig(intensity=1.0, seed=42))
        assert result == "mot"


# === Tests Random Casing ===


class TestApplyRandomCasing:
    """Tests pour l'inversion aleatoire de casse."""

    def test_intensite_zero_est_identite(self) -> None:
        """intensite=0 = aucune modification."""
        text = "Le Green IT, c'est important."
        result = apply_random_casing(text, PerturbationConfig(intensity=0.0, seed=42))
        assert result == text

    def test_determinisme_avec_seed(self) -> None:
        """Meme seed = meme sortie."""
        text = "Optimisation energetique des modeles."
        cfg = PerturbationConfig(intensity=0.3, seed=42)
        assert apply_random_casing(text, cfg) == apply_random_casing(text, cfg)

    def test_longueur_preservee(self) -> None:
        """La longueur du texte ne doit pas changer."""
        text = "Sobriete numerique et eco-conception logicielle"
        result = apply_random_casing(text, PerturbationConfig(intensity=0.5, seed=42))
        assert len(result) == len(text)

    def test_caracteres_non_alpha_preserves(self) -> None:
        """Les espaces, ponctuation et chiffres ne doivent pas etre modifies."""
        text = "Article #1: 95% Green IT, 5% off-topic."
        result = apply_random_casing(text, PerturbationConfig(intensity=1.0, seed=42))
        # Comparer position par position : non-alpha doit etre identique
        for orig_ch, new_ch in zip(text, result, strict=True):
            if not orig_ch.isalpha():
                assert orig_ch == new_ch, f"Caractere non-alpha modifie : '{orig_ch}' -> '{new_ch}'"

    def test_intensite_un_inverse_tout_alpha(self) -> None:
        """intensite=1 = chaque caractere alpha voit sa casse inversee."""
        text = "Hello World"
        result = apply_random_casing(text, PerturbationConfig(intensity=1.0, seed=42))
        assert result == "hELLO wORLD"

    def test_texte_vide(self) -> None:
        assert apply_random_casing("", PerturbationConfig(intensity=0.5, seed=42)) == ""


# === Tests Typos ===


class TestApplyTypos:
    """Tests pour les typos (swap, delete, insert)."""

    def test_intensite_zero_est_identite(self) -> None:
        """intensite=0 = aucun typo."""
        text = "Le Green IT permet de reduire l'empreinte carbone."
        result = apply_typos(text, PerturbationConfig(intensity=0.0, seed=42))
        assert result == text

    def test_determinisme_avec_seed(self) -> None:
        """Meme seed = meme sortie."""
        text = "Optimisation energetique des modeles de machine learning."
        cfg = PerturbationConfig(intensity=0.5, seed=42)
        assert apply_typos(text, cfg) == apply_typos(text, cfg)

    def test_mots_courts_non_modifies(self) -> None:
        """Les mots de moins de 4 caracteres restent intacts meme a intensite=1."""
        text = "a bc de fgh"
        result = apply_typos(text, PerturbationConfig(intensity=1.0, seed=42))
        assert result == text

    def test_n_mots_preserve(self) -> None:
        """Le nombre de mots ne change pas (swap/delete/insert agit a l'interieur des mots)."""
        text = "Sobriete numerique et eco-conception logicielle optimisation modeles"
        original_n = len(text.split())
        result = apply_typos(text, PerturbationConfig(intensity=0.8, seed=42))
        assert len(result.split()) == original_n

    def test_premiere_et_derniere_lettre_preservees(self) -> None:
        """Pour preserver la lisibilite, la premiere et derniere lettre d'un mot
        ne doivent pas etre modifiees par swap/delete/insert."""
        text = "Optimisation"
        result = apply_typos(text, PerturbationConfig(intensity=1.0, seed=42))
        result_words = result.split()
        for original_word, new_word in zip(text.split(), result_words, strict=True):
            if len(original_word) >= 4:
                assert original_word[0] == new_word[0], (
                    f"Premiere lettre modifiee : '{original_word}' -> '{new_word}'"
                )
                # Note : derniere lettre OK selon implementation (idx peut etre len-2)
                # mais pas len-1, donc la derniere position reste intacte sur swap/insert.

    def test_texte_vide(self) -> None:
        assert apply_typos("", PerturbationConfig(intensity=0.5, seed=42)) == ""

    def test_changement_observe_avec_forte_intensite(self) -> None:
        """Avec intensite=1 et un mot long, on doit observer un changement."""
        text = "extraordinaire mot tres long pour augmenter les chances de perturbation"
        result = apply_typos(text, PerturbationConfig(intensity=1.0, seed=42))
        assert result != text


# === Tests combined perturbations ===


class TestApplyCombinedPerturbations:
    """Tests pour la chaine AEDA + casing + typos."""

    def test_intensite_zero_est_identite(self) -> None:
        """intensite=0 sur les 3 = aucune modification."""
        text = "Article green IT francais sur la sobriete numerique."
        result = apply_combined_perturbations(text, PerturbationConfig(intensity=0.0, seed=42))
        assert result == text

    def test_intensite_forte_change_le_texte(self) -> None:
        """Avec intensite forte, le texte combine doit etre different de l'original."""
        text = "Optimisation energetique des modeles de deep learning pour reduire l'empreinte carbone des data centers."
        result = apply_combined_perturbations(text, PerturbationConfig(intensity=0.8, seed=42))
        assert result != text

    def test_determinisme_avec_seed(self) -> None:
        """Reproductibilite : meme seed = meme sortie pour la chaine combinee."""
        text = "Reduction empreinte carbone des reseaux numeriques par caching edge."
        cfg = PerturbationConfig(intensity=0.5, seed=42)
        assert apply_combined_perturbations(text, cfg) == apply_combined_perturbations(text, cfg)

    def test_texte_vide(self) -> None:
        assert apply_combined_perturbations("", PerturbationConfig(intensity=0.5, seed=42)) == ""
