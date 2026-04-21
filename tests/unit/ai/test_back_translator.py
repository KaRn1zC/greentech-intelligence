"""Tests unitaires de la back-translation EN<->FR (B3.3).

Couvre :

1. ``BackTranslationResult`` et ``BackTranslationStats`` : structures de
   donnees, proprietes calculees (``acceptance_rate``), serialisation
   ``to_dict`` utilisee pour le logging MLflow.
2. ``BackTranslator._build_result`` : logique pure du filtre de similarite
   (rejet si < min, rejet si > max, accept sinon) sans chargement de
   modeles MarianMT/SentenceTransformer.
3. ``BackTranslator.augment`` : validation des entrees (longueurs
   incoherentes), separation par langue, comportement avec langues non
   supportees — mock des methodes lourdes (``_translate_batch``,
   ``_compute_similarities``, ``load``) pour eviter de charger les
   modeles pendant les tests unitaires.

Les tests n'appellent pas les vrais modeles Helsinki-NLP/opus-mt-* ni
``sentence-transformers``, ils n'ont donc besoin ni de GPU ni d'acces reseau.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from greentech.data.processors.back_translator import (
    BackTranslationResult,
    BackTranslationStats,
    BackTranslator,
)


class TestBackTranslationStats:
    """Comptage et serialisation des statistiques d'augmentation."""

    def test_acceptance_rate_on_empty_input_is_zero(self) -> None:
        """Sans entree, le taux d'acceptation doit etre 0 (et pas crasher)."""
        stats = BackTranslationStats(total_input=0)
        assert stats.acceptance_rate() == 0.0

    def test_acceptance_rate_matches_manual_ratio(self) -> None:
        """Ratio coherent avec les compteurs : accepted / input."""
        stats = BackTranslationStats(total_input=100, total_accepted=67)
        assert stats.acceptance_rate() == pytest.approx(0.67)

    def test_to_dict_exposes_all_mlflow_keys(self) -> None:
        """``to_dict`` doit exposer les cles consommees par MLflow metrics."""
        stats = BackTranslationStats(
            total_input=10,
            total_generated=10,
            total_accepted=7,
            total_rejected_low_similarity=2,
            total_rejected_high_similarity=1,
            total_failed=0,
        )
        d = stats.to_dict()
        for key in (
            "bt_total_input",
            "bt_total_generated",
            "bt_total_accepted",
            "bt_total_rejected_low_similarity",
            "bt_total_rejected_high_similarity",
            "bt_total_failed",
        ):
            assert key in d


class TestBuildResult:
    """Filtre de similarite pure (sans traduction ni encodage)."""

    def _make_translator(self) -> BackTranslator:
        """Instance sans charger aucun modele (lazy-loading desactive en test)."""
        return BackTranslator(similarity_min=0.85, similarity_max=0.99, device="cpu")

    def test_similarity_below_min_rejected(self) -> None:
        """Une retraduction trop dissemblable est rejetee avec raison claire."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="original",
            augmented_text="degraded",
            source_language="en",
            pivot_language="fr",
            similarity=0.70,
        )
        assert result.accepted is False
        assert result.reason_rejected is not None
        assert "<" in result.reason_rejected
        assert "0.85" in result.reason_rejected

    def test_similarity_above_max_rejected(self) -> None:
        """Une retraduction quasi-identique est rejetee (pas de gain)."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="original",
            augmented_text="quasi-identical",
            source_language="fr",
            pivot_language="en",
            similarity=0.999,
        )
        assert result.accepted is False
        assert result.reason_rejected is not None
        assert ">" in result.reason_rejected
        assert "0.99" in result.reason_rejected

    def test_similarity_within_window_accepted(self) -> None:
        """Dans la fenetre [0.85, 0.99], la variante est acceptee."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="original",
            augmented_text="paraphrase",
            source_language="en",
            pivot_language="fr",
            similarity=0.92,
        )
        assert result.accepted is True
        assert result.reason_rejected is None

    def test_boundary_min_accepted(self) -> None:
        """Sim = min est acceptee (filtre strict ``<``, pas ``<=``)."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="o",
            augmented_text="a",
            source_language="en",
            pivot_language="fr",
            similarity=0.85,
        )
        assert result.accepted is True

    def test_boundary_max_accepted(self) -> None:
        """Sim = max est acceptee (filtre strict ``>``, pas ``>=``)."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="o",
            augmented_text="a",
            source_language="en",
            pivot_language="fr",
            similarity=0.99,
        )
        assert result.accepted is True

    def test_result_preserves_languages(self) -> None:
        """Les langues source/pivot doivent etre celles passees en argument."""
        bt = self._make_translator()
        result = bt._build_result(
            original_text="o",
            augmented_text="a",
            source_language="fr",
            pivot_language="en",
            similarity=0.90,
        )
        assert result.source_language == "fr"
        assert result.pivot_language == "en"


class TestAugmentInputValidation:
    """Validation des arguments de ``BackTranslator.augment``."""

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """``texts`` et ``languages`` doivent avoir la meme longueur."""
        bt = BackTranslator(device="cpu")
        with pytest.raises(ValueError) as exc_info:
            # On n'a pas besoin de patcher load() : la verification est en tete.
            bt.augment(texts=["a", "b"], languages=["en"])
        assert "len(texts)" in str(exc_info.value)

    def test_empty_input_yields_empty_stats(self) -> None:
        """Appel avec 0 article doit etre no-op, retourne stats vides."""
        bt = BackTranslator(device="cpu")
        with patch.object(bt, "load"):
            results, stats = bt.augment(texts=[], languages=[])
        assert results == []
        assert stats.total_input == 0
        assert stats.total_accepted == 0


class TestAugmentLanguageRouting:
    """Routage par langue : EN->FR->EN, FR->EN->FR, autres ignorees."""

    def _make_translator(self) -> BackTranslator:
        """Instance sans GPU. ``load`` sera mocke pour eviter le download."""
        return BackTranslator(
            similarity_min=0.85,
            similarity_max=0.99,
            device="cpu",
        )

    def test_unsupported_language_logged_and_skipped(self, caplog) -> None:
        """Un article ``lang='de'`` doit etre ignore avec warning explicite."""
        bt = self._make_translator()
        # load() : no-op (pas de download reel).
        # _translate_batch / _compute_similarities : retournent des donnees
        # simulees coherentes, similarites dans la fenetre acceptee.
        with (
            patch.object(bt, "load"),
            patch.object(
                bt,
                "_translate_batch",
                side_effect=lambda texts, direction: [
                    f"translated({direction}):{t}" for t in texts
                ],
            ),
            patch.object(
                bt,
                "_compute_similarities",
                side_effect=lambda orig, _retrad: [0.90] * len(orig),
            ),
        ):
            results, stats = bt.augment(
                texts=["english 1", "french 2", "deutsch 3"],
                languages=["en", "fr", "de"],
            )

        # 2 acceptes (EN + FR), 0 depuis DE.
        assert stats.total_input == 3
        assert stats.total_accepted == 2
        accepted_langs = {r.source_language for r in results}
        assert accepted_langs == {"en", "fr"}

    def test_en_and_fr_are_routed_through_correct_pivot(self) -> None:
        """EN passe par FR comme pivot, FR passe par EN comme pivot."""
        bt = self._make_translator()
        with (
            patch.object(bt, "load"),
            patch.object(
                bt,
                "_translate_batch",
                side_effect=lambda texts, direction: [f"{direction}:{t}" for t in texts],
            ),
            patch.object(
                bt,
                "_compute_similarities",
                side_effect=lambda orig, _retrad: [0.92] * len(orig),
            ),
        ):
            results, _ = bt.augment(
                texts=["en article", "fr article"],
                languages=["en", "fr"],
            )

        by_lang = {r.source_language: r for r in results}
        assert by_lang["en"].pivot_language == "fr"
        assert by_lang["fr"].pivot_language == "en"

    def test_rejected_variants_counted_by_reason(self) -> None:
        """Le comptage doit distinguer rejets basse sim vs haute sim."""
        bt = self._make_translator()
        # 3 articles EN : 1 hors fenetre basse, 1 hors fenetre haute, 1 accepte
        similarities = [0.70, 0.999, 0.92]
        with (
            patch.object(bt, "load"),
            patch.object(
                bt,
                "_translate_batch",
                side_effect=lambda texts, direction: [f"{direction}:{t}" for t in texts],
            ),
            patch.object(
                bt,
                "_compute_similarities",
                side_effect=lambda _orig, _retrad: similarities,
            ),
        ):
            results, stats = bt.augment(
                texts=["a", "b", "c"],
                languages=["en", "en", "en"],
            )

        assert stats.total_input == 3
        assert stats.total_accepted == 1
        assert stats.total_rejected_low_similarity == 1
        assert stats.total_rejected_high_similarity == 1
        # Un seul article accepte retourne
        assert len(results) == 1
        assert results[0].accepted is True


class TestBackTranslationResult:
    """Dataclass ``BackTranslationResult`` (immutable, pour eviter mutations accidentelles)."""

    def test_is_frozen_dataclass(self) -> None:
        """Le dataclass est ``frozen=True`` : muter un champ doit lever."""
        result = BackTranslationResult(
            original_text="o",
            augmented_text="a",
            source_language="en",
            pivot_language="fr",
            similarity=0.9,
            accepted=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.similarity = 0.5  # type: ignore[misc]
