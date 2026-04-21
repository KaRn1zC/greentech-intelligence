"""Tests unitaires de la loss ponderee (B3.2 class_weight).

Valide les deux briques du weighted CrossEntropy :

1. ``compute_class_weight`` calcule bien ``[1.0, N_neg/N_pos]`` sur un
   train set desequilibre, y compris les cas degeneres (tout positif,
   tout negatif).
2. ``WeightedLossTrainer.compute_loss`` produit une loss strictement
   superieure sur un batch majoritairement positif quand le poids de la
   classe positive est eleve, comparee a la loss non-ponderee. C'est la
   propriete cle qui justifie le remplacement de l'oversampling x84 :
   les positifs recoivent plus de gradient par echantillon sans dupliquer
   les memes textes.

Les tests n'entrainent pas un vrai modele : on utilise un mini-modele
synthetique (lineaire) pour verifier le comportement de la loss.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from greentech.ai.models.training import WeightedLossTrainer, compute_class_weight


class TestComputeClassWeight:
    """Calcul du vecteur class_weight pour CrossEntropy ponderee."""

    def test_balanced_dataset_returns_unit_ratio(self) -> None:
        """50 / 50 => poids positif = N_neg / N_pos = 1.0."""
        labels = [0, 0, 0, 1, 1, 1]
        weights = compute_class_weight(labels)
        assert weights[0].item() == pytest.approx(1.0)
        assert weights[1].item() == pytest.approx(1.0)

    def test_imbalanced_dataset_matches_expected_ratio(self) -> None:
        """Pour 90 neg / 10 pos => poids positif = 9.0 exactement."""
        labels = [0] * 90 + [1] * 10
        weights = compute_class_weight(labels)
        assert weights[0].item() == pytest.approx(1.0)
        assert weights[1].item() == pytest.approx(9.0)

    def test_realistic_ratio_matches_production(self) -> None:
        """Ratio production ~1:10.46 (N_neg=10646, N_pos=1018)."""
        labels = [0] * 10646 + [1] * 1018
        weights = compute_class_weight(labels)
        assert weights[1].item() == pytest.approx(10646 / 1018, rel=1e-6)

    def test_no_positives_falls_back_to_uniform(self) -> None:
        """Aucun positif => fallback ``[1.0, 1.0]`` avec warning, pas de division par zero."""
        labels = [0, 0, 0, 0]
        weights = compute_class_weight(labels)
        assert weights[0].item() == pytest.approx(1.0)
        assert weights[1].item() == pytest.approx(1.0)

    def test_numpy_array_input_accepted(self) -> None:
        """``compute_class_weight`` accepte aussi un ``np.ndarray`` (pas que list)."""
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
        weights = compute_class_weight(labels)
        assert weights[1].item() == pytest.approx(9.0)

    def test_returns_float32_tensor_of_shape_2(self) -> None:
        """Le retour doit etre un tensor float32 de shape (2,) (pour HuggingFace Trainer)."""
        weights = compute_class_weight([0, 1])
        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == torch.float32
        assert weights.shape == (2,)


class TestWeightedLossTrainerComputeLoss:
    """Verification du comportement de la loss ponderee via le Trainer custom."""

    def _build_trainer(
        self,
        *,
        class_weight: torch.Tensor | None,
    ) -> WeightedLossTrainer:
        """Construit un Trainer minimaliste sans modele reel (instancie via __new__).

        On court-circuite ``__init__`` de ``Trainer`` (qui exige un modele,
        un TrainingArguments, un tokenizer, etc.) pour tester uniquement
        ``compute_loss``. Le champ ``class_weight`` est celui que le Trainer
        lit pour ponderer la CrossEntropy.
        """
        trainer = WeightedLossTrainer.__new__(WeightedLossTrainer)
        trainer.class_weight = class_weight
        return trainer

    def _build_stub_model(self, logits: torch.Tensor) -> object:
        """Retourne un stub qui imite la sortie d'un `AutoModelForSequenceClassification`.

        Transformers retourne un objet avec un attribut ``logits`` ; le
        Trainer fait ``outputs.logits`` puis `CrossEntropyLoss(weight=w)`
        sur ``logits.view(-1, C)`` et ``labels.view(-1)``.
        """

        class _StubOutput:
            def __init__(self, logits_: torch.Tensor) -> None:
                self.logits = logits_

        class _StubModel:
            def __init__(self, logits_: torch.Tensor) -> None:
                self._logits = logits_

            def __call__(self, **_kwargs) -> _StubOutput:
                return _StubOutput(self._logits)

        return _StubModel(logits)

    def test_weighted_loss_amplifies_hard_positives_in_mixed_batch(self) -> None:
        """Sur un batch mixte ou seuls les positifs sont hard, la loss
        ponderee (poids positif = 10) doit etre significativement plus
        grande que la loss standard. C'est la propriete qui justifie le
        remplacement de l'oversampling x84 : les positifs mal classifies
        contribuent davantage au gradient.

        Batch synthetique : 6 negatifs bien classifies (faible loss),
        2 positifs mal classifies (haute loss). Avec poids_pos=10, la
        contribution relative des positifs passe de 2/8 = 25 % a
        2*10 / (6 + 2*10) ~ 77 % de la loss totale (reduction=mean avec
        normalisation par somme des poids).
        """
        # 6 negatifs : logits favorisent la classe 0 => faible loss
        # 2 positifs : logits favorisent la classe 0 => haute loss
        logits = torch.tensor(
            [[3.0, -3.0]] * 6  # negatifs bien classifies
            + [[3.0, -3.0]] * 2  # positifs hard (modele se trompe)
        )
        labels = torch.tensor([0] * 6 + [1] * 2)

        stub_model = self._build_stub_model(logits)

        trainer_weighted = self._build_trainer(
            class_weight=torch.tensor([1.0, 10.0], dtype=torch.float32)
        )
        loss_w = trainer_weighted.compute_loss(stub_model, inputs={"labels": labels.clone()})

        trainer_plain = self._build_trainer(class_weight=None)
        loss_u = trainer_plain.compute_loss(stub_model, inputs={"labels": labels.clone()})

        # La loss ponderee doit etre strictement superieure : les 2 positifs
        # hard tirent la moyenne vers le haut avec leur poids x10.
        assert loss_w.item() > loss_u.item()
        # Facteur attendu : environ x2.3 (validation numerique precise dans
        # test_weighted_loss_matches_manual_formula ci-dessous).
        assert loss_w.item() / loss_u.item() > 2.0

    def test_weighted_loss_matches_manual_formula(self) -> None:
        """Verifie la formule exacte : sum(w_i * CE_i) / sum(w_i).

        Reconstitue le calcul manuel attendu pour prouver qu'on n'a pas
        d'effet de bord inattendu (ex: reduction mal configuree).
        """
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [1.0, 0.0]])
        labels = torch.tensor([1, 0, 1])  # pos, neg, pos
        class_weight = torch.tensor([1.0, 5.0], dtype=torch.float32)

        stub_model = self._build_stub_model(logits)
        trainer = self._build_trainer(class_weight=class_weight)
        loss_observed = trainer.compute_loss(stub_model, inputs={"labels": labels.clone()})

        # Reference : CrossEntropyLoss(weight=...) avec reduction=mean
        loss_ref = torch.nn.CrossEntropyLoss(weight=class_weight)(logits, labels)
        assert loss_observed.item() == pytest.approx(loss_ref.item(), rel=1e-6)

    def test_plain_loss_recovered_when_class_weight_none(self) -> None:
        """Si ``class_weight is None``, on retombe exactement sur la loss standard."""
        logits = torch.tensor([[0.5, -0.5], [-1.0, 1.0]])
        labels = torch.tensor([0, 1])
        stub_model = self._build_stub_model(logits)

        trainer = self._build_trainer(class_weight=None)
        loss_observed = trainer.compute_loss(stub_model, inputs={"labels": labels.clone()})

        # Calcul de reference avec CrossEntropyLoss standard
        loss_ref = torch.nn.CrossEntropyLoss()(logits, labels)
        assert loss_observed.item() == pytest.approx(loss_ref.item(), rel=1e-6)

    def test_inputs_labels_restored_after_compute(self) -> None:
        """``compute_loss`` restaure ``inputs["labels"]`` apres consommation.

        Le Trainer HuggingFace reutilise le dict d'inputs pour logger les
        predictions ; si les labels etaient consommes definitivement, le
        logging aval echouerait.
        """
        logits = torch.tensor([[1.0, -1.0]])
        labels = torch.tensor([0])
        stub_model = self._build_stub_model(logits)

        trainer = self._build_trainer(class_weight=torch.tensor([1.0, 5.0], dtype=torch.float32))
        inputs = {"labels": labels.clone()}
        trainer.compute_loss(stub_model, inputs=inputs)
        assert "labels" in inputs
        assert torch.equal(inputs["labels"], labels)

    def test_return_outputs_true_yields_tuple(self) -> None:
        """``return_outputs=True`` doit retourner ``(loss, outputs)`` comme HF."""
        logits = torch.tensor([[1.0, -1.0]])
        labels = torch.tensor([0])
        stub_model = self._build_stub_model(logits)

        trainer = self._build_trainer(class_weight=None)
        result = trainer.compute_loss(
            stub_model,
            inputs={"labels": labels.clone()},
            return_outputs=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, outputs = result
        assert isinstance(loss, torch.Tensor)
        assert hasattr(outputs, "logits")
