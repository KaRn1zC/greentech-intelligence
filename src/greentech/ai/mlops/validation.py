"""Validation automatisée du modèle avec Deepchecks.

Suite de tests pour vérifier l'intégrité du modèle de classification
Green IT : data leakage, biais, robustesse au bruit, et conformité
des données d'entraînement.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from deepchecks.nlp import TextData
from deepchecks.nlp.suites import full_suite, model_evaluation, train_test_validation
from loguru import logger


@dataclass
class ValidationReport:
    """Résultat d'une suite de validation Deepchecks.

    Attributes:
        nom_suite: Nom de la suite exécutée.
        total_checks: Nombre total de vérifications.
        passed: Nombre de vérifications réussies.
        failed: Nombre de vérifications échouées.
        warnings: Nombre d'avertissements.
        chemin_rapport: Chemin du rapport HTML généré.
    """

    nom_suite: str
    total_checks: int
    passed: int
    failed: int
    warnings: int
    chemin_rapport: Path | None = None


def create_text_dataset(
    df: pd.DataFrame,
    *,
    text_col: str = "contenu",
    label_col: str = "est_green_it",
    nom: str = "dataset",
) -> TextData:
    """Crée un objet TextData Deepchecks depuis un DataFrame.

    Args:
        df: DataFrame contenant les textes et labels.
        text_col: Nom de la colonne contenant le texte.
        label_col: Nom de la colonne contenant les labels.
        nom: Nom du dataset pour les rapports.

    Returns:
        Objet TextData compatible avec les suites Deepchecks.
    """
    raw_text = df[text_col].tolist()
    label = df[label_col].tolist() if label_col in df.columns else None

    dataset = TextData(
        raw_text=raw_text,
        label=label,
        task_type="text_classification",
        name=nom,
    )

    logger.info(
        f"TextData '{nom}' créé : {len(raw_text)} textes, labels={'oui' if label else 'non'}"
    )
    return dataset


def run_train_test_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    output_dir: str | Path = "reports",
) -> ValidationReport:
    """Exécute la suite de validation train/test de Deepchecks.

    Vérifie la qualité des données d'entraînement et de test :
    fuite de données (data leakage), distribution des labels,
    longueur des textes, etc.

    Args:
        train_df: DataFrame d'entraînement (colonnes: contenu, est_green_it).
        test_df: DataFrame de test.
        output_dir: Dossier de sortie pour le rapport HTML.

    Returns:
        Rapport de validation avec les compteurs de résultats.
    """
    logger.info("Exécution de la suite train/test validation...")

    train_data = create_text_dataset(train_df, nom="train")
    test_data = create_text_dataset(test_df, nom="test")

    suite = train_test_validation()
    result = suite.run(train_dataset=train_data, test_dataset=test_data)

    return _save_report(result, "train_test_validation", output_dir)


def run_model_evaluation(
    test_df: pd.DataFrame,
    predictions: list[Any],
    *,
    probabilities: list[list[float]] | None = None,
    output_dir: str | Path = "reports",
) -> ValidationReport:
    """Exécute la suite d'évaluation du modèle Deepchecks.

    Analyse les performances du modèle : matrice de confusion,
    métriques par classe, biais potentiels.

    Args:
        test_df: DataFrame de test (colonnes: contenu, est_green_it).
        predictions: Liste des prédictions du modèle.
        probabilities: Probabilités par classe (optionnel).
        output_dir: Dossier de sortie pour le rapport HTML.

    Returns:
        Rapport de validation avec les compteurs de résultats.
    """
    logger.info("Exécution de la suite d'évaluation du modèle...")

    test_data = create_text_dataset(test_df, nom="test")

    suite = model_evaluation()
    result = suite.run(
        train_dataset=test_data,
        test_dataset=test_data,
        train_predictions=predictions,
        test_predictions=predictions,
        train_probas=probabilities,
        test_probas=probabilities,
    )

    return _save_report(result, "model_evaluation", output_dir)


def run_full_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    predictions: list[Any] | None = None,
    output_dir: str | Path = "reports",
) -> ValidationReport:
    """Exécute la suite complète de validation Deepchecks.

    Combine validation des données et évaluation du modèle
    pour un rapport exhaustif.

    Args:
        train_df: DataFrame d'entraînement.
        test_df: DataFrame de test.
        predictions: Prédictions du modèle sur le jeu de test (optionnel).
        output_dir: Dossier de sortie pour le rapport HTML.

    Returns:
        Rapport de validation complet.
    """
    logger.info("Exécution de la suite complète de validation...")

    train_data = create_text_dataset(train_df, nom="train")
    test_data = create_text_dataset(test_df, nom="test")

    suite = full_suite()
    result = suite.run(
        train_dataset=train_data,
        test_dataset=test_data,
        train_predictions=predictions,
        test_predictions=predictions,
    )

    return _save_report(result, "full_validation", output_dir)


def _save_report(
    result: Any,
    nom_suite: str,
    output_dir: str | Path,
) -> ValidationReport:
    """Sauvegarde le résultat Deepchecks en HTML et retourne un rapport.

    Args:
        result: Résultat de la suite Deepchecks.
        nom_suite: Nom de la suite pour le fichier.
        output_dir: Dossier de sortie.

    Returns:
        Rapport structuré avec les compteurs.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"deepchecks_{nom_suite}.html"

    result.save_as_html(str(report_file))

    # Extraire les compteurs depuis les résultats
    passed = sum(1 for check in result.results if check.passed_conditions())
    total = len(result.results)
    failed_checks = [check for check in result.results if not check.passed_conditions()]
    failed = len(failed_checks)
    warnings = total - passed - failed

    report = ValidationReport(
        nom_suite=nom_suite,
        total_checks=total,
        passed=passed,
        failed=failed,
        warnings=warnings,
        chemin_rapport=report_file,
    )

    logger.info(
        f"Rapport '{nom_suite}' : {passed}/{total} OK, "
        f"{failed} échecs, {warnings} avertissements → {report_file}"
    )

    if failed_checks:
        for check in failed_checks:
            logger.warning(f"  ❌ {check.header}: {check.display}")

    return report
