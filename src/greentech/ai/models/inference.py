"""Module d'inference pour le modele de classification Green IT.

Charge le modele gagnant (selectionne apres le benchmark Champion vs
Challengers) et fournit une interface simple pour classifier les articles en
production. Met a jour la base PostgreSQL avec les resultats de classification.

Le modele de production est une copie du vainqueur du benchmark
(DeBERTa, mDeBERTa, Qwen2.5-3B+LoRA, Llama 3.2 3B+LoRA ou Qwen3-4B+LoRA) dans
``models/production/``. La detection du type de modele (complet vs adaptateur
LoRA) est automatique via la presence du fichier ``adapter_config.json`` :

- Si ``adapter_config.json`` est present, on lit ``base_model_name_or_path``
  pour reconstruire le bon classifieur LoRA. Si le base model correspond a
  un Qwen3-4B (presence de ``qwen3-4b`` ou ``qwen3_4b`` dans le nom), on
  charge ``Qwen3Classifier`` (avec hyperparametres et
  target_modules adaptes), sinon on utilise le ``LoRAClassifier``
  generique pour Llama/Qwen2.5.
- Si le fichier est absent, on traite le dossier comme un modele complet
  (``DeBERTaClassifier`` pour DeBERTa EN-only ou
  ``MDeBERTaClassifier`` si le dossier s'appelle ``mdeberta``
  ou que ``config.json`` designe un mdeberta).

Calibration post-training (B3 protocole unifie avril 2026)
-----------------------------------------------------------
Si les fichiers ``temperature.json`` et ``optimal_threshold.json`` sont
presents a cote du modele, ils sont automatiquement appliques :

1. **Temperature scaling** : les logits sont divises par T avant softmax
   pour corriger la sur-confiance typique des modeles fine-tunes sur
   dataset desequilibre.
2. **Threshold tuning** : la decision binaire utilise le seuil optimal
   (typiquement entre 0.2 et 0.5 selon le modele) au lieu de 0.5 par defaut.

Si les fichiers sont absents, fallback silencieux sur le comportement
historique (argmax + seuil implicite 0.5) pour preserver la compatibilite
avec les modeles deja en production.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from loguru import logger
from sqlalchemy import select, update

from greentech.ai.mlops.calibration import apply_calibration, load_calibration
from greentech.ai.models.classifier import (
    BaseClassifier,
    LabelGreenIT,
    PredictionResult,
    TrainingConfig,
)
from greentech.config import BASE_DIR
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

# Chemin par defaut du modele de production
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "production"

# Instance globale du classifieur (lazy loading)
_classifier: BaseClassifier | None = None
_calibration_temperature: float | None = None
_calibration_threshold: float | None = None


async def get_classifier(model_path: Path | None = None) -> BaseClassifier:
    """Retourne le classifieur de production (singleton lazy-loaded).

    Detecte automatiquement le type de modele :
    - Si adapter_config.json est present → modele LoRA (LoRAClassifier)
    - Sinon → modele complet (DeBERTaClassifier)

    Charge le modele au premier appel, puis reutilise l'instance.

    Args:
        model_path: Chemin vers le modele (defaut: models/production).

    Returns:
        Instance du classifieur prete pour l'inference.

    Raises:
        FileNotFoundError: Si le modele n'est pas trouve.
        ValueError: Si la config adapter LoRA est invalide.
    """
    global _classifier, _calibration_temperature, _calibration_threshold  # noqa: PLW0603

    if _classifier is not None:
        return _classifier

    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        msg = (
            f"Modele de production introuvable : {path}. "
            "Lancez l'entrainement avec : uv run python -m greentech.ai.models.training"
        )
        raise FileNotFoundError(msg)

    adapter_config_path = path / "adapter_config.json"

    # On instancie puis on appelle load() AVANT d'assigner au cache global.
    # Sinon, un load() qui leve une exception laisse derriere lui une instance
    # avec model=None, et tout appel ulterieur retourne ce singleton "vide" qui
    # echoue eternellement avec "Modele non charge".
    if adapter_config_path.exists():
        # Adaptateur LoRA detecte : on dispatche vers la bonne sous-classe
        # de LoRAClassifier selon la famille du base model.
        from greentech.ai.models.training import (
            LoRAClassifier,
            Qwen3Classifier,
        )

        with open(adapter_config_path) as f:
            adapter_meta = json.load(f)

        base_model_name = adapter_meta.get("base_model_name_or_path")
        if not base_model_name:
            msg = f"adapter_config.json dans {path} ne contient pas 'base_model_name_or_path'"
            raise ValueError(msg)

        # Qwen3-4B a ses propres target_modules LoRA (all-linear depuis avril
        # 2026) et une config optimisee (batch/seq length + gradient
        # checkpointing) : on selectionne la sous-classe dediee pour
        # preserver la coherence entre entrainement et inference. Le match
        # est volontairement strict sur `qwen3-4b` / `qwen3_4b` pour ne pas
        # capturer par erreur les anciens adaptateurs `Qwen3.5-4B`.
        name_lower = base_model_name.lower()
        is_qwen3 = "qwen3-4b" in name_lower or "qwen3_4b" in name_lower
        config = TrainingConfig(nom_modele=base_model_name)
        candidate: BaseClassifier = (
            Qwen3Classifier(config) if is_qwen3 else LoRAClassifier(config)
        )
        candidate.load(path)
        logger.info(f"Modele LoRA ({base_model_name}) charge depuis {path}")
    else:
        # Modele complet : detecter mDeBERTa vs DeBERTa EN via le dossier et
        # le config.json. On dispatche vers MDeBERTaClassifier si
        # le modele est mdeberta pour appliquer les bons hyperparametres
        # (notamment max_length=384 au lieu de 512) a l'inference.
        from greentech.ai.models.training import (
            DeBERTaClassifier,
            MDeBERTaClassifier,
        )

        is_mdeberta = _detect_mdeberta(path)
        config = TrainingConfig(nom_modele=str(path))
        candidate = MDeBERTaClassifier(config) if is_mdeberta else DeBERTaClassifier(config)
        candidate.load(path)
        logger.info(
            f"Modele complet ({'mDeBERTa' if is_mdeberta else 'DeBERTa/autre'}) "
            f"charge depuis {path}"
        )

    # Charger la calibration post-training si disponible (temperature.json
    # et optimal_threshold.json sauvegardes par train_with_unified_protocol).
    # Fallback silencieux sur T=1.0 / seuil=0.5 si les fichiers sont absents.
    _calibration_temperature, _calibration_threshold = load_calibration(path)
    if _calibration_temperature is not None or _calibration_threshold is not None:
        logger.info(
            f"Calibration chargee : T={_calibration_temperature}, "
            f"seuil={_calibration_threshold}"
        )
    else:
        logger.info(
            "Aucune calibration trouvee (pas de temperature.json / optimal_threshold.json), "
            "fallback T=1.0 seuil=0.5"
        )

    _classifier = candidate
    return _classifier


def _detect_mdeberta(model_path: Path) -> bool:
    """Heuristique pour detecter un modele mDeBERTa-v3.

    Trois signaux cumulatifs (un suffit) :

    1. Le nom du dossier contient ``mdeberta``
    2. ``config.json`` contient un champ ``_name_or_path`` avec ``mdeberta``
    3. Le model_type interne est ``deberta-v2`` (architecture DeBERTa-v3)
       avec un vocab_size compatible multilingue (> 200 000)
    """
    if "mdeberta" in model_path.name.lower():
        return True
    config_json = model_path / "config.json"
    if not config_json.exists():
        return False
    try:
        data = json.loads(config_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    name = str(data.get("_name_or_path", "")).lower()
    if "mdeberta" in name:
        return True
    vocab_size = data.get("vocab_size", 0)
    model_type = data.get("model_type", "")
    return model_type in ("deberta-v2", "deberta") and vocab_size > 200_000


def _apply_calibration_to_result(result: PredictionResult) -> PredictionResult:
    """Reconstruit un ``PredictionResult`` avec T + threshold si disponibles.

    Si aucune calibration n'est chargee (T et threshold tous deux ``None``),
    retourne l'original inchange. Sinon :

    1. Applique ``T`` sur les logits reconstruits depuis ``proba_positive``
       via ``apply_calibration`` du module de calibration.
    2. Decide du label via le seuil optimal au lieu de 0.5.
    3. Recalcule ``score_confiance`` pour refleter la probabilite calibree
       de la classe retenue (coherent pour l'UI).

    Si ``proba_positive`` est ``None`` dans le result (classifieur externe),
    on ne peut pas calibrer et on retourne l'original.
    """
    if _calibration_temperature is None and _calibration_threshold is None:
        return result
    if result.proba_positive is None:
        return result

    # Reconstruire un logits (1, 2) depuis la proba positive pour apply_calibration
    import numpy as np

    proba_pos = float(result.proba_positive)
    proba_pos_clipped = min(max(proba_pos, 1e-7), 1.0 - 1e-7)
    logit_pos = float(np.log(proba_pos_clipped / (1.0 - proba_pos_clipped)))
    logits = np.array([[-logit_pos, logit_pos]], dtype=np.float32)

    calibrated_probs, calibrated_preds = apply_calibration(
        logits,
        temperature=_calibration_temperature,
        threshold=_calibration_threshold,
    )
    predicted_label = int(calibrated_preds[0])
    calibrated_pos_proba = float(calibrated_probs[0])
    calibrated_confidence = (
        calibrated_pos_proba if predicted_label == LabelGreenIT.GREEN.value
        else 1.0 - calibrated_pos_proba
    )

    return PredictionResult(
        label=LabelGreenIT(predicted_label),
        score_confiance=calibrated_confidence,
        temps_ms=result.temps_ms,
        modele=result.modele,
        proba_positive=calibrated_pos_proba,
    )


async def classify_article(article_id: int) -> PredictionResult:
    """Classifie un article et stocke le résultat en base.

    Lit l'article depuis PostgreSQL, exécute l'inférence sur le **résumé
    de classification** (colonne ``articles.resume``), puis met à jour
    les colonnes ``est_green_it``, ``score_confiance``,
    ``modele_classification`` et ``date_analyse``.

    Le classifieur a été entraîné sur la concaténation
    ``titre + "\\n\\n" + resume`` : on reproduit strictement cette
    représentation à l'inférence pour éviter toute dérive de distribution.
    Si le résumé n'a pas été généré au préalable (colonne ``resume``
    à NULL), l'appelant doit invoquer ``summarize_article`` avant d'appeler
    cette fonction — c'est le rôle de ``_run_analysis`` dans la route
    ``/analyze`` et du pipeline batch ``summarize-classif``.

    Args:
        article_id: Identifiant de l'article en base.

    Returns:
        Résultat de la classification.

    Raises:
        ValueError: Si l'article n'existe pas, n'a pas de contenu ou n'a
            pas encore de résumé de classification.
    """
    classifier = await get_classifier()

    async with async_session_factory() as session:
        stmt = select(Article).where(Article.id_article == article_id)
        result = await session.execute(stmt)
        article = result.scalar_one_or_none()

        if article is None:
            msg = f"Article id={article_id} introuvable"
            raise ValueError(msg)

        if not article.contenu:
            msg = f"Article id={article_id} sans contenu"
            raise ValueError(msg)

        if not article.resume:
            msg = (
                f"Article id={article_id} sans resume de classification. "
                "Lancer summarize_article() avant classify_article() "
                "(ou le batch scripts/generate_classification_summaries.py)."
            )
            raise ValueError(msg)

        # Inference : on reproduit strictement la feature d'entrainement
        # titre + "\n\n" + resume pour eviter toute derive de distribution.
        texte_pour_classification = f"{article.titre}\n\n{article.resume}"
        raw_prediction = await classifier.predict(texte_pour_classification)

        # Applique le temperature scaling et le seuil optimal si charges
        # (fallback silencieux sur le comportement historique argmax/0.5
        # si aucune calibration n'est associee au modele de production).
        prediction = _apply_calibration_to_result(raw_prediction)

        # Mise à jour en base
        from datetime import datetime

        stmt_update = (
            update(Article)
            .where(Article.id_article == article_id)
            .values(
                est_green_it=prediction.est_green_it,
                score_confiance=prediction.score_confiance,
                modele_classification=prediction.modele,
                date_analyse=datetime.now(UTC),
            )
        )
        await session.execute(stmt_update)
        await session.commit()

        logger.info(
            f"Article id={article_id} classifié : "
            f"{'Green IT' if prediction.est_green_it else 'Non Green IT'} "
            f"(confiance={prediction.score_confiance:.2%}, {prediction.temps_ms}ms)"
        )

        return prediction


async def classify_batch(*, limit: int = 100, force: bool = False) -> list[PredictionResult]:
    """Classifie un lot d'articles deja resumes.

    Ne selectionne que les articles disposant d'un ``resume`` non vide :
    la classification doit s'executer sur le resume (feature d'entrainement),
    pas sur le contenu brut. Lancer ``summarize_all_articles_for_classification``
    (ou le script batch ``generate_classification_summaries.py``) avant
    d'invoquer cette fonction sur un nouveau corpus.

    Args:
        limit: Nombre maximum d'articles a traiter.
        force: Si True, re-classifie aussi les articles deja analyses.

    Returns:
        Liste des resultats de classification.
    """
    async with async_session_factory() as session:
        stmt = (
            select(Article.id_article)
            .where(Article.contenu.isnot(None))
            .where(Article.resume.isnot(None))
        )
        if not force:
            stmt = stmt.where(Article.est_green_it.is_(None))
        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        article_ids = [row[0] for row in result.all()]

    if not article_ids:
        logger.info("Aucun article a classifier (pas de resume disponible ou deja classifies)")
        return []

    logger.info(f"Classification de {len(article_ids)} articles...")
    results = []

    for article_id in article_ids:
        prediction = await classify_article(article_id)
        results.append(prediction)

    green = sum(1 for r in results if r.est_green_it)
    non_green = len(results) - green
    logger.info(f"Batch termine : {green} Green IT / {non_green} Non Green IT")

    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(classify_batch())
