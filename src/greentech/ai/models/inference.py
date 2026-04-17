"""Module d'inference pour le modele de classification Green IT.

Charge le modele gagnant (selectionne apres le benchmark Champion vs Challengers)
et fournit une interface simple pour classifier les articles en production.
Met a jour la base PostgreSQL avec les resultats de classification.

Le modele de production est une copie du vainqueur du benchmark
(DeBERTa, Qwen2.5-3B+LoRA, Llama 3.2 3B+LoRA ou Qwen3-4B+LoRA) dans
`models/production/`. La detection du type de modele (complet vs adaptateur
LoRA) est automatique via la presence du fichier ``adapter_config.json`` :

- Si ``adapter_config.json`` est present, on lit ``base_model_name_or_path``
  pour reconstruire le bon classifieur LoRA. Si le base model correspond a
  un Qwen3-4B (presence de ``qwen3-4b`` ou ``qwen3_4b`` dans le nom), on
  charge ``ChallengerQwen3Classifier`` (avec hyperparametres et
  target_modules adaptes), sinon on utilise le ``ChallengerClassifier``
  generique pour Llama/Qwen2.5.
- Si le fichier est absent, on traite le dossier comme un modele complet
  (``ChampionClassifier``, typiquement DeBERTa).

"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from loguru import logger
from sqlalchemy import select, update

from greentech.ai.models.classifier import BaseClassifier, PredictionResult, TrainingConfig
from greentech.config import BASE_DIR
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

# Chemin par défaut du modèle de production
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "production"

# Instance globale du classifieur (lazy loading)
_classifier: BaseClassifier | None = None


async def get_classifier(model_path: Path | None = None) -> BaseClassifier:
    """Retourne le classifieur de production (singleton lazy-loaded).

    Detecte automatiquement le type de modele :
    - Si adapter_config.json est present → modele LoRA (ChallengerClassifier)
    - Sinon → modele complet (ChampionClassifier)

    Charge le modele au premier appel, puis reutilise l'instance.

    Args:
        model_path: Chemin vers le modele (defaut: models/production).

    Returns:
        Instance du classifieur prete pour l'inference.

    Raises:
        FileNotFoundError: Si le modele n'est pas trouve.
        ValueError: Si la config adapter LoRA est invalide.
    """
    global _classifier  # noqa: PLW0603

    if _classifier is not None:
        return _classifier

    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        msg = (
            f"Modèle de production introuvable : {path}. "
            "Lancez l'entraînement avec : uv run python -m greentech.ai.models.training"
        )
        raise FileNotFoundError(msg)

    adapter_config_path = path / "adapter_config.json"

    # On instancie puis on appelle load() AVANT d'assigner au cache global.
    # Sinon, un load() qui leve une exception laisse derriere lui une instance
    # avec model=None, et tout appel ulterieur retourne ce singleton "vide" qui
    # echoue eternellement avec "Modele non charge".
    if adapter_config_path.exists():
        # Adaptateur LoRA detecte : on dispatche vers la bonne sous-classe
        # de ChallengerClassifier selon la famille du base model.
        from greentech.ai.models.training import (
            ChallengerClassifier,
            ChallengerQwen3Classifier,
        )

        with open(adapter_config_path) as f:
            adapter_meta = json.load(f)

        base_model_name = adapter_meta.get("base_model_name_or_path")
        if not base_model_name:
            msg = f"adapter_config.json dans {path} ne contient pas 'base_model_name_or_path'"
            raise ValueError(msg)

        # Qwen3-4B a ses propres target_modules LoRA (attention-only) et une
        # config optimisee (batch/seq length + gradient checkpointing) : on
        # selectionne la sous-classe dediee pour preserver la coherence entre
        # entrainement et inference. Le match est volontairement strict sur
        # `qwen3-4b` / `qwen3_4b` pour ne pas capturer par erreur les anciens
        # adaptateurs `Qwen3.5-4B` si jamais il en reste.
        name_lower = base_model_name.lower()
        is_qwen3 = "qwen3-4b" in name_lower or "qwen3_4b" in name_lower
        config = TrainingConfig(nom_modele=base_model_name)
        candidate: BaseClassifier = (
            ChallengerQwen3Classifier(config) if is_qwen3 else ChallengerClassifier(config)
        )
        candidate.load(path)
        logger.info(f"Modèle LoRA ({base_model_name}) chargé depuis {path}")
    else:
        # Modele complet → ChampionClassifier
        from greentech.ai.models.training import ChampionClassifier

        config = TrainingConfig(nom_modele=str(path))
        candidate = ChampionClassifier(config)
        candidate.load(path)
        logger.info(f"Modèle complet chargé depuis {path}")

    _classifier = candidate
    return _classifier


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
        prediction = await classifier.predict(texte_pour_classification)

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
