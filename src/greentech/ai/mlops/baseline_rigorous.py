"""Baselines rigoureuses pour mesurer les capacites des modeles pre-entraines.

Remplace la baseline "tete random" historique (methodologiquement faible : on
mesurait l'alea de l'initialisation de la tete de classification, pas la
qualite des features du backbone) par trois methodes academiquement defendables :

1. **Linear probing** : on extrait les embeddings du backbone (mean pooling
   sur les hidden states), puis on entraine UNIQUEMENT une regression
   logistique sklearn ``class_weight='balanced'``. Le backbone reste GELE.
   Cela mesure la qualite intrinseque des representations du backbone,
   independamment du choix de tete - methode standard en SSL (CLIP, DINO,
   SimCLR, MoCo).

2. **Zero-shot NLI** (encoder mDeBERTa) : le modele a ete pre-entraine ou
   fine-tune sur NLI/XNLI. On utilise ``pipeline("zero-shot-classification")``
   qui projette le texte sur les labels candidats via entailment NLI -
   methode standard pour les encoders multilingues.

3. **Zero-shot prompting** (decoder Qwen3-4B-Instruct) : on envoie un prompt
   instructif au LLM et on parse sa reponse. Mesure la capacite de
   raisonnement contextuel du modele instruct.

Toutes les baselines ecrivent les memes metriques (MCC, F1, Precision,
Recall, ...) pour permettre une comparaison directe avec le futur
benchmark post-entrainement (P5.1).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger

from greentech.ai.models.baseline import compute_classification_metrics
from greentech.config import get_settings

if TYPE_CHECKING:
    from greentech.ai.models.baseline import BaselineResult


@dataclass(frozen=True)
class RigorousBaselineResult:
    """Resultat d'une baseline rigoureuse, format identique a BaselineResult."""

    model_name: str
    method: str  # 'linear_probing' | 'zero_shot_nli' | 'zero_shot_prompt'
    metrics: dict[str, float | int]
    predictions: list[int]
    latencies_ms: list[float]
    n_articles: int
    duration_seconds: float


# =============================================================================
# Linear probing
# =============================================================================


def _pick_device() -> str:
    """Detecte le meilleur device disponible (ROCm/CUDA > CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(f"GPU detecte : {torch.cuda.get_device_name(0)}")
            return "cuda"
    except ImportError:
        pass
    logger.warning("Aucun GPU detecte : linear probing sur CPU (extraction lente)")
    return "cpu"


def _extract_embeddings(
    model_name: str,
    texts: Iterable[str],
    *,
    max_length: int = 512,
    batch_size: int = 8,
    pooling: Literal["mean", "cls", "last_token"] = "mean",
) -> tuple[np.ndarray, list[float]]:
    """Extrait les embeddings de chaque texte en gelant le backbone.

    Args:
        model_name: Identifiant HF du modele (encoder ou decoder).
        texts: Iterable de textes a encoder.
        max_length: Longueur max des sequences tokenizees.
        batch_size: Taille de batch d'inference (impact memoire seulement).
        pooling: Strategie de pooling sur les hidden states :
            - ``"mean"`` : moyenne ponderee par attention_mask (recommandee
              pour encoders ET decoders, robuste, standard SBERT)
            - ``"cls"`` : prend le token [CLS] (specifique encoders BERT-style)
            - ``"last_token"`` : prend le dernier token non-padding (specifique
              decoders causals)

    Returns:
        Tuple ``(embeddings, latencies_ms)`` ou embeddings est de forme
        ``(n_texts, hidden_dim)`` et latencies une liste de N float.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    settings = get_settings()
    device = _pick_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    hf_token = settings.huggingface_token or None

    logger.info(
        f"Chargement de {model_name} pour extraction features "
        f"(device={device}, dtype={dtype}, pooling={pooling})"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=hf_token, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        token=hf_token,
        trust_remote_code=False,
    )
    model.to(torch.device(device))
    model.eval()

    texts_list = list(texts)
    n = len(texts_list)
    hidden_dim = model.config.hidden_size
    embeddings = np.zeros((n, hidden_dim), dtype=np.float32)
    latencies_ms: list[float] = []

    logger.info(f"Extraction features : {n} articles, batch_size={batch_size}")

    for start in range(0, n, batch_size):
        batch_texts = texts_list[start : start + batch_size]
        t0 = time.perf_counter()
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(torch.device(device)) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # outputs.last_hidden_state shape : (batch, seq_len, hidden_dim)
        hidden = outputs.last_hidden_state

        if pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "cls":
            pooled = hidden[:, 0, :]
        elif pooling == "last_token":
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), seq_lens]
        else:
            msg = f"Pooling inconnu : {pooling}"
            raise ValueError(msg)

        embeddings[start : start + len(batch_texts)] = pooled.float().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(batch_texts), 1)
        latencies_ms.extend([latency_ms] * len(batch_texts))

        if (start // batch_size) % 100 == 0 and start > 0:
            logger.info(f"  Avancement : {start}/{n} articles")

    # Liberation memoire GPU avant la classification (qui se fait sur CPU)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return embeddings, latencies_ms


def run_linear_probing(
    model_name: str,
    texts: list[str],
    labels: list[int],
    *,
    pooling: Literal["mean", "cls", "last_token"] = "mean",
    cv_folds: int = 5,
    seed: int = 42,
) -> RigorousBaselineResult:
    """Linear probing : extrait les features et entraine une regression logistique.

    Args:
        model_name: Identifiant HF du modele backbone (encoder ou decoder).
        texts: Textes du dataset.
        labels: Labels binaires (0 = Non Green IT, 1 = Green IT).
        pooling: Strategie de pooling pour l'extraction des embeddings.
        cv_folds: Nombre de folds pour la cross-validation de la regression
            logistique (5 = standard). Les predictions retournees sont les
            predictions out-of-fold (chaque article predit par un modele
            qui ne l'a pas vu).
        seed: Graine RNG pour la reproductibilite.

    Returns:
        Resultat avec metriques, predictions out-of-fold et latences.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    start_total = time.perf_counter()
    logger.info("=" * 70)
    logger.info(f"LINEAR PROBING : {model_name} (pooling={pooling})")
    logger.info("=" * 70)

    # 1) Extraction des features (modele frozen)
    embeddings, latencies_extraction = _extract_embeddings(
        model_name, texts, pooling=pooling
    )
    logger.info(f"Embeddings extraits : shape={embeddings.shape}")

    # 2) Cross-validated logistic regression avec class_weight='balanced'
    #    Sert a obtenir des predictions out-of-fold rigoureuses.
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    n = len(labels)
    predictions = np.zeros(n, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels), 1):
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(embeddings[train_idx], np.array(labels)[train_idx])
        predictions[val_idx] = clf.predict(embeddings[val_idx])
        logger.info(f"  Fold {fold_idx}/{cv_folds} : LogReg entrainee")

    metrics = compute_classification_metrics(
        labels, predictions.tolist(), latencies_extraction
    )
    duration = time.perf_counter() - start_total
    logger.info(
        f"Linear probing termine en {duration:.1f}s : "
        f"MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
        f"Recall={metrics['recall']:.4f}"
    )

    return RigorousBaselineResult(
        model_name=model_name,
        method="linear_probing",
        metrics=metrics,
        predictions=predictions.tolist(),
        latencies_ms=latencies_extraction,
        n_articles=n,
        duration_seconds=duration,
    )


# =============================================================================
# Zero-shot NLI (encoders multilingues type mDeBERTa)
# =============================================================================


def run_zero_shot_nli(
    model_name: str,
    texts: list[str],
    labels: list[int],
    *,
    candidate_labels: tuple[str, str] = (
        "An article about Green IT, sustainable computing, energy-efficient infrastructure, or eco-friendly digital technology",
        "An article unrelated to Green IT or sustainable computing",
    ),
    batch_size: int = 8,
) -> RigorousBaselineResult:
    """Zero-shot classification via pipeline NLI HuggingFace.

    Cette baseline exploite l'entrainement NLI implicite du modele pour
    projeter chaque texte sur les labels candidats via entailment. C'est
    le mode "zero-shot" standard pour mDeBERTa et XLM-RoBERTa.

    Args:
        model_name: Identifiant HF d'un modele NLI (idealement mDeBERTa-NLI
            ou similaire fine-tune NLI ; un modele base accepte aussi mais
            performe moins bien sans NLI fine-tune).
        texts: Textes a classifier.
        labels: Labels reels (0/1).
        candidate_labels: Tuple de descriptions textuelles des classes,
            ordre ``(label=1, label=0)``. Phrases en anglais privilegiees
            car mDeBERTa-NLI est ancre sur XNLI.
        batch_size: Taille de batch pour le pipeline.

    Returns:
        Resultat structure compatible avec le rapport consolide.
    """
    import torch
    from transformers import pipeline

    start_total = time.perf_counter()
    device = _pick_device()
    logger.info("=" * 70)
    logger.info(f"ZERO-SHOT NLI : {model_name} (device={device})")
    logger.info("=" * 70)
    logger.info(f"Label Green IT (1) : '{candidate_labels[0]}'")
    logger.info(f"Label Non Green IT (0) : '{candidate_labels[1]}'")

    clf = pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=0 if device == "cuda" else -1,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    n = len(texts)
    predictions: list[int] = []
    latencies_ms: list[float] = []

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        t0 = time.perf_counter()
        results = clf(batch, candidate_labels=list(candidate_labels))
        if isinstance(results, dict):  # single text case
            results = [results]
        for r in results:
            top_label = r["labels"][0]
            pred = 1 if top_label == candidate_labels[0] else 0
            predictions.append(pred)
        latency_ms = (time.perf_counter() - t0) * 1000 / max(len(batch), 1)
        latencies_ms.extend([latency_ms] * len(batch))

        if (start // batch_size) % 50 == 0 and start > 0:
            logger.info(f"  Avancement : {start}/{n} articles")

    metrics = compute_classification_metrics(labels, predictions, latencies_ms)
    duration = time.perf_counter() - start_total
    logger.info(
        f"Zero-shot NLI termine en {duration:.1f}s : "
        f"MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
        f"Recall={metrics['recall']:.4f}"
    )

    return RigorousBaselineResult(
        model_name=model_name,
        method="zero_shot_nli",
        metrics=metrics,
        predictions=predictions,
        latencies_ms=latencies_ms,
        n_articles=n,
        duration_seconds=duration,
    )


# =============================================================================
# Zero-shot prompting (decoder instruct type Qwen3-4B-Instruct)
# =============================================================================

_ZERO_SHOT_PROMPT_TEMPLATE = (
    "Tu es un expert en Green IT. Determine si l'article suivant aborde de "
    "facon substantielle un sujet Green IT (sobriete numerique, "
    "eco-conception logicielle, efficacite energetique du materiel IT, "
    "data center durable, e-waste, IA frugale).\n\n"
    "Article :\n{text}\n\n"
    "Reponds UNIQUEMENT par 'OUI' (Green IT) ou 'NON' (Non Green IT)."
)


async def run_zero_shot_prompt(
    model_name: str,
    texts: list[str],
    labels: list[int],
    *,
    max_text_chars: int = 2000,
) -> RigorousBaselineResult:
    """Zero-shot prompting via un decoder LLM instruct (Qwen3-4B-Instruct).

    Envoie un prompt explicite au LLM et parse la reponse (OUI/NON).
    Utilise le ``LocalQwenClient`` du projet pour exploiter le GPU local
    sans dependre du quota HF Serverless.

    **Note** : fonction async (le LLM client utilise ``async def``). Doit
    etre appelee avec ``await`` depuis un event loop existant. Ne pas
    encapsuler dans ``asyncio.run()`` si l'appelant est deja async (cf.
    ``benchmark_baseline_rigorous.py`` qui orchestre 4 baselines en
    ``async def run()``).

    Args:
        model_name: Identifiant HF d'un modele instruct.
        texts: Textes a classifier.
        labels: Labels reels (0/1).
        max_text_chars: Tronque chaque article a cette longueur pour
            controler la latence et eviter les depassements de contexte.

    Returns:
        Resultat avec metriques + predictions.
    """
    from greentech.ai.services.llm_local import LocalQwenClient

    start_total = time.perf_counter()
    logger.info("=" * 70)
    logger.info(f"ZERO-SHOT PROMPT : {model_name}")
    logger.info("=" * 70)

    client = LocalQwenClient()
    client.load(model_name=model_name)

    preds: list[int] = []
    lats: list[float] = []
    for idx, text in enumerate(texts):
        t0 = time.perf_counter()
        truncated = text[:max_text_chars]
        prompt = _ZERO_SHOT_PROMPT_TEMPLATE.format(text=truncated)
        messages = [{"role": "user", "content": prompt}]
        try:
            response = await client.chat_completion(
                messages=messages,
                max_tokens=8,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()
            # Parse simple : "oui" → 1, sinon → 0
            pred = 1 if raw.startswith("oui") or "oui" in raw[:10] else 0
        except Exception as exc:
            logger.warning(f"  Article {idx} : erreur LLM ({exc}), pred=0")
            pred = 0
        preds.append(pred)
        lats.append((time.perf_counter() - t0) * 1000)
        if (idx + 1) % 250 == 0:
            logger.info(f"  Avancement : {idx + 1}/{len(texts)} articles")

    metrics = compute_classification_metrics(labels, preds, lats)
    duration = time.perf_counter() - start_total
    logger.info(
        f"Zero-shot prompt termine en {duration:.1f}s : "
        f"MCC={metrics['mcc']:.4f}, F1={metrics['f1']:.4f}, "
        f"Recall={metrics['recall']:.4f}"
    )

    return RigorousBaselineResult(
        model_name=model_name,
        method="zero_shot_prompt",
        metrics=metrics,
        predictions=preds,
        latencies_ms=lats,
        n_articles=len(texts),
        duration_seconds=duration,
    )


# =============================================================================
# Conversion vers BaselineResult pour reutilisation tracking existant
# =============================================================================


def to_baseline_result(rigorous: RigorousBaselineResult) -> BaselineResult:
    """Adapte un RigorousBaselineResult au format BaselineResult standard.

    Permet de reutiliser ``track_baseline()`` (MLflow + JSON + Prometheus)
    sans dupliquer le code de persistance.
    """
    from greentech.ai.models.baseline import BaselineResult

    # On encode la methode dans le nom pour distinguer les runs MLflow.
    safe_model_name = f"{rigorous.model_name} ({rigorous.method})"
    return BaselineResult(
        model_name=safe_model_name,
        metrics=rigorous.metrics,
        predictions=rigorous.predictions,
        latencies_ms=rigorous.latencies_ms,
        n_articles=rigorous.n_articles,
        duration_seconds=rigorous.duration_seconds,
    )
