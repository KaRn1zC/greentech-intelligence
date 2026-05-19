"""Back-translation EN<->FR via Helsinki-NLP/opus-mt (MarianMT).

Augmente la classe minoritaire d'un classifieur desequilibre en generant
des paraphrases automatiques via traduction aller-retour. Chaque article
positif EN passe par FR puis revient en EN (et symetriquement pour FR).
Les micro-variations introduites par les modeles MarianMT (synonymes,
ordre des mots) produisent un texte semantiquement equivalent mais
lexicalement different, avec le meme label.

Pourquoi MarianMT plutot qu'un LLM generique ?
----------------------------------------------
1. **Specialise traduction** : entraine sur corpus paralleles OPUS,
   scores BLEU ~38 EN-FR, superieur aux LLM generaux pour la traduction pure.
2. **Rapide** : 75M parametres par direction, ~100 phrases/s sur RX 7900 XTX.
3. **Deterministe** (greedy decoding) : reproductible run apres run,
   essentiel pour versionner le dataset augmente avec DVC.
4. **Leger** : ~150 Mo VRAM par modele, laisse la place pour d'autres taches.
5. **Green IT** : 75M params vs 4B pour Qwen = 50x moins d'empreinte carbone
   pour une tache simple.

Pipeline par article positif
-----------------------------
1. Determiner la langue source (``en`` ou ``fr``, issue du champ
   ``articles.langue``).
2. Traduire ``resume`` vers la langue pivot (opposee).
3. Retraduire depuis la langue pivot vers la langue source originale.
4. Filtrer par similarite cosine (sentence-transformers) entre original
   et retraduit. Rejeter si hors [0.85, 0.99] (trop different = bruit,
   trop proche = quasi-duplicata inutile).
5. Garder le couple (titre original, resume_backtranslated, label=1).

Les variantes ainsi generees portent ``augmentation_source =
'opus-mt-backtranslation'`` pour pouvoir etre tracees et exclues
systematiquement du val/test split lors du K-fold (anti-leakage).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from loguru import logger
from transformers import MarianMTModel, MarianTokenizer

# Modeles MarianMT sur le Hub. ~150 Mo chacun, telecharges au premier usage.
OPUS_MT_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
OPUS_MT_FR_EN = "Helsinki-NLP/opus-mt-fr-en"

# Modele de similarite cross-langues pour le filtre qualite.
# ~470 Mo, supporte 50+ langues (EN, FR inclus).
SIMILARITY_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Bornes de similarite cosine pour accepter une variante.
# < 0.85 : trop different de l'original (le modele de trad a halluci).
# > 0.99 : quasi-identique (copy-paste inutile, pas de gain d'augmentation).
DEFAULT_SIMILARITY_MIN = 0.85
DEFAULT_SIMILARITY_MAX = 0.99

# Taille max des sequences MarianMT. Les resumes font 150-220 mots ~= 300 tokens
# FR avec le tokenizer SentencePiece de opus-mt, on garde une marge a 512.
MARIAN_MAX_LENGTH = 512


SupportedLanguage = Literal["en", "fr"]


@dataclass(frozen=True)
class BackTranslationResult:
    """Resultat d'une back-translation pour un seul article.

    Attributes:
        original_text: Texte source (resume original).
        augmented_text: Texte retraduit apres le cycle source -> pivot -> source.
        source_language: Langue de l'article original ("en" ou "fr").
        pivot_language: Langue intermediaire utilisee (l'oppose de source).
        similarity: Similarite cosine entre original et retraduit, dans [-1, 1].
        accepted: ``True`` si la similarite est dans la fenetre d'acceptation.
        reason_rejected: Raison du rejet si ``accepted=False``, sinon ``None``.
    """

    original_text: str
    augmented_text: str
    source_language: SupportedLanguage
    pivot_language: SupportedLanguage
    similarity: float
    accepted: bool
    reason_rejected: str | None = None


@dataclass
class BackTranslationStats:
    """Statistiques agregees d'un batch d'augmentation.

    Attributes:
        total_input: Nombre d'articles positifs soumis en entree.
        total_generated: Nombre de variantes generees (= total_input si
            la traduction n'echoue jamais).
        total_accepted: Nombre de variantes acceptees par le filtre.
        total_rejected_low_similarity: Rejets pour similarite < min.
        total_rejected_high_similarity: Rejets pour similarite > max.
        total_failed: Nombre de traductions qui ont plante (exception capturee).
        durations_seconds: Temps total de traitement et breakdown par etape.
    """

    total_input: int = 0
    total_generated: int = 0
    total_accepted: int = 0
    total_rejected_low_similarity: int = 0
    total_rejected_high_similarity: int = 0
    total_failed: int = 0
    durations_seconds: dict[str, float] = field(default_factory=dict)

    def acceptance_rate(self) -> float:
        """Ratio d'articles acceptes, dans [0, 1]."""
        if self.total_input == 0:
            return 0.0
        return self.total_accepted / self.total_input

    def to_dict(self) -> dict[str, float]:
        """Serialise pour logging MLflow."""
        return {
            "bt_total_input": self.total_input,
            "bt_total_generated": self.total_generated,
            "bt_total_accepted": self.total_accepted,
            "bt_total_rejected_low_similarity": self.total_rejected_low_similarity,
            "bt_total_rejected_high_similarity": self.total_rejected_high_similarity,
            "bt_total_failed": self.total_failed,
            "bt_acceptance_rate": self.acceptance_rate(),
            **{f"bt_duration_{k}_s": v for k, v in self.durations_seconds.items()},
        }


class BackTranslator:
    """Augmentateur par back-translation EN<->FR via opus-mt.

    Usage typique::

        bt = BackTranslator()
        bt.load()
        variants, stats = bt.augment(
            texts=[article.resume for article in positives],
            languages=[article.langue for article in positives],
        )

    Les modeles sont lazy-loaded au premier appel de ``load()`` pour
    permettre d'instancier la classe sans exploser la VRAM quand on
    n'en a pas besoin (ex: tests unitaires).
    """

    def __init__(
        self,
        similarity_min: float = DEFAULT_SIMILARITY_MIN,
        similarity_max: float = DEFAULT_SIMILARITY_MAX,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        """Initialise le back-translator sans charger les modeles.

        Args:
            similarity_min: Seuil de similarite en-dessous duquel une
                variante est rejetee (suspicion d'hallucination).
            similarity_max: Seuil au-dessus duquel une variante est rejetee
                (quasi-duplicata, gain d'augmentation nul).
            batch_size: Taille de batch pour la traduction (16 est un bon
                compromis sur RX 7900 XTX 24 Go avec MarianMT 75M).
            device: "cuda", "cpu" ou None (auto-detect). ROCm expose cuda.
        """
        self.similarity_min = similarity_min
        self.similarity_max = similarity_max
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded au premier appel de load()
        self._en_fr_model: MarianMTModel | None = None
        self._en_fr_tokenizer: MarianTokenizer | None = None
        self._fr_en_model: MarianMTModel | None = None
        self._fr_en_tokenizer: MarianTokenizer | None = None
        self._sim_model = None  # SentenceTransformer, import paresseux

    def load(self) -> None:
        """Charge les 3 modeles (2 MarianMT + 1 SentenceTransformer).

        ~600 Mo de VRAM total sur GPU. Idempotent : appels multiples
        ne rechargent pas les modeles deja en memoire.
        """
        if self._en_fr_model is None:
            logger.info(f"Chargement MarianMT EN->FR ({OPUS_MT_EN_FR}) sur {self.device}")
            self._en_fr_tokenizer = MarianTokenizer.from_pretrained(OPUS_MT_EN_FR)
            self._en_fr_model = MarianMTModel.from_pretrained(OPUS_MT_EN_FR).to(self.device)
            self._en_fr_model.eval()

        if self._fr_en_model is None:
            logger.info(f"Chargement MarianMT FR->EN ({OPUS_MT_FR_EN}) sur {self.device}")
            self._fr_en_tokenizer = MarianTokenizer.from_pretrained(OPUS_MT_FR_EN)
            self._fr_en_model = MarianMTModel.from_pretrained(OPUS_MT_FR_EN).to(self.device)
            self._fr_en_model.eval()

        if self._sim_model is None:
            # Import tardif pour eviter de ralentir le module s'il n'est
            # pas utilise dans un contexte d'augmentation.
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Chargement SentenceTransformer ({SIMILARITY_MODEL_NAME}) sur {self.device}"
            )
            self._sim_model = SentenceTransformer(SIMILARITY_MODEL_NAME, device=self.device)

    def _translate_batch(
        self,
        texts: list[str],
        direction: Literal["en->fr", "fr->en"],
    ) -> list[str]:
        """Traduit un batch de textes dans la direction specifiee."""
        if direction == "en->fr":
            tokenizer = self._en_fr_tokenizer
            model = self._en_fr_model
        else:
            tokenizer = self._fr_en_tokenizer
            model = self._fr_en_model

        if tokenizer is None or model is None:
            msg = f"Modele {direction} non charge. Appelez .load() d'abord."
            raise RuntimeError(msg)

        results: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MARIAN_MAX_LENGTH,
            ).to(self.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=MARIAN_MAX_LENGTH,
                    num_beams=1,  # greedy pour reproductibilite
                )

            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded)

        return results

    def _compute_similarities(
        self,
        originals: list[str],
        retranslated: list[str],
    ) -> np.ndarray:
        """Calcule la similarite cosine article par article.

        Retourne un array de shape ``(n,)`` avec valeurs dans [-1, 1].
        """
        if self._sim_model is None:
            msg = "SentenceTransformer non charge. Appelez .load() d'abord."
            raise RuntimeError(msg)

        emb_orig = self._sim_model.encode(
            originals, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        emb_retr = self._sim_model.encode(
            retranslated,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Cosine similarity par ligne : normalise puis produit scalaire diagonal
        emb_orig_n = emb_orig / np.linalg.norm(emb_orig, axis=1, keepdims=True).clip(min=1e-9)
        emb_retr_n = emb_retr / np.linalg.norm(emb_retr, axis=1, keepdims=True).clip(min=1e-9)
        return (emb_orig_n * emb_retr_n).sum(axis=1)

    def augment(
        self,
        texts: list[str],
        languages: list[str],
    ) -> tuple[list[BackTranslationResult], BackTranslationStats]:
        """Applique la back-translation sur une liste d'articles.

        Args:
            texts: Resumes sources (150-220 mots typiques).
            languages: Langue de chaque resume ("en" ou "fr"). Les autres
                langues sont silencieusement ignorees (aucun pivot disponible).

        Returns:
            Couple ``(resultats, stats)``. Seuls les resultats acceptes
            doivent etre ajoutes au dataset d'entrainement (filtrer sur
            ``result.accepted``).

        Raises:
            ValueError: Si ``len(texts) != len(languages)``.
        """
        if len(texts) != len(languages):
            msg = (
                f"len(texts)={len(texts)} != len(languages)={len(languages)}. "
                "Chaque texte doit avoir sa langue associee."
            )
            raise ValueError(msg)

        self.load()

        stats = BackTranslationStats(total_input=len(texts))
        t_start = time.perf_counter()

        # Partition par langue source (on ne traite que en/fr, les autres
        # langues sont hors perimetre — le dataset final ne contient que
        # en/fr apres le nettoyage B2.9).
        en_indices = [i for i, lang in enumerate(languages) if lang == "en"]
        fr_indices = [i for i, lang in enumerate(languages) if lang == "fr"]
        other_indices = [i for i, lang in enumerate(languages) if lang not in ("en", "fr")]

        if other_indices:
            logger.warning(
                f"{len(other_indices)} articles dans une langue non supportee "
                f"par MarianMT EN<->FR, ignores : "
                f"{sorted({languages[i] for i in other_indices})}"
            )

        results: list[BackTranslationResult | None] = [None] * len(texts)

        # Cycle EN : en -> fr (pivot) -> en (retour)
        if en_indices:
            t_en = time.perf_counter()
            en_texts = [texts[i] for i in en_indices]
            en_to_fr = self._translate_batch(en_texts, "en->fr")
            en_retranslated = self._translate_batch(en_to_fr, "fr->en")
            en_similarities = self._compute_similarities(en_texts, en_retranslated)

            for idx, i in enumerate(en_indices):
                results[i] = self._build_result(
                    original_text=en_texts[idx],
                    augmented_text=en_retranslated[idx],
                    source_language="en",
                    pivot_language="fr",
                    similarity=float(en_similarities[idx]),
                )
            stats.durations_seconds["en_cycle"] = time.perf_counter() - t_en
            logger.info(
                f"Cycle EN->FR->EN : {len(en_indices)} articles en "
                f"{stats.durations_seconds['en_cycle']:.1f} s"
            )

        # Cycle FR : fr -> en (pivot) -> fr (retour)
        if fr_indices:
            t_fr = time.perf_counter()
            fr_texts = [texts[i] for i in fr_indices]
            fr_to_en = self._translate_batch(fr_texts, "fr->en")
            fr_retranslated = self._translate_batch(fr_to_en, "en->fr")
            fr_similarities = self._compute_similarities(fr_texts, fr_retranslated)

            for idx, i in enumerate(fr_indices):
                results[i] = self._build_result(
                    original_text=fr_texts[idx],
                    augmented_text=fr_retranslated[idx],
                    source_language="fr",
                    pivot_language="en",
                    similarity=float(fr_similarities[idx]),
                )
            stats.durations_seconds["fr_cycle"] = time.perf_counter() - t_fr
            logger.info(
                f"Cycle FR->EN->FR : {len(fr_indices)} articles en "
                f"{stats.durations_seconds['fr_cycle']:.1f} s"
            )

        # Compiler stats
        accepted_results: list[BackTranslationResult] = []
        for r in results:
            if r is None:
                stats.total_failed += 1
                continue
            stats.total_generated += 1
            if r.accepted:
                stats.total_accepted += 1
                accepted_results.append(r)
            elif r.similarity < self.similarity_min:
                stats.total_rejected_low_similarity += 1
            else:
                stats.total_rejected_high_similarity += 1

        stats.durations_seconds["total"] = time.perf_counter() - t_start
        logger.info(
            f"Back-translation terminee : {stats.total_accepted}/{stats.total_input} "
            f"acceptes ({stats.acceptance_rate():.1%}), "
            f"{stats.total_rejected_low_similarity} rejets trop dissemblables, "
            f"{stats.total_rejected_high_similarity} rejets trop similaires, "
            f"{stats.total_failed} echecs, "
            f"en {stats.durations_seconds['total']:.1f} s"
        )
        return accepted_results, stats

    def _build_result(
        self,
        original_text: str,
        augmented_text: str,
        source_language: SupportedLanguage,
        pivot_language: SupportedLanguage,
        similarity: float,
    ) -> BackTranslationResult:
        """Applique le filtre de similarite et construit le resultat."""
        if similarity < self.similarity_min:
            return BackTranslationResult(
                original_text=original_text,
                augmented_text=augmented_text,
                source_language=source_language,
                pivot_language=pivot_language,
                similarity=similarity,
                accepted=False,
                reason_rejected=f"similarity {similarity:.3f} < {self.similarity_min}",
            )
        if similarity > self.similarity_max:
            return BackTranslationResult(
                original_text=original_text,
                augmented_text=augmented_text,
                source_language=source_language,
                pivot_language=pivot_language,
                similarity=similarity,
                accepted=False,
                reason_rejected=f"similarity {similarity:.3f} > {self.similarity_max}",
            )
        return BackTranslationResult(
            original_text=original_text,
            augmented_text=augmented_text,
            source_language=source_language,
            pivot_language=pivot_language,
            similarity=similarity,
            accepted=True,
        )
