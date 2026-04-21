"""Resume de classification : prompt et logique centralises.

Ce module definit la representation canonique qu'utilise le classifieur
Qwen3-4B + LoRA pour decider si un article est Green IT. Il est invoque
a deux moments distincts de la chaine :

- Lors de la **constitution du dataset d'entrainement**, par le script
  batch ``scripts/generate_classification_summaries.py`` qui parcourt
  la table ``articles`` et peuple la colonne ``articles.resume``.
- Lors de l'**inference temps-reel** dans ``routes/analyze.py``, juste
  avant de passer l'article au classifieur entraine.

Avoir un module unique garantit que le prompt et les parametres utilises
a l'entrainement sont strictement identiques a ceux utilises a l'inference.
C'est la condition necessaire pour eviter toute derive de distribution
entre les deux phases, derive qui se traduirait par une degradation
silencieuse des metriques en production.

Les constantes publiques ``CLASSIFICATION_SYSTEM_PROMPT``,
``CLASSIFICATION_USER_PROMPT_TEMPLATE``, ``CLASSIFICATION_MAX_TOKENS``
et ``CLASSIFICATION_TEMPERATURE`` sont exposees pour permettre aux autres
modules (tests, tracking MLflow, documentation) de s'y referer par
import plutot que par duplication.

"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from greentech.ai.services.llm_dispatcher import (
    chat_completion,
    is_hf_quota_exhausted,
)
from greentech.config import get_settings

# =============================================================================
# PROMPT DE CLASSIFICATION (abstract dense, style scientifique uniforme)
# =============================================================================

CLASSIFICATION_SYSTEM_PROMPT = (
    "Tu es un redacteur scientifique specialise dans la synthese d'articles "
    "technologiques. Tu produis des resumes denses, factuels et neutres, "
    "dans le style d'un abstract de publication scientifique."
)

CLASSIFICATION_USER_PROMPT_TEMPLATE = (
    "Redige un resume en francais de l'article ci-dessous. Contraintes strictes :\n"
    "\n"
    "- Longueur OBLIGATOIRE : minimum 150 mots, ideal 180 a 220 mots. "
    "Un resume de moins de 150 mots est incomplet et INACCEPTABLE : "
    "utilise 4 a 6 phrases denses pour couvrir l'integralite du sujet. "
    "Ne t'arrete JAMAIS avant 150 mots.\n"
    "- Style : abstract scientifique - factuel, neutre. Pas de style "
    "journalistique, pas d'adverbes d'opinion, pas de listes a puces, "
    "pas d'emojis, pas de formules d'accroche.\n"
    "- Contenu : le sujet central de l'article, l'approche ou la technologie "
    "mise en avant, les resultats ou enjeux principaux mentionnes. Si "
    "l'article inclut des chiffres cles (pourcentages, gains energetiques, "
    "benchmarks), reprends-les dans le resume.\n"
    "- Vocabulaire technique : conserve les acronymes et termes techniques "
    "en anglais tels quels (NLP, GPU, ML, AI, API, SDK, LLM, ...), ne les "
    "traduis pas et ne les remplace pas par des abreviations inventees.\n"
    "- Preservation des signaux techniques : conserve les mentions eventuelles "
    "d'efficacite energetique, de sobriete numerique, de durabilite, "
    "d'empreinte carbone, d'eco-conception ou d'economie circulaire si et "
    "seulement si elles figurent dans l'article. N'invente rien.\n"
    "- Si l'article est redige dans une autre langue que le francais, "
    "traduis-le en francais pour le resume.\n"
    "\n"
    "Article :\n"
    "{contenu}\n"
    "\n"
    "Resume en francais (180-220 mots ideal, 150 minimum, 4-6 phrases) :"
)

# Borne dure cote LLM pour garantir que le resume tient dans le budget
# tokens du classifieur Qwen3-4B + LoRA (max_length=512). Avec ~30 tokens
# pour le titre, ~5 tokens speciaux et 2 pour le separateur `\n\n`, il
# reste environ 475 tokens disponibles pour le resume : on cap a 450 pour
# conserver une marge de securite contre la tokenisation sub-word des
# mots techniques longs (ex. "decarbonation", "photovoltaique").
CLASSIFICATION_MAX_TOKENS = 450

# Temperature faible : on veut des resumes quasi-deterministes pour deux
# raisons. (1) Reproductibilite : deux executions successives du script
# batch sur le meme dataset produisent des resumes tres proches, donc le
# dataset d'entrainement reste stable entre deux runs. (2) Fidelite au
# texte : on minimise le risque que le LLM invente un aspect ecologique
# absent de l'article source, ce qui empoisonnerait le label appris.
CLASSIFICATION_TEMPERATURE = 0.2

# Borne haute du texte d'entree envoye au LLM. Au-dela, on tronque a la
# derniere phrase complete pour eviter de couper en plein milieu d'une
# idee. 15000 caracteres correspondent approximativement a 4000 tokens,
# ce qui reste tres confortable face aux 32k tokens de contexte du
# fallback local Qwen2.5-3B et aux 262k tokens de Qwen3-4B-Instruct-2507
# cote cloud. On capture ainsi l'integralite de 95%+ des articles web
# et la totalite des abstracts arXiv.
CLASSIFICATION_MAX_INPUT_CHARS = 15_000
CLASSIFICATION_MIN_INPUT_CHARS = 50

# Entropie minimale sur le contenu : on rejette les textes qui n'ont pas
# au moins 10 caracteres uniques distincts. Cela filtre les contenus
# manifestement invalides (ex: "AAAAAAAA..." repete, placeholder de test,
# garbage issu d'un upload casse). Observee en production le 2026-04-19 :
# certains articles de test utilisateur contenaient 100 fois le meme
# caractere et le LLM repondait "Desole, aucun article fourni" ce qui
# polluait le dataset d'entrainement avec un faux resume.
CLASSIFICATION_MIN_UNIQUE_CHARS = 10

# Motifs de contenu degrades connus a filtrer avant appel LLM. Concerne
# principalement les preprints arXiv retires / retractes qui ne gardent
# qu'une note laconique ("This paper has been withdrawn by the author")
# insuffisante pour un resume informatif. Le LLM produirait un resume
# vide ou une hallucination, inutile pour l'entrainement.
_DEGENERATE_CONTENT_PATTERNS = (
    "this paper has been withdrawn",
    "this preprint has been withdrawn",
    "this draft is withdrawn",
    "this paper has been retracted",
    "this paper has been temporarily removed",
    "this paper has been removed",
)


@dataclass(frozen=True)
class SummaryResult:
    """Resultat d'un appel de resume par le LLM.

    Attributes:
        resume: Texte du resume genere, ou None en cas d'echec.
        temps_ms: Temps d'inference en millisecondes, borne inferieure
            incluant le temps reseau HF ou le temps GPU local.
        modele: Identifiant du backend utilise. La valeur reflete le
            fallback : ``"qwen_local"`` si HF est epuise, sinon le nom
            du modele HF.
        succes: True si le LLM a retourne un resume non vide.
        erreur: Message d'erreur eventuel (None si succes).
    """

    resume: str | None
    temps_ms: int
    modele: str
    succes: bool
    erreur: str | None = None


def _truncate_input(text: str) -> str:
    """Tronque le texte au dernier point avant la borne pour preserver le sens.

    Args:
        text: Texte brut de l'article, deja strip().

    Returns:
        Texte eventuellement tronque a ``CLASSIFICATION_MAX_INPUT_CHARS``,
        en coupant sur une fin de phrase quand possible pour ne pas
        laisser une phrase inachevee en fin d'entree.
    """
    if len(text) <= CLASSIFICATION_MAX_INPUT_CHARS:
        return text

    truncated = text[:CLASSIFICATION_MAX_INPUT_CHARS]
    last_period = truncated.rfind(".")
    if last_period > CLASSIFICATION_MAX_INPUT_CHARS // 2:
        truncated = truncated[: last_period + 1]

    logger.debug(
        f"Contenu tronque pour resume de classification : "
        f"{len(text)} -> {len(truncated)} caracteres"
    )
    return truncated


def _backend_tag(model_hf: str) -> str:
    """Retourne l'identifiant du backend actif pour les logs et la tracabilite.

    Args:
        model_hf: Identifiant du modele HF cible (utilise si le quota n'est
            pas epuise).

    Returns:
        ``"qwen_local"`` si HF est indisponible pour la session, sinon
        ``model_hf``.
    """
    return "qwen_local" if is_hf_quota_exhausted() else model_hf


async def summarize_for_classification(text: str) -> SummaryResult:
    """Genere un resume de classification a partir du texte brut d'un article.

    Fonction pure : ne lit ni n'ecrit en base. Applique le prompt de
    classification centralise et retourne le texte du resume. L'appelant
    est responsable de la persistance (en base via ``articles.resume``,
    en cache, ou simplement en memoire pour l'inference temps-reel).

    Le resume produit respecte par construction le budget tokens du
    classifieur entraine (450 tokens max), ce qui garantit qu'il sera
    utilisable directement comme entree du modele sans troncature
    supplementaire par le tokenizer au moment de l'inference.

    Args:
        text: Contenu brut de l'article. Toute langue acceptee (le prompt
            force la traduction en francais si necessaire).

    Returns:
        SummaryResult contenant le resume (ou None en cas d'echec) et
        les metriques associees (latence, backend utilise, succes).
    """
    settings = get_settings()
    model_hf = settings.huggingface_model_summarizer

    if not text or len(text.strip()) < CLASSIFICATION_MIN_INPUT_CHARS:
        logger.warning(
            f"Texte trop court pour un resume de classification : "
            f"{len(text or '')} chars (minimum {CLASSIFICATION_MIN_INPUT_CHARS})"
        )
        return SummaryResult(
            resume=None,
            temps_ms=0,
            modele=model_hf,
            succes=False,
            erreur=(f"Contenu insuffisant (minimum {CLASSIFICATION_MIN_INPUT_CHARS} caracteres)"),
        )

    stripped = text.strip()

    # Detection d'entropie trop faible : "AAAAAAA..." repete, placeholder
    # de test, garbage. Les textes naturels ont au moins 10-15 caracteres
    # uniques meme sur 50 chars (alphabet + espaces + ponctuation).
    unique_chars = set(stripped.lower())
    if len(unique_chars) < CLASSIFICATION_MIN_UNIQUE_CHARS:
        logger.warning(
            f"Contenu rejete (entropie insuffisante : {len(unique_chars)} chars "
            f"distincts, minimum {CLASSIFICATION_MIN_UNIQUE_CHARS}) : "
            f"{stripped[:60]!r}..."
        )
        return SummaryResult(
            resume=None,
            temps_ms=0,
            modele=model_hf,
            succes=False,
            erreur=f"Entropie du contenu trop faible ({len(unique_chars)} chars distincts)",
        )

    # Rejet des preprints retires / retractes : leur texte se reduit a
    # "This paper has been withdrawn..." et ne peut pas produire un
    # resume utile. Detection par sous-chaine (casse insensible).
    lower = stripped.lower()
    for pattern in _DEGENERATE_CONTENT_PATTERNS:
        if pattern in lower[:300]:  # on cherche dans les 300 premiers chars
            logger.warning(
                f"Contenu rejete (preprint retire/retracte detecte) : "
                f"{stripped[:80]!r}..."
            )
            return SummaryResult(
                resume=None,
                temps_ms=0,
                modele=model_hf,
                succes=False,
                erreur="Preprint retire/retracte sans abstract exploitable",
            )

    texte_tronque = _truncate_input(text.strip())
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CLASSIFICATION_USER_PROMPT_TEMPLATE.format(contenu=texte_tronque),
        },
    ]

    start = time.perf_counter()
    try:
        completion = await chat_completion(
            messages=messages,
            max_tokens=CLASSIFICATION_MAX_TOKENS,
            temperature=CLASSIFICATION_TEMPERATURE,
            model_hf=model_hf,
        )
        temps_ms = int((time.perf_counter() - start) * 1000)

        contenu = completion.choices[0].message.content if completion.choices else ""
        resume = contenu.strip()
        backend = _backend_tag(model_hf)

        if not resume:
            logger.warning(f"Le LLM a retourne un resume vide ({temps_ms}ms, backend={backend})")
            return SummaryResult(
                resume=None,
                temps_ms=temps_ms,
                modele=backend,
                succes=False,
                erreur="Le LLM a retourne un resume vide",
            )

        logger.info(
            f"Resume de classification genere en {temps_ms}ms "
            f"({len(resume)} chars, backend={backend})"
        )
        return SummaryResult(
            resume=resume,
            temps_ms=temps_ms,
            modele=backend,
            succes=True,
        )

    except Exception as exc:
        temps_ms = int((time.perf_counter() - start) * 1000)
        logger.error(f"Echec resume de classification ({temps_ms}ms) : {exc}")
        return SummaryResult(
            resume=None,
            temps_ms=temps_ms,
            modele=_backend_tag(model_hf),
            succes=False,
            erreur=str(exc),
        )
