"""Dispatcher Hugging Face -> Qwen local avec detection du quota epuise.

Ce module centralise l'acces au LLM instructif utilise dans toute la chaine
d'IA du projet (classification Green IT et resumes d'articles). Il encapsule
la strategie de fallback :

1. Tant que le quota mensuel HF Inference Providers n'est pas epuise, les
   requetes passent par `AsyncInferenceClient` (rapide, gratuit).
2. Des qu'une requete echoue avec un code HTTP 402 ("Payment Required"),
   l'etat `_hf_quota_exhausted` passe a True. Les appels suivants dans la
   meme session basculent directement sur `LocalQwenClient` (auto-hosting
   sur le GPU AMD RX 7900 XTX + ROCm 7.2), sans re-tenter HF.
3. Au prochain demarrage du processus, l'etat est reset : on retente HF
   (utile si le quota a ete recharge entre-temps, par ex. debut de mois).

Interface
---------

`chat_completion` expose la meme signature que l'API HF
(`messages`, `max_tokens`, `temperature`) et renvoie un objet dont la
forme (`response.choices[0].message.content`) est compatible avec les
deux clients.

"""

from __future__ import annotations

import threading
from typing import Any

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger

from greentech.ai.services.llm_local import LocalQwenClient
from greentech.config import get_settings

# Etat de session : True quand l'API HF ne peut plus servir le modele demande
# (quota epuise OU modele non supporte par les providers actifs). Dans les deux
# cas on bascule sur le local pour le reste de la session pour eviter les
# aller-retours couteux. Pas persistant entre processus : une nouvelle execution
# retente HF.
_hf_unavailable: bool = False
_state_lock = threading.Lock()


def is_hf_quota_exhausted() -> bool:
    """Retourne True si HF a deja echoue (402 ou 400 model_not_supported)."""
    return _hf_unavailable


def mark_hf_unavailable(reason: str) -> None:
    """Marque l'API HF comme indisponible et trace le basculement vers le local.

    Args:
        reason: Motif court (``"quota epuise"``, ``"modele non supporte"``, etc.)
            affiche dans le log pour faciliter le diagnostic.
    """
    global _hf_unavailable
    with _state_lock:
        if _hf_unavailable:
            return
        _hf_unavailable = True
        logger.warning(
            f"API HF Inference Providers indisponible ({reason}). "
            "Bascule sur le modele Qwen local (GPU AMD ROCm si disponible, "
            "sinon CPU) pour la suite de la session."
        )


# Alias conserve pour retro-compatibilite avec le code appelant qui connaissait
# la nomenclature historique centree sur le quota epuise.
def mark_hf_quota_exhausted() -> None:
    """Alias historique de ``mark_hf_unavailable("quota epuise - HTTP 402")``."""
    mark_hf_unavailable("quota epuise - HTTP 402")


def reset_hf_quota_flag() -> None:
    """Reinitialise l'etat (utile en test pour repasser par HF)."""
    global _hf_unavailable
    with _state_lock:
        _hf_unavailable = False


def _is_quota_exhausted_error(exc: BaseException) -> bool:
    """Detecte si une exception correspond a un quota HF epuise (HTTP 402)."""
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 402:
            return True
    return "Payment Required" in str(exc)


def _is_model_not_supported_error(exc: BaseException) -> bool:
    """Detecte si HF refuse le modele (ex. ``Qwen/Qwen2.5-3B-Instruct`` pas
    expose par les Inference Providers actifs du compte).

    L'erreur se presente sous la forme d'un HTTP 400 avec un message contenant
    ``model_not_supported`` ou ``is not supported by any provider``. On couvre
    les deux formulations pour resister aux futures rotations de wording cote
    HF.
    """
    message = str(exc).lower()
    indicators = (
        "model_not_supported",
        "is not supported by any provider",
        "not supported by any provider you have enabled",
    )
    return any(ind in message for ind in indicators)


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 200,
    temperature: float = 0.1,
    model_hf: str | None = None,
) -> Any:
    """Envoie une requete chat au LLM instructif avec fallback automatique.

    Tente d'abord l'API HF Serverless. En cas d'erreur 402 (quota mensuel
    epuise), bascule immediatement et pour le reste de la session sur le
    modele Qwen local (GPU AMD ROCm). Les erreurs non liees au quota sont
    re-levees telles quelles : l'appelant reste responsable de sa politique
    de retry.

    Args:
        messages: Messages au format OpenAI/HF
            (`[{"role": "system", "content": "..."}, ...]`).
        max_tokens: Plafond de tokens generes.
        temperature: Temperature de sampling (0 = argmax).
        model_hf: Identifiant du modele HF a utiliser cote cloud. Si None,
            on utilise `settings.huggingface_model_classifier_llm`
            (meme modele pour classification et resumes par defaut).

    Returns:
        Objet avec `response.choices[0].message.content` contenant la
        generation, compatible avec les deux backends.
    """
    settings = get_settings()
    model = model_hf or settings.huggingface_model_classifier_llm

    if is_hf_quota_exhausted():
        local = LocalQwenClient.get()
        return await local.chat_completion(messages, max_tokens=max_tokens, temperature=temperature)

    client = AsyncInferenceClient(
        model=model,
        token=settings.huggingface_token,
        timeout=60.0,
    )
    try:
        response = await client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response
    except Exception as exc:
        if _is_quota_exhausted_error(exc):
            mark_hf_quota_exhausted()
            logger.info("Nouvel essai de la requete en cours sur le modele local apres 402")
            local = LocalQwenClient.get()
            return await local.chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        if _is_model_not_supported_error(exc):
            mark_hf_unavailable(f"modele '{model}' non servi par les Inference Providers actifs")
            logger.info(f"Modele {model} indisponible cote HF, bascule sur le modele local")
            local = LocalQwenClient.get()
            return await local.chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        raise
    finally:
        await client.close()
