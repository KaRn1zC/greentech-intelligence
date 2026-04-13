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

# Etat de session : True quand un 402 a ete observe au moins une fois.
# Pas persistant entre processus : une nouvelle execution retente HF.
_hf_quota_exhausted: bool = False
_state_lock = threading.Lock()


def is_hf_quota_exhausted() -> bool:
    """Retourne True si un appel HF a deja echoue avec un 402 dans la session."""
    return _hf_quota_exhausted


def mark_hf_quota_exhausted() -> None:
    """Marque le quota HF comme epuise et trace le basculement vers le local."""
    global _hf_quota_exhausted
    with _state_lock:
        if _hf_quota_exhausted:
            return
        _hf_quota_exhausted = True
        logger.warning(
            "Quota HF Inference Providers epuise (HTTP 402). "
            "Bascule sur le modele Qwen local (GPU AMD ROCm) pour la suite "
            "de la session."
        )


def reset_hf_quota_flag() -> None:
    """Reinitialise l'etat (utile en test pour repasser par HF)."""
    global _hf_quota_exhausted
    with _state_lock:
        _hf_quota_exhausted = False


def _is_quota_exhausted_error(exc: BaseException) -> bool:
    """Detecte si une exception correspond a un quota HF epuise (HTTP 402)."""
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 402:
            return True
    return "Payment Required" in str(exc)


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
        return await local.chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )

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
            logger.info(
                "Nouvel essai de la requete en cours sur le modele local apres 402"
            )
            local = LocalQwenClient.get()
            return await local.chat_completion(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        raise
    finally:
        await client.close()
