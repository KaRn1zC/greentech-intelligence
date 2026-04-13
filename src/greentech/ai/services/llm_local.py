"""Inference locale Qwen via ROCm sur le GPU AMD RX 7900 XTX.

Ce module fournit une implementation locale du meme modele Qwen que celui
appele en production via l'API Hugging Face Serverless. Il sert de fallback
automatique lorsque le quota mensuel HF est epuise (erreur HTTP 402
Payment Required), de facon a garantir que le pipeline de classification
et de resume reste fonctionnel sans interruption.

Strategie de design
-------------------

- **Chargement paresseux** : le modele (plusieurs Go) n'est telecharge et
  charge en VRAM que lorsqu'un premier appel local est reellement necessaire.
  Les scripts qui n'utilisent que HF ne paient donc jamais ce cout.
- **Singleton** : un seul chargement par processus. Les appels successifs
  reutilisent le modele en memoire pour amortir le cout initial.
- **Interface compatible HF** : la methode `chat_completion` renvoie un
  objet dont la forme (`response.choices[0].message.content`) est compatible
  avec l'usage existant du `AsyncInferenceClient`, ce qui permet au
  dispatcher de basculer sans modifier les appelants.

Materiel cible
--------------

- **PC fixe** : AMD Radeon RX 7900 XTX (24 Go VRAM), ROCm 7.2, torch 2.9.1+rocm.
  `torch.cuda.is_available()` renvoie True car ROCm expose l'API CUDA.
- **PC portable** : fallback via `torch_directml` si installe, sinon CPU.

Le modele retenu est le **meme** que celui utilise cote HF cloud :
`Qwen/Qwen2.5-7B-Instruct`. Cela garantit une continuite qualitative
totale entre les deux backends : un article donne obtient le meme type
de reponse qu'il soit traite par HF ou localement. En FP16 sur la
RX 7900 XTX (24 Go VRAM), le 7B occupe environ 14 Go, ce qui laisse
une marge confortable pour le contexte et les activations. La latence
d'inference est de l'ordre de 5-8 s par article, acceptable pour un
traitement par lots.

"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from greentech.config import get_settings

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


# Modele par defaut pour le fallback local. Identique au modele HF utilise
# en production pour garantir une continuite de qualite entre les deux
# backends (cloud et local). Le 7B en FP16 occupe ~14 Go de VRAM, ce qui
# rentre largement dans les 24 Go de la RX 7900 XTX.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass(frozen=True)
class _LocalMessage:
    """Imite la forme de `response.choices[0].message` retournee par HF."""

    content: str


@dataclass(frozen=True)
class _LocalChoice:
    """Imite la forme de `response.choices[0]` retournee par HF."""

    message: _LocalMessage
    index: int = 0


@dataclass(frozen=True)
class LocalChatCompletionResponse:
    """Reponse du client local compatible avec l'API `AsyncInferenceClient`.

    L'appelant peut acceder a `response.choices[0].message.content` exactement
    comme avec la reponse de `AsyncInferenceClient.chat_completion`.
    """

    choices: list[_LocalChoice]


class LocalQwenClient:
    """Client d'inference locale Qwen, singleton, charge paresseusement.

    L'instance est unique pour le processus et partage le modele en VRAM
    entre tous les appels (classification et resumes). Le chargement est
    protege par un verrou pour supporter les appels concurrents sans charger
    plusieurs fois.
    """

    _instance: LocalQwenClient | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or getattr(
            settings, "huggingface_model_local_fallback", DEFAULT_LOCAL_MODEL
        )
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._model: PreTrainedModel | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._load_lock = threading.Lock()

    @classmethod
    def get(cls) -> LocalQwenClient:
        """Renvoie l'instance singleton (cree si premier appel)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def _pick_device() -> tuple[str, torch.dtype]:
        """Selectionne le meilleur device disponible et le dtype associe.

        Priorite :
        1. CUDA/ROCm (PC fixe avec RX 7900 XTX + ROCm)
        2. DirectML (PC portable sans ROCm)
        3. CPU (derniere solution, tres lent)
        """
        import torch

        if torch.cuda.is_available():
            logger.info(
                f"Device local : cuda ({torch.cuda.get_device_name(0)}), "
                f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} Go VRAM"
            )
            return "cuda", torch.float16

        try:
            import torch_directml

            device = torch_directml.device()
            logger.info(f"Device local : DirectML ({device})")
            return str(device), torch.float16
        except ImportError:
            pass

        logger.warning(
            "Aucun GPU detecte pour l'inference locale. Utilisation du CPU "
            "(lent : plusieurs dizaines de secondes par article)."
        )
        return "cpu", torch.float32

    def _ensure_loaded(self) -> None:
        """Charge le modele et le tokenizer en memoire si ce n'est pas deja fait."""
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device, dtype = self._pick_device()
            logger.info(
                f"Chargement du modele local {self.model_name} "
                f"(device={device}, dtype={dtype})..."
            )

            settings = get_settings()
            hf_token = settings.huggingface_token or None

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=hf_token,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=hf_token,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model.to(torch.device(device))
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._dtype = dtype

            logger.info("Modele local charge et pret a l'inference")

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Generation synchrone (appelee via `asyncio.to_thread`).

        Args:
            messages: Liste de messages au format OpenAI/HF
                (`[{"role": "system", "content": "..."}]`).
            max_new_tokens: Nombre maximum de tokens a generer.
            temperature: Temperature de sampling (0 = argmax deterministe).

        Returns:
            Texte genere, sans le prompt, avec les caracteres speciaux decode.
        """
        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        # Application du chat template propre a Qwen (tokens <|im_start|> etc.)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(torch.device(self._device)) for k, v in inputs.items()}

        do_sample = temperature > 0
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # On ne garde que les tokens reellement generes (apres le prompt).
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return generated_text

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> LocalChatCompletionResponse:
        """Inference locale asynchrone compatible avec l'API HF.

        Le chargement du modele (s'il n'a pas encore ete fait) et la
        generation sont executes dans un thread worker pour ne pas bloquer
        la boucle asyncio. Plusieurs appels concurrents serialisent
        naturellement sur le GPU (ce qui est souhaite pour eviter de saturer
        la VRAM).

        Args:
            messages: Messages au format chat (system/user/assistant).
            max_tokens: Plafond de tokens generes (nom aligne sur l'API HF).
            temperature: Temperature de sampling.

        Returns:
            Reponse dans un format compatible avec `AsyncInferenceClient`.
        """

        def _run() -> str:
            self._ensure_loaded()
            return self._generate_sync(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        text = await asyncio.to_thread(_run)
        return LocalChatCompletionResponse(
            choices=[_LocalChoice(message=_LocalMessage(content=text))]
        )

    async def close(self) -> None:
        """Compatibilite d'API avec `AsyncInferenceClient.close`.

        Ne libere pas la VRAM : le modele est conserve pour les appels
        suivants. Une liberation explicite peut etre faite via `unload`.
        """
        return None

    def unload(self) -> None:
        """Libere explicitement la VRAM occupee par le modele.

        A utiliser uniquement si l'on sait qu'il n'y aura plus d'appels
        locaux dans la session (par exemple a la fin d'un long batch).
        """
        if self._model is None:
            return
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning(f"Echec liberation GPU : {exc}")
            logger.info("Modele local dechargé, VRAM liberee")
