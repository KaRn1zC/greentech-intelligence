"""Inference locale Qwen via ROCm sur le GPU AMD RX 7900 XTX.

Ce module fournit une cascade de fallback locale activee automatiquement
par le dispatcher lorsque l'API HuggingFace Inference Providers ne peut
pas servir la requete (quota epuise HTTP 402, ou modele cloud non servi
par les providers actifs du compte HTTP 400). L'objectif est que le
pipeline de classification et de resume reste toujours operationnel,
meme hors-ligne ou en fin de quota mensuel.

Strategie de fallback en cascade
--------------------------------

1. **Modele cloud (de reference)** : ``Qwen/Qwen3-4B-Instruct-2507``. Utilise
   par le dispatcher en premier via ``AsyncInferenceClient``. Non concerne
   par ce module : ce module ne s'active que lorsque le cloud echoue.
2. **Local n1** : ``Qwen/Qwen2.5-3B-Instruct`` (``DEFAULT_LOCAL_MODEL``).
   Choix par defaut quand la memoire de la machine le permet. C'est le
   modele le plus proche en taille de parametres du cloud (4B -> 3B), ce
   qui minimise l'ecart qualitatif entre les deux backends.
3. **Local n2 (lightweight)** : ``Qwen/Qwen2.5-1.5B-Instruct``. Bascule
   automatique si :

   - le preflight memoire detecte moins de 8 Go VRAM (FP16 GPU) ou
     14 Go RAM (FP32 CPU) disponibles, OU
   - le chargement du 3B leve un ``OutOfMemoryError`` malgre le preflight
     (d'autres processus concurrents ayant grignote la memoire).

Les deux modeles locaux sont configurables via les variables d'environnement
``HUGGINGFACE_MODEL_LOCAL_FALLBACK`` et
``HUGGINGFACE_MODEL_LOCAL_FALLBACK_LIGHTWEIGHT``.

Strategie de design
-------------------

- **Chargement paresseux** : le modele (plusieurs Go) n'est telecharge et
  charge en VRAM que lorsqu'un premier appel local est reellement necessaire.
  Les scripts qui n'utilisent que HF ne paient donc jamais ce cout.
- **Singleton** : un seul chargement par processus. Les appels successifs
  reutilisent le modele en memoire pour amortir le cout initial.
- **Interface compatible HF** : la methode ``chat_completion`` renvoie un
  objet dont la forme (``response.choices[0].message.content``) est compatible
  avec l'usage existant du ``AsyncInferenceClient``, ce qui permet au
  dispatcher de basculer sans modifier les appelants.

Materiel cible
--------------

- **PC fixe** : AMD Radeon RX 7900 XTX (24 Go VRAM), ROCm 7.2, torch 2.9.1+rocm.
  ``torch.cuda.is_available()`` renvoie True car ROCm expose l'API CUDA.
  Latence typique pour le 3B en FP16 : 2-4 s par article, acceptable pour
  un traitement interactif.
- **PC portable** : fallback via ``torch_directml`` si installe, sinon CPU.
  Sur CPU, le preflight memoire declenche generalement la bascule vers le
  1.5B pour garder des temps de reponse soutenables.

Note sur le choix cloud vs local
--------------------------------

On ne vise pas une egalite de taille de parametres entre cloud et local :
le cloud (4B) sert de reference qualitative, le local (3B ou 1.5B) assure
la continuite de service. La difference de generation entre Qwen3-4B et
Qwen2.5-3B/1.5B est acceptable pour un fallback, et largement preferable
a un pipeline qui s'arreterait en cas de 402.

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
# backends (cloud et local). Le 3B en FP16 occupe ~6 Go de VRAM, ce qui
# rentre tres largement dans les 24 Go de la RX 7900 XTX et fonctionne
# sur la majorite des GPU discrets modernes.
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Modele de secours lorsque la machine cible ne dispose pas d'assez de memoire
# pour le 3B. Environ 3 Go en FP16, 6 Go en FP32, ce qui rentre dans la quasi
# totalite des configurations modernes (y compris laptops sans GPU dedie).
LIGHTWEIGHT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Seuils empiriques de memoire requise pour charger le Qwen 3B sans se
# retrouver en OOM (poids + activations + contexte de chat moyen).
# FP16 sur GPU : ~6 Go effectifs, on exige 8 pour garder un peu de marge.
# FP32 sur CPU : le poids seul fait ~12 Go + overhead PyTorch, donc 14 Go
# minimum de RAM disponible avant de tenter.
_MIN_VRAM_GB_FOR_DEFAULT_FP16 = 8.0
_MIN_RAM_GB_FOR_DEFAULT_FP32 = 14.0


def _fix_mojibake(text: str) -> str:
    """Repare le mojibake UTF-8 lu-comme-Latin-1 parfois produit par les
    tokenizers byte-level BPE.

    Les tokenizers Qwen / GPT-2 / Llama encodent chaque byte UTF-8 comme un
    caractere Unicode distinct de la plage Latin-1 (0x00-0xFF). Selon la
    combinaison de version de ``transformers``, de ``tokenizers`` et de
    configuration ``clean_up_tokenization_spaces``, la methode ``decode``
    peut renvoyer la chaine dans laquelle chaque byte UTF-8 est reste sous
    forme de caractere Latin-1 au lieu d'etre assemble en caractere Unicode.
    Consequence : ``"réduit"`` (bytes 0xC3 0xA9) devient ``"rÃ©duit"``
    (caracteres U+00C3 et U+00A9).

    Cette fonction detecte ce cas par la presence d'une sequence
    caracteristique (``Ã`` suivi d'un caractere Latin-1) et tente une
    reinterpretation ``encode('latin-1').decode('utf-8')``. Si la tentative
    leve une ``UnicodeDecodeError``, le texte est rendu tel quel : cela
    signifie qu'il ne s'agissait pas de mojibake mais d'un vrai ``Ã``
    legitime (cas tres rare en francais).

    Args:
        text: Texte potentiellement mojibake renvoye par le tokenizer.

    Returns:
        Texte avec les caracteres francais (``é``, ``è``, ``à``, ``ç``, ``ù``,
        etc.) correctement reassembles, ou texte inchange si aucun mojibake
        n'est detecte ou si la conversion echoue.
    """
    if not text:
        return text
    # Signatures caracteristiques du mojibake UTF-8→Latin-1 sur du francais.
    mojibake_markers = ("Ã", "Â", "\u00e2\u0080")
    if not any(marker in text for marker in mojibake_markers):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


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

    @staticmethod
    def _available_memory_gb(device: str) -> float | None:
        """Estime la memoire disponible (en Go) sur le device cible.

        Sur GPU on interroge directement le runtime pour obtenir la VRAM libre.
        Sur CPU on lit ``/proc/meminfo`` (disponible dans tous les conteneurs
        Linux utilises par ce projet) pour recuperer ``MemAvailable``, qui
        reflete la memoire reellement allouable sans swap excessif. En dehors
        de Linux on renvoie ``None`` : l'appelant prefere alors tenter le
        modele demande et attraper un eventuel OOM plutot que bloquer
        inutilement sur une heuristique imparfaite.
        """
        try:
            import torch

            if device == "cuda" and torch.cuda.is_available():
                free_bytes, _total_bytes = torch.cuda.mem_get_info()
                return float(free_bytes) / 1024**3
        except Exception as exc:
            logger.debug(f"Impossible de lire la VRAM disponible : {exc}")

        try:
            with open("/proc/meminfo", encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        kib = int(line.split()[1])
                        return kib * 1024 / 1024**3
        except OSError:
            return None
        return None

    @classmethod
    def _resolve_model_name(cls, requested: str, device: str) -> str:
        """Choisit le modele a charger en fonction de la memoire disponible.

        Si ``requested`` designe un modele "gros" (3B ou plus) mais que la
        machine courante n'a pas assez de memoire (VRAM sur GPU, RAM sur CPU),
        on bascule sur le modele lightweight (1.5B) defini dans la
        configuration. Cela permet au projet de fonctionner sur les postes
        moins puissants qui clonent le depot sans savoir ajuster manuellement
        ``HUGGINGFACE_MODEL_LOCAL_FALLBACK``.

        Args:
            requested: Nom du modele demande (via ``settings`` ou parametre).
            device: Device selectionne (``cuda``, ``cpu``, etc.).

        Returns:
            Nom du modele effectivement a charger.
        """
        # On ne declenche le fallback que pour les modeles "plein format"
        # (3B, 7B...). Les plus petits (1.5B, 0.5B) tiennent toujours en
        # memoire et leur chargement ne merite pas de preflight.
        name_lower = requested.lower()
        needs_check = any(tag in name_lower for tag in ("7b", "3b"))
        if not needs_check:
            return requested

        available = cls._available_memory_gb(device)
        if available is None:
            logger.debug("Memoire non mesurable sur ce systeme, tentative avec le modele demande")
            return requested

        # Seuils calibres pour le 3B par defaut. Pour un 7B le code essaiera
        # d'abord le chargement et basculera sur OOM via le try/except en aval.
        threshold = (
            _MIN_VRAM_GB_FOR_DEFAULT_FP16 if device == "cuda" else _MIN_RAM_GB_FOR_DEFAULT_FP32
        )
        # Si on a affaire a un 7B, on exige deux fois plus de memoire qu'un 3B
        # pour deviner correctement la trajectoire. Le fallback final sur OOM
        # reste en place si l'estimation est trop optimiste.
        if "7b" in name_lower:
            threshold *= 2

        if available >= threshold:
            logger.info(
                f"Memoire suffisante ({available:.1f} Go >= {threshold:.1f} Go), "
                f"chargement du modele complet {requested}"
            )
            return requested

        lightweight = getattr(
            get_settings(),
            "huggingface_model_local_fallback_lightweight",
            LIGHTWEIGHT_LOCAL_MODEL,
        )
        logger.warning(
            f"Memoire insuffisante pour {requested} "
            f"({available:.1f} Go disponibles, {threshold:.1f} Go requis). "
            f"Bascule automatique sur {lightweight}."
        )
        return lightweight

    def _ensure_loaded(self) -> None:
        """Charge le modele et le tokenizer en memoire si ce n'est pas deja fait.

        Deux garde-fous successifs pour que le pipeline reste fonctionnel meme
        sur des postes sous-dimensionnes :

        1. **Preflight memoire** : avant le chargement, on compare la RAM/VRAM
           disponible au besoin estime du modele demande. Si la machine n'a pas
           les ressources, on bascule des le depart sur le modele lightweight.
        2. **Fallback a chaud sur OOM** : si l'estimation etait trop optimiste
           et que PyTorch leve un OutOfMemoryError pendant ``from_pretrained``
           ou ``.to(device)``, on capture l'erreur et on retente avec le modele
           lightweight. Cela couvre les cas ou d'autres processus consomment
           de la memoire en parallele.
        """
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device, dtype = self._pick_device()
            settings = get_settings()
            hf_token = settings.huggingface_token or None
            lightweight_name = getattr(
                settings,
                "huggingface_model_local_fallback_lightweight",
                LIGHTWEIGHT_LOCAL_MODEL,
            )

            effective_name = self._resolve_model_name(self.model_name, device)

            def _load(name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
                logger.info(
                    f"Chargement du modele local {name} (device={device}, dtype={dtype})..."
                )
                tok = AutoTokenizer.from_pretrained(
                    name,
                    token=hf_token,
                    trust_remote_code=False,
                )
                mdl = AutoModelForCausalLM.from_pretrained(
                    name,
                    token=hf_token,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=False,
                )
                mdl.to(torch.device(device))
                mdl.eval()
                return tok, mdl

            try:
                tokenizer, model = _load(effective_name)
            except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as err:
                # RuntimeError couvre les OOM CPU sur certains backends ainsi
                # que les erreurs "CUDA out of memory" remontees differemment
                # selon la version de PyTorch/ROCm.
                message = str(err).lower()
                is_oom = (
                    isinstance(err, (torch.cuda.OutOfMemoryError, MemoryError))
                    or "out of memory" in message
                )
                if not is_oom or effective_name == lightweight_name:
                    raise
                logger.warning(
                    f"Echec du chargement de {effective_name} pour cause de "
                    f"memoire insuffisante : {err}. Bascule sur {lightweight_name}."
                )
                # On libere ce qui a pu etre alloue avant la bascule pour
                # maximiser la memoire disponible au second essai.
                try:
                    import gc

                    gc.collect()
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                effective_name = lightweight_name
                tokenizer, model = _load(effective_name)

            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._dtype = dtype
            self.model_name = effective_name

            logger.info(f"Modele local {effective_name} charge et pret a l'inference")

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
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return _fix_mojibake(generated_text)

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
