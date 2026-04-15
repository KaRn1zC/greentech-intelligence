"""Verification Green IT par LLM judge (etage 2 du pipeline de classification).

Ce module constitue le **second etage** du pipeline de classification hybride.
Il prend en entree les articles marques `CANDIDATE` par le pre-filtre mots-cles
(`scripts/auto_annotate_dataset.py`) et les soumet a un LLM instructif
(`Qwen/Qwen3-4B-Instruct-2507` via l'API Hugging Face Serverless, avec
fallback local Qwen2.5-3B/1.5B si le quota est epuise) pour obtenir
une decision binaire finale.

Pourquoi un LLM en seconde passe
--------------------------------

Le pre-filtre mots-cles est volontairement permissif : tout article presentant
ne serait-ce qu'un soupcon de signal Green est envoye ici pour verification.
Le LLM lit le texte en contexte et tranche :

- `est_green_it = True` : l'article parle principalement de Green IT tel
  qu'il est defini dans le prompt systeme (efficacite energetique des
  infrastructures IT, empreinte carbone du numerique, sobriete numerique,
  eco-conception logicielle, e-waste, etc.).
- `est_green_it = False` : l'article utilise un vocabulaire Green IT de
  facon periphrastique ou marginale, sans en faire son sujet principal.

Le LLM retourne aussi un score de confiance et une courte justification.
Ces elements sont persistes en base pour la tracabilite.

Format d'echange
----------------

Sortie structuree en JSON (le LLM est contraint a ce format par le prompt
systeme, avec parseur tolerant cote Python pour encaisser un eventuel texte
de prologue/epilogue).

"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

from loguru import logger

from greentech.ai.services.llm_dispatcher import (
    chat_completion,
    is_hf_quota_exhausted,
)
from greentech.config import get_settings

# === Bornes d'entree ===
# Qwen3-4B-Instruct-2507 supporte 262k tokens de contexte mais on tronque a
# 3000 caracteres pour controler la latence et la consommation du quota HF
# (et rester aligne avec ce que le fallback local Qwen2.5-3B digere sans peine).
MAX_INPUT_CHARS = 3000
MIN_INPUT_CHARS = 50

# === Prompt systeme (definition permissive du Green IT) ===
# On accepte qu'un article soit Green IT des lors qu'une part significative
# (pas forcement la totalite) de son contenu traite d'un aspect Green IT.
# L'objectif est de maximiser le recall : en cas de doute, le LLM doit
# pencher vers la classification Green IT plutot que l'exclure.
CLASSIFIER_SYSTEM_PROMPT = (
    "Tu es un expert en Green IT charge de classifier des articles "
    "technologiques. Ta mission : determiner si un article aborde le Green IT "
    "de maniere significative.\n\n"
    "Definition retenue du Green IT (inclusive) :\n"
    "Un article est Green IT s'il traite, meme partiellement mais de facon "
    "substantielle, d'un des themes suivants :\n"
    "  - reduction de la consommation energetique ou de l'empreinte carbone "
    "des infrastructures numeriques (data centers, cloud, reseaux) ;\n"
    "  - efficacite energetique des materiels IT (serveurs, GPU, accelerateurs, "
    "puces basse consommation, refroidissement vert) ;\n"
    "  - sobriete numerique, eco-conception logicielle, optimisation "
    "energetique de modeles IA/ML (quantization, pruning, distillation, "
    "compression de modeles visant l'energie, IA frugale) ;\n"
    "  - mesure, suivi, reporting de l'empreinte carbone du numerique ;\n"
    "  - e-waste, economie circulaire des equipements electroniques, "
    "refurbishing, durabilite du materiel, sustainable hardware ;\n"
    "  - energies renouvelables dans un contexte numerique "
    "(data center solaire, cloud bas carbone, hydrogene vert pour data centers) ;\n"
    "  - usage du numerique pour la transition ecologique quand l'angle IT "
    "est clairement present (pas seulement une mention accessoire).\n\n"
    "Sont exclus (Non Green IT) :\n"
    "  - recherche IA/ML purement theorique portant uniquement sur la "
    "precision ou la complexite algorithmique, sans consideration energetique ;\n"
    "  - cryptomonnaies, rapports boursiers et previsions de marche ;\n"
    "  - cybersecurite pure, gaming, metaverse, reseaux sociaux, "
    "smartphones grand public ;\n"
    "  - energies renouvelables ou vehicules electriques sans lien avec "
    "le numerique ;\n"
    "  - sujets sante ou sciences appliquees qui utilisent simplement de l'IA "
    "sans aborder son impact environnemental.\n\n"
    "Regles de decision :\n"
    "  - En cas de doute raisonnable, classe comme Green IT (est_green_it=true).\n"
    "  - Un article qui mentionne de facon non accessoire l'energie, le carbone, "
    "la durabilite ou l'efficacite energetique dans un contexte numerique est "
    "Green IT, meme si ce n'est pas son unique sujet.\n"
    "  - Un article de recherche ML qui optimise la consommation ou reduit "
    "l'empreinte carbone d'un modele est Green IT.\n\n"
    "Tu dois repondre UNIQUEMENT avec un objet JSON valide, sans texte avant "
    "ni apres, suivant ce schema exact :\n"
    '{"est_green_it": true|false, "confiance": 0.0 a 1.0, '
    '"raison": "explication courte en francais"}'
)

CLASSIFIER_USER_PROMPT_TEMPLATE = (
    "Classifie l'article suivant comme Green IT ou non, selon la definition "
    "donnee. Rappelle-toi : en cas de doute raisonnable, penche vers Green IT. "
    "Renvoie uniquement le JSON demande.\n\n"
    "Titre : {titre}\n\n"
    "Contenu :\n{contenu}\n\n"
    "JSON de classification :"
)

CLASSIFIER_MAX_NEW_TOKENS = 220
# Legerement moins deterministe que precedemment pour laisser plus de place
# aux nuances sur les cas ambigus. Reste tres faible pour un LLM.
CLASSIFIER_TEMPERATURE = 0.2


@dataclass(frozen=True)
class ClassifierVerdict:
    """Verdict rendu par le LLM judge pour un article candidat.

    Attributes:
        est_green_it: Decision binaire finale.
        confiance: Score de confiance du LLM dans sa decision (0.0 a 1.0).
        raison: Courte justification textuelle fournie par le modele.
        modele: Identifiant du modele utilise.
        temps_ms: Temps d'inference en millisecondes.
        succes: False en cas d'echec de parsing ou d'appel API.
        erreur: Message d'erreur si `succes` est False.
    """

    est_green_it: bool | None
    confiance: float
    raison: str
    modele: str
    temps_ms: int
    succes: bool
    erreur: str | None = None


def _truncate_content(text: str) -> str:
    """Tronque le texte a MAX_INPUT_CHARS en coupant sur une fin de phrase."""
    if len(text) <= MAX_INPUT_CHARS:
        return text
    truncated = text[:MAX_INPUT_CHARS]
    last_period = truncated.rfind(".")
    if last_period > MAX_INPUT_CHARS // 2:
        truncated = truncated[: last_period + 1]
    return truncated


# Extrait le premier objet JSON trouve dans une reponse textuelle, meme si
# le modele a ajoute du prologue ou de l'epilogue (markdown, "Voici le JSON :"
# etc.). On reste robuste face a des derapages de formatage.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# Repare les backslashes suivis d'un caractere non reconnu par la spec JSON.
# Les sequences valides sont \", \\, \/, \b, \f, \n, \r, \t, \uXXXX.
_INVALID_ESCAPE_RE = re.compile(r'\\(?![\\"/bfnrtu])')
# Extraction regex de secours quand json.loads refuse toutes les tentatives.
_FALLBACK_GREEN_RE = re.compile(r'"est[_ ]green[_ ]it"\s*:\s*(true|false)', re.IGNORECASE)
_FALLBACK_CONF_RE = re.compile(r'"confiance"\s*:\s*([\d.]+)')
_FALLBACK_REASON_RE = re.compile(r'"raison"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _fallback_regex_parse(text: str) -> dict[str, object] | None:
    """Extraction tolerante par regex lorsque json.loads echoue.

    Utile quand le LLM genere du texte libre autour des champs attendus
    (markdown, retours a la ligne internes, backslashes parasites) mais
    que les champs sont presents en sequence. Seul `est_green_it` est
    obligatoire : sans lui, on renvoie None et l'appelant re-leve.
    """
    green_match = _FALLBACK_GREEN_RE.search(text)
    if not green_match:
        return None
    result: dict[str, object] = {"est_green_it": green_match.group(1).lower() == "true"}
    conf_match = _FALLBACK_CONF_RE.search(text)
    if conf_match:
        try:
            result["confiance"] = float(conf_match.group(1))
        except ValueError:
            result["confiance"] = 0.5
    reason_match = _FALLBACK_REASON_RE.search(text)
    if reason_match:
        result["raison"] = reason_match.group(1).replace('\\"', '"')
    return result


def _parse_verdict(raw: str) -> tuple[bool | None, float, str]:
    """Parse la reponse brute du LLM et extrait (est_green_it, confiance, raison).

    Strategie de parsing robuste en trois passes :

    1. `json.loads` direct sur la premiere region `{...}` trouvee.
    2. Si echec : nettoyage des backslashes invalides (tres frequent avec
       les LLM qui produisent du JSON sans respecter strictement la spec).
    3. Si echec : extraction par regex des trois champs attendus.

    Raises:
        ValueError: Si aucune des trois passes ne permet de recuperer au
            moins le champ booleen `est_green_it`.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        raise ValueError(f"Aucun JSON trouve dans la reponse : {raw[:200]!r}")

    json_str = match.group(0)
    payload: dict[str, object] | None = None

    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = _INVALID_ESCAPE_RE.sub(r"\\\\", json_str)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = _fallback_regex_parse(json_str)

    if payload is None:
        raise ValueError(
            f"Impossible de parser la reponse LLM meme en mode tolerant : {raw[:200]!r}"
        )

    est_green = payload.get("est_green_it")
    if not isinstance(est_green, bool):
        raise ValueError(f"Champ 'est_green_it' absent ou non booleen : {payload!r}")

    confiance = payload.get("confiance", 0.5)
    try:
        confiance = float(confiance)
    except (TypeError, ValueError):
        confiance = 0.5
    confiance = max(0.0, min(1.0, confiance))

    raison = str(payload.get("raison", "") or "").strip()
    if len(raison) > 500:
        raison = raison[:497] + "..."

    return est_green, confiance, raison


# Nombre maximal de tentatives par article en cas d'erreur transitoire
# (rate limit, timeout, 5xx). Avec un backoff exponentiel de 2^n secondes,
# trois tentatives couvrent 2+4 = 6s de delai total supplementaire.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0


async def verify_green_it_candidate(
    titre: str,
    contenu: str,
) -> ClassifierVerdict:
    """Interroge le LLM judge pour classifier definitivement un candidat.

    Tente jusqu'a `MAX_RETRIES` fois en cas d'erreur transitoire (rate limit
    HF, timeout reseau) avec un backoff exponentiel. Les erreurs de parsing
    du JSON de reponse ne declenchent pas de retry. En cas d'epuisement du
    quota HF mensuel (HTTP 402), le dispatcher bascule automatiquement sur
    le modele Qwen local (GPU AMD ROCm) sans modification du code appelant.

    Args:
        titre: Titre de l'article.
        contenu: Contenu complet (sera tronque si necessaire).

    Returns:
        Verdict structure du LLM. En cas d'echec apres tous les retries,
        `succes=False` et `est_green_it=None` : l'appelant doit decider
        quoi faire (laisser l'article en attente pour un prochain run).
    """
    settings = get_settings()
    modele = settings.huggingface_model_classifier_llm

    # Garde-fou : entree trop courte = decision par defaut NON Green IT
    if not contenu or len(contenu.strip()) < MIN_INPUT_CHARS:
        return ClassifierVerdict(
            est_green_it=False,
            confiance=0.5,
            raison="Contenu trop court pour classification fiable",
            modele=modele,
            temps_ms=0,
            succes=True,
        )

    contenu_tronque = _truncate_content(contenu)
    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CLASSIFIER_USER_PROMPT_TEMPLATE.format(
                titre=titre or "(titre manquant)",
                contenu=contenu_tronque,
            ),
        },
    ]

    last_error: str | None = None

    for attempt in range(MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = await chat_completion(
                messages=messages,
                max_tokens=CLASSIFIER_MAX_NEW_TOKENS,
                temperature=CLASSIFIER_TEMPERATURE,
                model_hf=modele,
            )
            temps_ms = int((time.perf_counter() - start) * 1000)

            raw_text = (response.choices[0].message.content or "").strip()
            est_green, confiance, raison = _parse_verdict(raw_text)

            backend = "qwen_local" if is_hf_quota_exhausted() else modele
            return ClassifierVerdict(
                est_green_it=est_green,
                confiance=confiance,
                raison=raison or "Reponse LLM sans justification",
                modele=backend,
                temps_ms=temps_ms,
                succes=True,
            )

        except ValueError as exc:
            # Parsing impossible : erreur deterministe, pas de retry
            last_error = f"Parsing: {exc}"
            logger.warning(f"Parsing LLM echoue : {exc}")
            break

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.debug(
                    f"Tentative {attempt + 1} echouee ({exc.__class__.__name__}), "
                    f"retry dans {wait:.1f}s"
                )
                await asyncio.sleep(wait)
            else:
                logger.warning(f"Echec verification LLM apres {MAX_RETRIES} tentatives : {exc!r}")

    return ClassifierVerdict(
        est_green_it=None,
        confiance=0.0,
        raison="",
        modele=modele,
        temps_ms=0,
        succes=False,
        erreur=last_error,
    )


async def verify_green_it_batch(
    articles: list[tuple[int, str, str]],
    *,
    delay_seconds: float = 0.5,
) -> dict[int, ClassifierVerdict]:
    """Traite un lot d'articles sequentiellement avec fallback local automatique.

    Le dispatcher bascule de lui-meme sur le modele Qwen local (GPU AMD
    ROCm) lorsque le quota mensuel HF est epuise. Les appels restent
    sequentiels : cela respecte le fair-use cote HF, et en mode local le
    GPU constitue de toute facon un goulet naturel.

    Args:
        articles: Liste de tuples (id_article, titre, contenu).
        delay_seconds: Pause entre chaque appel. Neutralisee automatiquement
            en mode local (plus de rate limit a respecter).

    Returns:
        Dictionnaire {id_article: verdict}.
    """
    verdicts: dict[int, ClassifierVerdict] = {}

    for index, (id_article, titre, contenu) in enumerate(articles, start=1):
        logger.debug(f"[{index}/{len(articles)}] Verification article {id_article}")
        verdict = await verify_green_it_candidate(titre=titre, contenu=contenu)
        verdicts[id_article] = verdict

        if index < len(articles) and delay_seconds > 0 and not is_hf_quota_exhausted():
            await asyncio.sleep(delay_seconds)

    return verdicts
