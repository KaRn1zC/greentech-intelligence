"""Perturbations de texte pour tests de robustesse du classifieur (B3.6).

Ce module implemente trois techniques de perturbation textuelle utilisees
pour mesurer la robustesse du classifieur Green IT au bruit reel observe
en production : typos, variations de casing, ponctuation aleatoire.

Reference principale : AEDA (Karimi et al. 2021, arXiv:2108.13230) -
"An Easier Data Augmentation Technique" - insertion aleatoire de
signes de ponctuation sans modification semantique.

Utilisations
------------

1. **Tests de robustesse** : mesurer la baisse de MCC entre predictions
   sur le test set original et sur le test set perturbe. Une baisse > 10
   points de MCC indique un modele fragile au bruit.
2. **Data augmentation** complementaire a la back-translation (B3.3).
   Note : non utilise dans le pipeline d'entrainement actuel, conserve
   ici comme outil de validation.

Les fonctions sont **deterministes** par construction quand un seed est
fourni : la meme entree produit la meme sortie, essentielle pour la
reproductibilite des tests Deepchecks.

"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Caracteres de ponctuation autorises pour AEDA (Karimi et al. 2021).
AEDA_PUNCTUATION_MARKS: tuple[str, ...] = (".", ",", "!", "?", ";", ":")


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration commune aux fonctions de perturbation.

    Attributes:
        intensity: Probabilite par mot (ou caractere selon la fonction) d'etre
            perturbe. Plage [0.0, 1.0]. 0.0 = aucune perturbation. 0.1 =
            recommande pour des tests de robustesse realistes.
        seed: Graine pour la reproductibilite. ``None`` = non deterministe.
    """

    intensity: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity doit etre dans [0.0, 1.0], recu {self.intensity}")


def apply_aeda(text: str, config: PerturbationConfig | None = None) -> str:
    """Insere des signes de ponctuation aleatoires entre les mots (AEDA).

    Reference : Karimi et al. 2021, "AEDA: An Easier Data Augmentation
    Technique for Text Classification" (arXiv:2108.13230).

    L'idee : insertions aleatoires de ``.``, ``,``, ``!``, ``?``, ``;``, ``:``
    a des positions inter-mots avec une probabilite par espace egale a
    ``config.intensity``. Aucun mot n'est modifie. Aucun caractere n'est
    supprime. La structure semantique est preservee, seule la ponctuation
    devient bruitee, ce qui simule du contenu mal edite ou des transcriptions
    automatiques.

    Args:
        text: Texte source.
        config: Parametres de perturbation. ``None`` = defaut (10 %, seed
            non fixe).

    Returns:
        Texte perturbe. Vide si ``text`` est vide.
    """
    if not text:
        return text
    cfg = config or PerturbationConfig()
    rng = random.Random(cfg.seed) if cfg.seed is not None else random

    words = text.split(" ")
    result: list[str] = []
    for i, word in enumerate(words):
        result.append(word)
        # Insertion entre mots (pas apres le dernier)
        if i < len(words) - 1 and rng.random() < cfg.intensity:
            mark = rng.choice(AEDA_PUNCTUATION_MARKS)
            result.append(mark)
    return " ".join(result)


def apply_random_casing(text: str, config: PerturbationConfig | None = None) -> str:
    """Inverse aleatoirement la casse de certains caracteres alphabetiques.

    Simule des erreurs de saisie (caps lock accidentel, smartphone autocaps,
    OCR sur fonte mixte). Chaque caractere alphabetique a une probabilite
    ``config.intensity`` de voir sa casse inversee. Les autres caracteres
    sont conserves intacts.

    Args:
        text: Texte source.
        config: Parametres de perturbation.

    Returns:
        Texte perturbe.
    """
    if not text:
        return text
    cfg = config or PerturbationConfig()
    rng = random.Random(cfg.seed) if cfg.seed is not None else random

    chars: list[str] = []
    for ch in text:
        if ch.isalpha() and rng.random() < cfg.intensity:
            chars.append(ch.lower() if ch.isupper() else ch.upper())
        else:
            chars.append(ch)
    return "".join(chars)


def apply_typos(text: str, config: PerturbationConfig | None = None) -> str:
    """Introduit des typos realistes par swap, delete et insert de caracteres.

    Trois operations possibles, choisies uniformement parmi les mots eligibles
    a la perturbation :

    1. **Swap** : echange deux caracteres adjacents (frequent au clavier).
    2. **Delete** : supprime un caractere (frappe loupee).
    3. **Insert** : duplique un caractere adjacent (touche tenue).

    Les mots de moins de 4 caracteres ne sont pas perturbes (trop court
    pour preserver le sens).

    Args:
        text: Texte source.
        config: Parametres de perturbation. ``intensity`` est la probabilite
            qu'un mot >= 4 caracteres soit altere.

    Returns:
        Texte perturbe.
    """
    if not text:
        return text
    cfg = config or PerturbationConfig()
    rng = random.Random(cfg.seed) if cfg.seed is not None else random

    words = text.split(" ")
    result: list[str] = []
    for word in words:
        if len(word) < 4 or rng.random() >= cfg.intensity:
            result.append(word)
            continue

        op = rng.choice(["swap", "delete", "insert"])
        # Eviter de toucher au tout premier et au tout dernier caractere
        # pour preserver la majuscule initiale et la ponctuation collee finale.
        idx = rng.randint(1, len(word) - 2)

        if op == "swap":
            chars = list(word)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            result.append("".join(chars))
        elif op == "delete":
            result.append(word[:idx] + word[idx + 1 :])
        else:  # insert
            result.append(word[:idx] + word[idx] + word[idx:])

    return " ".join(result)


def apply_combined_perturbations(
    text: str,
    config: PerturbationConfig | None = None,
) -> str:
    """Applique sequentiellement AEDA, casing et typos avec la meme intensite.

    Utilise pour les tests de robustesse "worst case" : un texte fortement
    bruite que le modele doit encore classer correctement.

    Args:
        text: Texte source.
        config: Configuration commune aux 3 perturbations.

    Returns:
        Texte ayant subi les 3 perturbations dans l'ordre AEDA -> casing -> typos.
    """
    cfg = config or PerturbationConfig()
    out = apply_aeda(text, cfg)
    out = apply_random_casing(out, cfg)
    out = apply_typos(out, cfg)
    return out
