"""TIES-merging manuel (Yadav et al. NeurIPS 2023, arXiv:2306.01708) sur les
3 adapters Qwen3-4B + LoRA top-1 issus du K=3×2 du 2026-05-17.

Pourquoi un TIES manuel ?
-------------------------

PEFT 0.19.0 refuse explicitement ``add_weighted_adapter(combination_type="ties")``
quand l'un des adapters contient ``modules_to_save`` (la tête de classification
``Linear(hidden, num_labels)`` est wrappée dans un ``ModulesToSaveWrapper``).
Le fallback ``_average_lora_deltas`` produit un "uniform soup" (Wortsman ICML
2022) qui marche mais est sous-optimal (Yadav 2023 : -0.01 à -0.02 MCC vs vrai
TIES). On contourne PEFT et on applique TIES directement sur les
``adapter_model.safetensors``.

Algorithme TIES par tenseur (LoRA A et B uniquement) :

1. **Trim** : par tenseur de chaque adapter, ne garder que la fraction
   ``density`` des paramètres avec la plus grande magnitude. Mettre les autres
   à zéro.
2. **Sign-elect** : par position scalaire, calculer le signe majoritaire
   pondéré par magnitude (somme algébrique). C'est le "signe gagnant".
3. **Disjoint merge** : moyenne arithmétique uniquement sur les contributions
   d'adapters qui s'accordent avec le signe gagnant (les conflits sont
   éliminés).

La **tête de classification** (``base_model.model.score.weight``) est traitée
à part : elle n'est pas un delta LoRA mais la pleine valeur apprise. Lui
appliquer TIES (trim 50%) la dégraderait. On fait donc une moyenne
arithmétique simple des K têtes (cohérent avec linear merge sur cette couche).

Sortie
------

* ``models/qwen3/folds_ties/adapter_ties/`` : adapter LoRA TIES-fusionné
  (rechargeable avec PEFT comme un adapter unique).
* ``models/qwen3/merged/`` : modèle base + adapter fusionné via PEFT
  ``merge_and_unload`` (écrase l'uniform soup précédent).
* ``models/qwen3/ensemble_config.json`` : strategy mise à jour
  ``"ties_manual"``.

Usage
-----

::

    uv run python scripts/ties_merge_qwen3_manual.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
QWEN3_ROOT = BASE_DIR / "models" / "qwen3"
FOLDS_TOP1 = [
    QWEN3_ROOT / "folds" / "fold_1_seed_2",
    QWEN3_ROOT / "folds" / "fold_2_seed_1",
    QWEN3_ROOT / "folds" / "fold_3_seed_2",
]
TIES_ADAPTER_DIR = QWEN3_ROOT / "folds_ties" / "adapter_ties"
MERGED_DIR = QWEN3_ROOT / "merged"
ENSEMBLE_PATH = QWEN3_ROOT / "ensemble_config.json"

DENSITY = 0.5


def ties_merge_tensor(tensors: list[torch.Tensor], density: float = 0.5) -> torch.Tensor:
    """Applique TIES (trim + sign-elect + disjoint merge) sur K tenseurs.

    Implémentation fidèle à Yadav et al. NeurIPS 2023 (Algorithm 1) :

    * Trim par adapter : ``top-k`` par magnitude absolue, ``k = density * numel``.
      Les paramètres non retenus sont mis à zéro.
    * Sign-elect : ``sign(sum(stacked))`` par position après trim. La somme
      algébrique pondère naturellement par magnitude.
    * Disjoint merge : moyenne arithmétique sur les seuls éléments qui
      s'accordent avec le signe gagnant (les autres sont masqués). Évite les
      annulations destructives entre tâches.

    Args:
        tensors: Liste de K tenseurs de même shape (un par adapter).
        density: Fraction conservée par le trim (0.5 = top 50% magnitudes).

    Returns:
        Tenseur fusionné de même shape et dtype que l'entrée.
    """
    target_dtype = tensors[0].dtype
    tensors_f32 = [t.to(torch.float32) for t in tensors]

    trimmed = []
    for t in tensors_f32:
        flat_abs = t.flatten().abs()
        k = max(1, int(density * flat_abs.numel()))
        threshold = torch.topk(flat_abs, k, largest=True).values[-1]
        mask = t.abs() >= threshold
        trimmed.append(t * mask)

    stacked = torch.stack(trimmed)

    sign_sum = stacked.sum(dim=0)
    sign = torch.sign(sign_sum)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    mask_agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
    count_agree = mask_agree.sum(dim=0).clamp(min=1).to(torch.float32)
    merged = (stacked * mask_agree).sum(dim=0) / count_agree

    return merged.to(target_dtype)


def average_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Moyenne arithmétique standard, conserve dtype et device."""
    target_dtype = tensors[0].dtype
    stacked_f32 = torch.stack([t.to(torch.float32) for t in tensors])
    return stacked_f32.mean(dim=0).to(target_dtype)


def is_lora_delta(key: str) -> bool:
    """True pour les clés ``lora_A`` / ``lora_B`` (deltas) — TIES applicable.

    False pour les clés ``score`` / ``classifier`` (tête de classification,
    poids absolus) — appliquer moyenne arithmétique à la place.
    """
    return "lora_A" in key or "lora_B" in key


def apply_ties_to_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Fusionne K state dicts d'adapters via TIES (deltas) + moyenne (tête)."""
    keys = sorted(state_dicts[0].keys())
    for idx, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != set(keys):
            msg = f"Adapter {idx} a une structure de clés différente du premier adapter"
            raise ValueError(msg)

    merged: dict[str, torch.Tensor] = {}
    n_ties = 0
    n_avg = 0
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if is_lora_delta(key):
            merged[key] = ties_merge_tensor(tensors, density=DENSITY)
            n_ties += 1
        else:
            merged[key] = average_tensor(tensors)
            n_avg += 1

    logger.info(f"TIES appliqué sur {n_ties} tenseurs LoRA + moyenne sur {n_avg} tenseurs head")
    return merged


def main() -> int:
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)

    logger.info("#" * 78)
    logger.info("#  TIES-merging manuel Qwen3 (Yadav 2023)")
    logger.info("#" * 78)

    for path in FOLDS_TOP1:
        if not (path / "adapter_model.safetensors").exists():
            msg = f"Adapter manquant : {path}"
            raise FileNotFoundError(msg)

    logger.info(f"Chargement des {len(FOLDS_TOP1)} adapters top-1...")
    state_dicts = [load_file(str(path / "adapter_model.safetensors")) for path in FOLDS_TOP1]

    sample_key = next(iter(state_dicts[0].keys()))
    logger.info(f"Exemple : {len(state_dicts[0])} tenseurs, premier='{sample_key}' shape={tuple(state_dicts[0][sample_key].shape)}")

    merged_state = apply_ties_to_state_dicts(state_dicts)

    TIES_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    save_file(merged_state, str(TIES_ADAPTER_DIR / "adapter_model.safetensors"))
    logger.info(f"Adapter TIES sauvegardé : {TIES_ADAPTER_DIR / 'adapter_model.safetensors'}")

    for fname in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "README.md"):
        src = FOLDS_TOP1[0] / fname
        if src.exists():
            shutil.copy(src, TIES_ADAPTER_DIR / fname)

    logger.info("Chargement base Qwen3-4B + adapter TIES pour merge_and_unload final...")
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    base_model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen3-4B",
        num_labels=2,
        torch_dtype=torch.bfloat16,
    )
    peft_model = PeftModel.from_pretrained(base_model, TIES_ADAPTER_DIR)

    logger.info("merge_and_unload() sur l'adapter TIES unique...")
    merged_model = peft_model.merge_and_unload()

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    logger.info(f"Modèle final TIES sauvegardé : {MERGED_DIR}")

    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = FOLDS_TOP1[0] / fname
        if src.exists():
            shutil.copy(src, MERGED_DIR / fname)

    # Copie la calibration K-fold (T et seuil moyens) depuis la racine du modèle
    # vers merged/ pour que ``get_classifier`` les charge automatiquement après
    # la redirection ``ties_manual`` → ``merged/``. Sans cela, le fallback
    # T=1.0 seuil=0.5 est appliqué et la précision chute drastiquement.
    for fname in ("temperature.json", "optimal_threshold.json"):
        src = QWEN3_ROOT / fname
        if src.exists():
            shutil.copy(src, MERGED_DIR / fname)
            logger.info(f"Calibration copiée : {fname}")

    config = json.loads(ENSEMBLE_PATH.read_text(encoding="utf-8"))
    config["strategy"] = "ties_manual"
    config["metadata"]["ties_density"] = DENSITY
    config["metadata"]["ties_applied_at"] = datetime.now(UTC).isoformat()
    config["metadata"]["fix_applied"] = "TIES manuel sur safetensors (P4.15)"
    config["metadata"]["ties_reference"] = "Yadav et al. NeurIPS 2023, arXiv:2306.01708"
    ENSEMBLE_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info(f"ensemble_config.json mis à jour : strategy='ties_manual'")

    logger.info("=" * 78)
    logger.info("  TIES manuel OK — Qwen3 ensemble final prêt pour P4.5 / P5.1")
    logger.info(f"  Adapter TIES : {TIES_ADAPTER_DIR}")
    logger.info(f"  Modèle fusionné : {MERGED_DIR}")
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
