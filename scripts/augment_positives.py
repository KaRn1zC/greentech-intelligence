"""Augmente les articles Green IT positifs via back-translation EN<->FR.

Execute la back-translation opus-mt sur les positifs du ``golden_dataset.csv``
pour doubler la classe minoritaire (1 018 -> ~2 036 positifs), ramenant le
ratio de desequilibre de 1:10.5 a ~1:5.25.

Le script produit ``data/golden_dataset_augmented.csv`` (union des originaux
+ variantes acceptees), prive du champ ``score_confiance`` et
``modele_classification`` pour les variantes (elles heritent seulement du
label). Une colonne supplementaire ``augmentation_source`` distingue :

- ``""`` (vide) : article original issu de la classification B2.9
- ``"opus-mt-backtranslation"`` : variante generee par ce script

Les scripts de K-fold traitement doivent **exclure ces variantes du val/test**
split pour eviter la fuite d'evaluation (on ne peut pas tester sur un texte
derivé d'un texte vu en train). Concretement, la stratification se fait sur
les originaux puis on ajoute les variantes au train split uniquement.

Usage
-----

    # Augmentation standard (1 variante par positif)
    uv run python scripts/augment_positives.py

    # Dry-run (compte sans ecrire)
    uv run python scripts/augment_positives.py --dry-run

    # Overrider les seuils de similarite
    uv run python scripts/augment_positives.py --sim-min 0.90 --sim-max 0.98

Performance
-----------

Sur RX 7900 XTX 24 Go avec batch=16 : ~20-30 min pour les 1 018 positifs
(2 cycles de traduction par article, ~75M params par direction).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from loguru import logger

# Force UTF-8 sur stdout/stderr (meme raison que retrain_pipeline.py :
# cp1252 sur Windows casse l'affichage des titres FR et \ufffd).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from greentech.ai.mlops.tracking import ExperimentConfig, tracked_experiment  # noqa: E402
from greentech.config import BASE_DIR  # noqa: E402
from greentech.data.processors.back_translator import (  # noqa: E402
    DEFAULT_SIMILARITY_MAX,
    DEFAULT_SIMILARITY_MIN,
    BackTranslator,
)
from greentech.utils.logger import setup_logging  # noqa: E402

INPUT_FILE = BASE_DIR / "data" / "golden_dataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "golden_dataset_augmented.csv"

# Colonnes du CSV d'entree (cf. scripts/export_golden_dataset.py)
INPUT_COLUMNS = [
    "id_article",
    "titre",
    "url",
    "resume_classification",
    "source_nom",
    "date_publication",
    "langue",
    "label_green_it",
    "score_confiance",
    "modele_classification",
]

# Colonnes du CSV de sortie (ajoute augmentation_source)
OUTPUT_COLUMNS = [*INPUT_COLUMNS, "augmentation_source"]


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Augmente les positifs via back-translation EN<->FR (opus-mt)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"CSV golden d'entree (defaut: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"CSV augmente de sortie (defaut: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--sim-min",
        type=float,
        default=DEFAULT_SIMILARITY_MIN,
        help=(
            f"Similarite cosine minimale pour accepter une variante "
            f"(defaut: {DEFAULT_SIMILARITY_MIN})"
        ),
    )
    parser.add_argument(
        "--sim-max",
        type=float,
        default=DEFAULT_SIMILARITY_MAX,
        help=(
            f"Similarite cosine maximale (au-dela c'est un quasi-duplicata "
            f"sans interet, defaut: {DEFAULT_SIMILARITY_MAX})"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Taille de batch MarianMT (defaut: 16)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne pas ecrire le CSV, juste compter et afficher les stats.",
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> list[dict[str, str]]:
    """Charge le CSV golden comme liste de dict."""
    if not input_path.exists():
        msg = (
            f"CSV golden introuvable : {input_path}. "
            "Lancer 'uv run python scripts/export_golden_dataset.py' avant."
        )
        raise FileNotFoundError(msg)

    with input_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Charge : {len(rows)} articles depuis {input_path}")
    return rows


def filter_positives(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Garde uniquement les articles avec label_green_it=1 et langue en/fr.

    Les autres langues (si elles reapparaissent apres B2.9) sont loguees
    et ignorees car hors perimetre des modeles opus-mt utilises.
    """
    positives = [row for row in rows if row["label_green_it"] == "1"]
    logger.info(f"  dont {len(positives)} positifs (label_green_it=1)")

    supported = [row for row in positives if row["langue"] in ("en", "fr")]
    ignored = [row for row in positives if row["langue"] not in ("en", "fr")]

    if ignored:
        logger.warning(
            f"  {len(ignored)} positifs ignores (langue hors en/fr) : "
            f"{sorted({row['langue'] for row in ignored})}"
        )
    return supported


def write_augmented_csv(
    output_path: Path,
    original_rows: list[dict[str, str]],
    variant_rows: list[dict[str, str]],
) -> None:
    """Ecrit le CSV augmente : originaux (tous) + variantes acceptees (positifs)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        # Originaux : augmentation_source vide
        for row in original_rows:
            enriched = dict(row)
            enriched["augmentation_source"] = ""
            writer.writerow(enriched)

        # Variantes : heritent du label et des metadonnees, nouveau resume
        for row in variant_rows:
            writer.writerow(row)

    logger.info(f"CSV augmente ecrit : {output_path}")


def build_variant_row(
    original_row: dict[str, str],
    augmented_resume: str,
    similarity: float,
) -> dict[str, str]:
    """Construit la ligne CSV d'une variante a partir de l'original."""
    return {
        # id_article negatif pour distinguer les variantes des originaux sans
        # collision possible (les ids originaux sont positifs, cf. sequences PG).
        # On garde l'id original en valeur absolue pour tracer la filiation.
        "id_article": f"-{original_row['id_article']}",
        "titre": original_row["titre"],
        "url": original_row["url"],
        "resume_classification": augmented_resume,
        "source_nom": original_row["source_nom"],
        "date_publication": original_row["date_publication"],
        "langue": original_row["langue"],
        "label_green_it": original_row["label_green_it"],
        # Les variantes n'ont pas de score LLM (pas re-classifiees)
        "score_confiance": "",
        "modele_classification": "opus-mt-backtranslation",
        "augmentation_source": f"opus-mt-backtranslation-sim{similarity:.3f}",
    }


def main() -> int:
    """Point d'entree CLI."""
    # Active le logging console + fichier rotatif logs/greentech_<date>.log
    # (+ Loki si accessible). Sans cet appel, loguru n'ecrit que sur stderr
    # et le script n'est pas monitorable via tail -F du fichier de log.
    setup_logging()
    args = parse_args()

    logger.info("=" * 70)
    logger.info("  AUGMENTATION PAR BACK-TRANSLATION EN<->FR")
    logger.info("=" * 70)
    logger.info(f"  Input  : {args.input}")
    logger.info(f"  Output : {args.output}")
    logger.info(f"  Similarite acceptee : [{args.sim_min}, {args.sim_max}]")
    logger.info(f"  Dry-run : {args.dry_run}")

    # Charger + filtrer
    all_rows = load_dataset(args.input)
    positives = filter_positives(all_rows)
    if not positives:
        logger.error("Aucun positif a augmenter. Abandon.")
        return 1

    # Back-translation
    bt = BackTranslator(
        similarity_min=args.sim_min,
        similarity_max=args.sim_max,
        batch_size=args.batch_size,
    )

    texts = [row["resume_classification"] for row in positives]
    languages = [row["langue"] for row in positives]

    exp_config = ExperimentConfig(
        nom_experience="greentech-classification",
        nom_run=f"augment-back-translation-{int(time.time())}",
        tags={
            "phase": "b3-augmentation",
            "method": "opus-mt-backtranslation",
        },
        params={
            "similarity_min": args.sim_min,
            "similarity_max": args.sim_max,
            "batch_size": args.batch_size,
            "n_positives_input": len(positives),
        },
    )

    with tracked_experiment(exp_config):
        import mlflow

        accepted, stats = bt.augment(texts=texts, languages=languages)
        mlflow.log_metrics({k: float(v) for k, v in stats.to_dict().items()})

    logger.info("")
    logger.info("Statistiques finales :")
    for key, value in stats.to_dict().items():
        if "duration" in key:
            logger.info(f"  {key:40s} : {value:.1f} s")
        else:
            logger.info(f"  {key:40s} : {value}")

    if args.dry_run:
        logger.info("[DRY-RUN] Aucun fichier ecrit.")
        return 0

    if not accepted:
        logger.warning("Aucune variante acceptee, pas d'augmentation produite.")
        # On ecrit quand meme le CSV de sortie pour que le pipeline puisse
        # continuer (sera identique au golden original + colonne augmentation_source).
        write_augmented_csv(args.output, all_rows, variant_rows=[])
        return 0

    # Aligner chaque variante acceptee avec son article source via le texte
    # original (les resultats gardent l'ordre d'entree dans la meme langue,
    # mais apres filtrage tous les indices ne correspondent plus un-a-un ;
    # on reconstruit via hash du texte original).
    text_to_row = {row["resume_classification"]: row for row in positives}
    variant_rows: list[dict[str, str]] = []
    for result in accepted:
        original_row = text_to_row.get(result.original_text)
        if original_row is None:
            # Ne devrait jamais arriver car on transmet exactement les textes
            # charges. Safety net pour attraper un bug de back_translator.
            logger.warning(
                f"Variante acceptee sans article source correspondant, ignoree "
                f"(len={len(result.original_text)})"
            )
            continue
        variant_rows.append(
            build_variant_row(
                original_row=original_row,
                augmented_resume=result.augmented_text,
                similarity=result.similarity,
            )
        )

    write_augmented_csv(args.output, all_rows, variant_rows)

    logger.info("")
    logger.info("Bilan :")
    logger.info(f"  Articles originaux conserves  : {len(all_rows)}")
    logger.info(f"  Variantes generees acceptees  : {len(variant_rows)}")
    logger.info(f"  Total dataset augmente        : {len(all_rows) + len(variant_rows)}")
    pos_original = sum(1 for r in all_rows if r["label_green_it"] == "1")
    pos_augmented = pos_original + len(variant_rows)
    total_augmented = len(all_rows) + len(variant_rows)
    ratio = (total_augmented - pos_augmented) / max(pos_augmented, 1)
    logger.info(
        f"  Ratio desequilibre positifs   : 1:{ratio:.1f} "
        f"(etait 1:{(len(all_rows) - pos_original) / max(pos_original, 1):.1f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
