"""Pipeline complet de re-collecte, re-annotation et re-entrainement.

Orchestre toutes les etapes pour agrandir le corpus, re-annoter le dataset,
re-entrainer le modele Qwen3-4B + LoRA, benchmarker la nouvelle version
contre la meilleure version historique et le modele de base, puis promouvoir
automatiquement en production UNIQUEMENT si la nouvelle version est meilleure.

Le pipeline garantit que l'application utilise toujours la meilleure version
entrainee du modele, jamais une regression.

Le modele cible est defini dans ``settings.huggingface_model_trainer_base``
(par defaut ``Qwen/Qwen3-4B``, Apache-2.0, multilingue natif). L'ancien
modele ``meta-llama/Llama-3.2-3B`` reste disponible en tant que challenger
historique via ``greentech.ai.models.training challenger-llama``.

Usage:
    # Pipeline complet recommande
    uv run python scripts/retrain_pipeline.py

    # Etapes individuelles
    uv run python scripts/retrain_pipeline.py collect            # Collecte depuis les sources
    uv run python scripts/retrain_pipeline.py clean              # Supprime articles inexploitables (< 50 chars, NULL, placeholder)
    uv run python scripts/retrain_pipeline.py summarize-classif  # Resume LLM pour tous les articles (feature d'entrainement)
    uv run python scripts/retrain_pipeline.py annotate           # Etage 1 : pre-filtre mots-cles (binaire)
    uv run python scripts/retrain_pipeline.py classify           # Etage 2 : LLM judge sur les candidats
    uv run python scripts/retrain_pipeline.py summarize-green    # Resume ecologique pour les Green IT confirmes
    uv run python scripts/retrain_pipeline.py summarize          # Alias : enchaine summarize-classif puis summarize-green
    uv run python scripts/retrain_pipeline.py export-golden      # Regenere golden_dataset.csv depuis la DB
    uv run python scripts/retrain_pipeline.py baseline           # Metriques Qwen3-4B sans fine-tuning
    uv run python scripts/retrain_pipeline.py train              # Entrainement rapide (split 80/20)
    uv run python scripts/retrain_pipeline.py train-cv           # Entrainement robuste (K-fold stratifie K=5)
    uv run python scripts/retrain_pipeline.py benchmark          # Benchmark nouveau vs production vs baseline
    uv run python scripts/retrain_pipeline.py auto-promote       # Benchmark + promotion conditionnelle
    uv run python scripts/retrain_pipeline.py promote            # Promotion manuelle (forcee)

    # Ingerer un fichier manuellement
    uv run python scripts/retrain_pipeline.py ingest-file data/mon_fichier.json

    # Combiner des etapes (workflow recommande apres ajout de nouvelles donnees)
    uv run python scripts/retrain_pipeline.py collect clean summarize-classif annotate classify summarize-green export-golden train-cv auto-promote

Notes:
    - `train`    : entrainement 80/20 rapide, evaluation fragile (4 Green IT au test)
    - `train-cv` : entrainement K-fold (~5x plus long) mais evaluation robuste sur
                   les 22 Green IT grace a la rotation train/test et a la moyenne
                   sur les folds. Recommande pour figer une version du modele.
    - `baseline` : evalue Qwen3-4B brut sur l'integralite du dataset (5808 articles).
                   Pas de data leakage car le modele n'a rien appris.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from loguru import logger

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossiers de modeles
MODELS_DIR = BASE_DIR / "models"
PRODUCTION_DIR = MODELS_DIR / "production"
VERSIONS_DIR = MODELS_DIR / "versions"
# Dossier du modele actuellement entraine par le pipeline. Qwen3-4B + LoRA
# remplace l'ancien Llama 3.2 3B + LoRA comme base d'entrainement par defaut.
# L'ancien dossier `challenger-llama` reste accessible pour les runs legacy
# lances via `uv run python -m greentech.ai.models.training challenger-llama`.
TRAIN_DIR = MODELS_DIR / "challenger-qwen3"
# Identifiant du modele dans le registre d'entrainement (`training.py`).
TRAIN_MODEL_TYPE = "challenger-qwen3"

# Fichiers de reference pour les metriques
BEST_METRICS_FILE = MODELS_DIR / "best_metrics.json"
BASELINE_METRICS_FILE = MODELS_DIR / "baseline_metrics.json"

# =============================================================================
# CRITERES DE PROMOTION
# =============================================================================
# Contexte : dataset fortement desequilibre (~0.4% de Green IT).
# Dans ce cas, le F1 seul est bruite (22 positifs = chaque prediction pese lourd).
# On combine trois criteres pour garantir qu'un modele promu est reellement meilleur
# ET reste utile metier (ne triche pas en predisant tout en Non Green IT).
#
#   1. MCC (Matthews Correlation Coefficient) : metrique principale, robuste au
#      desequilibre. Bornee entre -1 (pire que l'aleatoire) et +1 (parfait).
#      Un modele qui predit une seule classe obtient MCC = 0.
#
#   2. Recall Green IT : garde-fou metier. Rater un article Green IT (faux negatif)
#      est plus couteux que de produire un faux positif, donc on impose un plancher.
#
#   3. F1 (non-regression) : on tolere une legere baisse du F1 si MCC progresse,
#      mais on refuse une regression majeure.
MCC_EPSILON = 0.01  # Tolerance d'egalite sur le MCC (absorbe le bruit statistique)
MIN_RECALL_GREEN_IT = 0.5  # Plancher de recall sur la classe minoritaire
F1_REGRESSION_TOLERANCE = 0.95  # On accepte jusqu'a -5% de F1 si MCC progresse

# Plafond d'ecart-type MCC entre folds (garde-fou de stabilite CV).
# Le seuil est adaptatif : avec un dataset tres desequilibre (peu de positifs),
# chaque fold n'a qu'une poignee de Green IT en test, ce qui rend la variance
# inter-folds mecaniquement elevee meme pour un bon modele. Tant qu'on n'a pas
# au moins 50 Green IT, on tolere un ecart-type de 0.25 ; au-dela, on revient
# au seuil strict 0.15 qui detecte une vraie instabilite du modele.
SMALL_DATASET_GREEN_THRESHOLD = 50
MAX_MCC_STD_SMALL = 0.25  # Seuil tolerant pour datasets avec peu de Green IT
MAX_MCC_STD_LARGE = 0.15  # Seuil strict pour datasets de taille suffisante


def _max_mcc_std(n_green: int | None) -> float:
    """Calcule le plafond MCC std a appliquer en fonction du nombre de positifs.

    Args:
        n_green: Nombre de Green IT dans le dataset complet (ou None si inconnu).

    Returns:
        0.25 si moins de 50 Green IT, 0.15 sinon (defaut conservateur si None).
    """
    if n_green is None or n_green < SMALL_DATASET_GREEN_THRESHOLD:
        return MAX_MCC_STD_SMALL
    return MAX_MCC_STD_LARGE


# =============================================================================
# UTILITAIRES
# =============================================================================

# Regex detectant une ligne deja formatee par loguru (le sous-process ecrit
# deja dans le meme fichier de log via son propre handler, donc inutile de la
# re-injecter : on aurait des doublons).
# Format attendu : "2026-04-15 00:26:58 | WARNING  | module:function:line | ..."
_LOGURU_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| [A-Z]+\s+\|"
)

# Regex pour stripper les codes ANSI (coloration loguru console) AVANT de
# tester _LOGURU_LINE_RE. Sans ce stripping, les lignes colorees du
# subprocess (\033[32m...\033[0m) ne sont pas reconnues comme "deja
# formatees par loguru" et sont re-loggees a tort comme ERROR par
# _forward_stream, produisant des centaines de faux positifs (incident du
# 17 avril 2026 : 869 fausses erreurs sur sql_ingester).
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Regex detectant les warnings Python standards (`warnings.warn` via stderr).
# Ces lignes ne passent pas par loguru et ne sont donc PAS dans le fichier de
# log sans capture explicite. Exemples :
#   "C:\path\file.py:42: DeprecationWarning: torch_dtype is deprecated, ..."
#   "UserWarning: ..."
_PYTHON_WARNING_RE = re.compile(
    r"\b(DeprecationWarning|FutureWarning|UserWarning|PendingDeprecationWarning"
    r"|RuntimeWarning|SyntaxWarning|ImportWarning|ResourceWarning|BytesWarning"
    r"|UnicodeWarning)\b"
)

# Regex detectant le debut d'une traceback Python sur stderr. On elève le
# niveau de log a ERROR pour que ces exceptions soient visibles dans Grafana.
_TRACEBACK_START_RE = re.compile(r"^Traceback \(most recent call last\):")


def _forward_stream(
    stream: IO[str],
    *,
    source_label: str,
    is_stderr: bool,
    console_stream: IO[str],
) -> None:
    """Relit un stream ligne par ligne et route vers console + loguru.

    Tourne dans un thread dedie : garde le flux parent-console en temps reel
    (UX preservee, pas de buffering qui cache la progression des scripts
    longs) tout en persistant les lignes critiques (warnings Python,
    tracebacks) dans le fichier de log via loguru parent.

    Args:
        stream: Flux texte de Popen (stdout ou stderr du sous-process).
        source_label: Identifiant du sous-process (ex. ``"module:training"``
            ou ``"script:retrain_pipeline"``) ajoute aux extras loguru.
        is_stderr: True si le flux est stderr (impacte la detection du niveau
            et la redirection console).
        console_stream: Flux console parent ou republier la ligne telle
            quelle (``sys.stdout`` ou ``sys.stderr``).
    """
    traceback_active = False
    for raw_line in stream:
        line = raw_line.rstrip("\n")
        # Republier tel quel sur la console parent pour preserver l'UX temps
        # reel (l'utilisateur continue de voir la progression des scripts).
        console_stream.write(raw_line)
        console_stream.flush()

        if not line.strip():
            traceback_active = False
            continue

        # Les lignes deja formatees par loguru sont ecrites dans le fichier
        # de log commun par le sous-process lui-meme : ne pas les republier
        # depuis le parent (doublons garantis). On strip les codes ANSI avant
        # le test car loguru en mode console colore ses lignes avec des
        # escape sequences (\033[32m...\033[0m) qui empechent le regex
        # loguru de matcher sur le texte brut.
        clean_line = _ANSI_ESCAPE_RE.sub("", line)
        if _LOGURU_LINE_RE.match(clean_line):
            traceback_active = False
            continue

        level = "DEBUG"
        if is_stderr:
            if _TRACEBACK_START_RE.match(line) or traceback_active:
                level = "ERROR"
                traceback_active = True
            elif _PYTHON_WARNING_RE.search(line):
                level = "WARNING"
            else:
                # stderr generique (messages de torch, scrapy, spark, etc.) :
                # en INFO pour avoir une trace sans polluer les WARNINGs.
                level = "INFO"
        else:
            # stdout non-loguru : stats de progression, prints explicites, ...
            level = "INFO"

        logger.opt(depth=1).bind(subprocess=source_label).log(level, line)


def _stream_subprocess(cmd: list[str], source_label: str) -> int:
    """Lance un sous-process et streame stdout/stderr vers console + loguru.

    Remplace ``subprocess.run(..., capture_output=False)`` qui laissait
    echapper les warnings Python hors de notre systeme de logs. Avec cette
    version, tout ce qui sort sur stderr (y compris ``DeprecationWarning``
    emis par ``warnings.warn``) est traduit en entree loguru et persiste
    dans ``logs/greentech_<date>.log``.

    Args:
        cmd: Commande a executer (liste d'arguments, pas de shell).
        source_label: Identifiant court du sous-process affiche dans les
            logs parent (ex. ``"module:training"``, ``"script:classify"``).

    Returns:
        Code de retour du sous-process (0 = succes).
    """
    process = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line-buffered pour un streaming temps reel
    )

    # Threads daemon : si le parent est tue, ils s'arretent sans bloquer.
    # Lire stdout et stderr en parallele evite le deadlock classique sur PIPE
    # plein quand l'un des flux est lent.
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(
        target=_forward_stream,
        kwargs={
            "stream": process.stdout,
            "source_label": source_label,
            "is_stderr": False,
            "console_stream": sys.stdout,
        },
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_stream,
        kwargs={
            "stream": process.stderr,
            "source_label": source_label,
            "is_stderr": True,
            "console_stream": sys.stderr,
        },
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    return_code = process.wait()
    # Attendre la fin des threads de lecture pour ne pas perdre les dernieres
    # lignes bufferisees (Popen.wait() rend la main avant que les PIPE soient
    # completement drains).
    stdout_thread.join()
    stderr_thread.join()
    return return_code


def _run_module(module: str, *args: str) -> bool:
    """Execute un module Python via ``uv run``.

    La sortie stdout/stderr du sous-process est streamee en temps reel vers
    la console parent ET vers le fichier loguru (warnings Python, tracebacks
    inclus), ce qui permet de detecter les DeprecationWarning/FutureWarning
    dans les analyses de logs a posteriori.

    Args:
        module: Chemin du module Python a executer (ex. ``greentech.ai.models.training``).
        *args: Arguments supplementaires passes au module.

    Returns:
        True si l'execution a reussi (return code == 0).
    """
    cmd = ["uv", "run", "python", "-m", module, *args]
    logger.info(f"Execution : {' '.join(cmd)}")
    return_code = _stream_subprocess(cmd, source_label=f"module:{module}")
    if return_code != 0:
        logger.error(f"Echec de {module} (code {return_code})")
        return False
    return True


def _run_script(script: str, *args: str) -> bool:
    """Execute un script Python via ``uv run``.

    Meme comportement de capture que :func:`_run_module` : les warnings et
    erreurs du sous-process sont repris par loguru parent et persistes dans
    le fichier de log (ce qui permet l'analyse post-mortem de tous les
    DeprecationWarning / tracebacks emis sur stderr).

    Args:
        script: Chemin relatif du script depuis la racine du projet.
        *args: Arguments supplementaires passes au script.

    Returns:
        True si l'execution a reussi (return code == 0).
    """
    cmd = ["uv", "run", "python", str(BASE_DIR / script), *args]
    logger.info(f"Execution : {' '.join(cmd)}")
    script_name = Path(script).stem
    return_code = _stream_subprocess(cmd, source_label=f"script:{script_name}")
    if return_code != 0:
        logger.error(f"Echec de {script} (code {return_code})")
        return False
    return True


def _generate_version_tag() -> str:
    """Genere un tag de version horodate."""
    return datetime.now(UTC).strftime("v%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict | None:
    """Charge un fichier JSON s'il existe."""
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_best_metrics(metrics: dict, version_tag: str) -> None:
    """Sauvegarde les metriques de la meilleure version connue."""
    data = {
        "version": version_tag,
        "date": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }
    BEST_METRICS_FILE.write_text(json.dumps(data, indent=2))
    logger.info(
        f"Meilleur modele enregistre : {version_tag} "
        f"(MCC={metrics.get('mcc', 0.0):.4f}, F1={metrics['f1']:.4f})"
    )


def _compute_detailed_metrics(
    y_true: list[int],
    y_pred: list[int],
    latencies_ms: list[float] | None = None,
) -> dict[str, float | int]:
    """Calcule un jeu complet de metriques pour comparer les versions d'un modele.

    Produit :
    - MCC (Matthews Correlation Coefficient) : metrique principale, robuste au
      desequilibre de classes, recommandee par la litterature sur ce type de probleme.
    - F1, accuracy, balanced_accuracy, precision, recall, specificite.
    - Matrice de confusion (TP/TN/FP/FN).
    - Distribution reelle vs predite par classe.

    Args:
        y_true: Labels reels (0 = Non Green IT, 1 = Green IT).
        y_pred: Labels predits par le modele.
        latencies_ms: Latences d'inference en millisecondes (optionnel).

    Returns:
        Dictionnaire contenant toutes les metriques ci-dessus, pret a etre
        serialise en JSON ou compare entre versions.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    nb_reels_green_it = int(sum(1 for lbl in y_true if lbl == 1))
    nb_reels_non_green_it = int(sum(1 for lbl in y_true if lbl == 0))
    nb_pred_green_it = int(sum(1 for p in y_pred if p == 1))
    nb_pred_non_green_it = int(sum(1 for p in y_pred if p == 0))

    specificite = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics: dict[str, float | int] = {
        # Metriques principales
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "specificite": specificite,
        # Matrice de confusion
        "vrais_positifs": int(tp),
        "vrais_negatifs": int(tn),
        "faux_positifs": int(fp),
        "faux_negatifs": int(fn),
        # Distribution reelle vs predite
        "nb_reels_green_it": nb_reels_green_it,
        "nb_reels_non_green_it": nb_reels_non_green_it,
        "nb_predictions_green_it": nb_pred_green_it,
        "nb_predictions_non_green_it": nb_pred_non_green_it,
        "total_echantillons": len(y_true),
    }

    if latencies_ms:
        metrics["latence_moyenne_ms"] = float(np.mean(latencies_ms))

    return metrics


def _log_detailed_metrics(metrics: dict, titre: str) -> None:
    """Affiche un rapport detaille des metriques dans les logs.

    Tolere les dictionnaires partiels (anciens formats) : les champs absents
    sont affiches comme "N/A" plutot que de faire echouer le rapport.

    Args:
        metrics: Dictionnaire produit par `_compute_detailed_metrics`,
            ou dictionnaire legacy avec un sous-ensemble des cles.
        titre: Titre du bloc affiche (ex. "Modele de base", "Nouveau modele").
    """

    def _fmt(key: str, fmt: str = ".4f") -> str:
        val = metrics.get(key)
        if val is None:
            return "N/A"
        return format(val, fmt)

    def _pct(key: str) -> str:
        val = metrics.get(key)
        if val is None:
            return ""
        return f"  ({val * 100:.2f}%)"

    def _int(key: str) -> str:
        val = metrics.get(key)
        return str(val) if val is not None else "N/A"

    logger.info("")
    logger.info("-" * 72)
    logger.info(f"  {titre}")
    logger.info("-" * 72)
    logger.info(f"  MCC (critere principal)  : {_fmt('mcc')}")
    logger.info(f"  F1-score                 : {_fmt('f1')}")
    logger.info(f"  Accuracy (reussite)      : {_fmt('accuracy')}{_pct('accuracy')}")
    logger.info(
        f"  Balanced accuracy        : {_fmt('balanced_accuracy')}{_pct('balanced_accuracy')}"
    )
    logger.info(f"  Precision (fiabilite)    : {_fmt('precision')}{_pct('precision')}")
    logger.info(f"  Recall (rappel)          : {_fmt('recall')}{_pct('recall')}")
    logger.info(f"  Specificite              : {_fmt('specificite')}{_pct('specificite')}")

    has_confusion = all(
        k in metrics for k in ("vrais_positifs", "vrais_negatifs", "faux_positifs", "faux_negatifs")
    )
    if has_confusion:
        total = metrics.get("total_echantillons", 0)
        logger.info("")
        logger.info(f"  Matrice de confusion (sur {total} articles) :")
        logger.info(
            f"    Vrais positifs  (Green IT correctement detectes)     : {_int('vrais_positifs')}"
        )
        logger.info(
            f"    Vrais negatifs  (Non Green IT correctement detectes) : {_int('vrais_negatifs')}"
        )
        logger.info(
            f"    Faux positifs   (Non Green IT classes Green IT)      : {_int('faux_positifs')}"
        )
        logger.info(
            f"    Faux negatifs   (Green IT classes Non Green IT)      : {_int('faux_negatifs')}"
        )

    has_distribution = all(
        k in metrics
        for k in (
            "nb_reels_green_it",
            "nb_reels_non_green_it",
            "nb_predictions_green_it",
            "nb_predictions_non_green_it",
        )
    )
    if has_distribution:
        logger.info("")
        logger.info("  Distribution :")
        logger.info(
            f"    Reels    : Green IT = {metrics['nb_reels_green_it']:>5d}  |  "
            f"Non Green IT = {metrics['nb_reels_non_green_it']:>5d}"
        )
        logger.info(
            f"    Predits  : Green IT = {metrics['nb_predictions_green_it']:>5d}  |  "
            f"Non Green IT = {metrics['nb_predictions_non_green_it']:>5d}"
        )

    if "latence_moyenne_ms" in metrics:
        logger.info("")
        logger.info(f"  Latence moyenne       : {_fmt('latence_moyenne_ms', '.2f')} ms")
    logger.info("-" * 72)


# =============================================================================
# COLLECTE
# =============================================================================


def step_collect() -> bool:
    """Re-collecte des donnees depuis les sources configurees.

    Interroge sequentiellement les trois types de sources :

    1. **API REST/JSON** : The Guardian (source principale, 5000 req/jour)
       puis Dev.to (complement, aucune cle requise). La NewsData.io est
       desactivee depuis avril 2026 car son free tier tronque le contenu
       au message "ONLY AVAILABLE IN PAID PLANS".
    2. **Scraping** : TechCrunch Climate (Scrapy + Playwright sur le DOM rendu).
    3. **Nettoyage + Ingestion SQL** sur l'ensemble des nouveaux articles.

    Les erreurs sur les collectes REST/JSON n'interrompent pas le pipeline :
    on continue avec les sources restantes, puis on fait le bilan final.
    Seul un echec du nettoyage Spark ou de l'ingestion SQL est considere
    comme bloquant.
    """
    logger.info("=" * 70)
    logger.info("  COLLECTE DES DONNEES")
    logger.info("=" * 70)

    ok = True

    logger.info("\n--- Collecte API (The Guardian) ---")
    if not _run_module("greentech.data.collectors.guardian_collector"):
        logger.warning("Collecte Guardian echouee (cle API manquante ou quota), on continue")

    logger.info("\n--- Collecte API (Dev.to) ---")
    if not _run_module("greentech.data.collectors.devto_collector"):
        logger.warning("Collecte Dev.to echouee (probleme reseau ?), on continue")

    logger.info("\n--- Scraping web (TechCrunch) ---")
    if not _run_module("greentech.data.collectors.scraper"):
        logger.warning("Scraping echoue, on continue")

    logger.info("\n--- Nettoyage Spark ---")
    if not _run_module("greentech.data.processors.spark_cleaner"):
        logger.error("Nettoyage Spark echoue")
        ok = False

    if ok:
        logger.info("\n--- Ingestion PostgreSQL ---")
        if not _run_module("greentech.data.storage.sql_ingester"):
            logger.error("Ingestion SQL echouee")
            ok = False

    return ok


def step_ingest_file(file_path: str) -> bool:
    """Ingere un fichier local puis nettoie et ingere dans PostgreSQL."""
    logger.info("=" * 70)
    logger.info(f"  INGESTION FICHIER : {file_path}")
    logger.info("=" * 70)

    path = Path(file_path)
    if not path.exists():
        logger.error(f"Fichier introuvable : {file_path}")
        return False

    if not _run_module("greentech.data.collectors.file_ingester", str(path)):
        return False

    if not _run_module("greentech.data.processors.spark_cleaner"):
        return False
    return _run_module("greentech.data.storage.sql_ingester")


# =============================================================================
# NETTOYAGE (suppression des articles inexploitables)
# =============================================================================


async def step_clean() -> bool:
    """Supprime les articles inexploitables de la base de donnees.

    Cible trois categories d'articles qui polluent le dataset sans apporter
    aucun signal a l'entrainement du classifieur ni au resume :

    1. Articles **sans contenu** (``contenu IS NULL``) : ingestion echouee
       ou partielle, pas de texte a analyser.
    2. Articles **au contenu trop court** (< 50 caracteres) : abstracts
       arXiv corrompus/tronques dans le dump Kaggle, ou articles web dont
       l'extraction a echoue. Le LLM ne peut rien en tirer et ils font
       gonfler le compteur d'echecs du pipeline de resume.
    3. Articles avec le **placeholder NewsData.io** (``"ONLY AVAILABLE IN
       PAID PLANS"``) : vestiges de l'ancien pipeline NewsData dont le
       free tier tronquait le contenu. Filet de securite au cas ou de
       nouveaux articles arrivent depuis MinIO legacy.

    L'etape est idempotente (DELETE WHERE) et rapide (quelques secondes).
    Elle s'insere entre ``collect`` et ``summarize-classif`` dans le
    pipeline par defaut, mais peut aussi etre lancee manuellement via
    ``uv run python scripts/retrain_pipeline.py clean``.
    """
    from sqlalchemy import delete, func, or_

    from greentech.ai.services.classification_summarizer import (
        CLASSIFICATION_MIN_INPUT_CHARS,
    )
    from greentech.data.storage.database import async_session_factory
    from greentech.data.storage.models import Article

    logger.info("=" * 70)
    logger.info("  NETTOYAGE DES ARTICLES INEXPLOITABLES")
    logger.info("=" * 70)

    async with async_session_factory() as session:
        # Comptage pre-suppression pour le log detaille
        from sqlalchemy import select

        preview = await session.execute(
            select(
                func.count().filter(Article.contenu.is_(None)).label("sans_contenu"),
                func.count()
                .filter(
                    Article.contenu.isnot(None),
                    func.length(Article.contenu) < CLASSIFICATION_MIN_INPUT_CHARS,
                )
                .label("trop_courts"),
                func.count()
                .filter(Article.contenu.contains("ONLY AVAILABLE IN PAID PLANS"))
                .label("placeholder_newsdata"),
            )
        )
        row = preview.one()
        sans_contenu = row.sans_contenu
        trop_courts = row.trop_courts
        placeholder = row.placeholder_newsdata

        total_a_supprimer = sans_contenu + trop_courts + placeholder
        if total_a_supprimer == 0:
            logger.info("Aucun article inexploitable detecte, base propre")
            return True

        logger.info(
            f"Articles a supprimer : {total_a_supprimer} "
            f"(sans contenu={sans_contenu}, trop courts={trop_courts}, "
            f"placeholder NewsData={placeholder})"
        )

        # Suppression en une requete atomique
        stmt = delete(Article).where(
            or_(
                Article.contenu.is_(None),
                func.length(Article.contenu) < CLASSIFICATION_MIN_INPUT_CHARS,
                Article.contenu.contains("ONLY AVAILABLE IN PAID PLANS"),
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        nb_supprimes = result.rowcount

    logger.info(f"Nettoyage termine : {nb_supprimes} articles supprimes")
    return True


# =============================================================================
# ANNOTATION (pipeline de classification hybride en deux etages)
# =============================================================================


async def step_annotate() -> bool:
    """Etage 1 : pre-filtre mots-cles permissif.

    Applique le scoring multi-criteres sur tous les articles non encore
    classifies pour decider : NON_GREEN (rejet direct) ou CANDIDATE (a
    envoyer au LLM judge). Ecrit le resultat dans la colonne
    `articles.modele_classification = 'keyword_filter'`.

    Appel direct de ``scripts.auto_annotate_dataset.run_prefilter`` dans le
    meme process Python que le pipeline. Cette architecture remplace le
    lancement subprocess historique pour eviter la saturation du PIPE
    stdout/stderr sous Windows (incident du 16 avril 2026 : freeze
    systematique apres ~4 articles a cause du buffer PIPE 4-8 KB
    surcharge par les logs SQLAlchemy verbose).
    """
    logger.info("=" * 70)
    logger.info("  PRE-FILTRE MOTS-CLES (etage 1)")
    logger.info("=" * 70)
    from auto_annotate_dataset import run_prefilter

    try:
        stats = await run_prefilter()
    except Exception as exc:
        logger.exception(f"Echec pre-filtre mots-cles : {exc}")
        return False
    logger.info(f"Pre-filtre termine : {stats}")
    return True


async def step_classify() -> bool:
    """Etage 2 : verification LLM des candidats.

    Interroge `Qwen/Qwen3-4B-Instruct-2507` via HF Serverless (avec fallback
    local Qwen2.5-3B sur GPU AMD ROCm si le quota est epuise) pour
    classifier definitivement les articles marques CANDIDATE par l'etage 1.
    Met a jour `est_green_it` et passe `modele_classification` a
    `'keyword_filter+qwen_llm_judge'`.

    Le LLM judge lit le `contenu` brut (pas le resume) pour maximiser la
    qualite du ground truth : il a 32k tokens de contexte dispo cote
    cloud, donc autant lui donner l'article complet.

    Appel direct Python dans le process courant (voir ``step_annotate``
    pour la motivation anti-PIPE).
    """
    logger.info("=" * 70)
    logger.info("  LLM JUDGE - VERIFICATION DES CANDIDATS (etage 2)")
    logger.info("=" * 70)
    from classify_candidates import classify_all_candidates

    try:
        stats = await classify_all_candidates()
    except Exception as exc:
        logger.exception(f"Echec LLM judge : {exc}")
        return False
    logger.info(f"LLM judge termine : {stats}")
    return True


async def step_summarize_classification() -> bool:
    """Genere le resume de classification pour tous les articles.

    Applique le prompt centralise (``CLASSIFICATION_SYSTEM_PROMPT`` +
    ``CLASSIFICATION_USER_PROMPT_TEMPLATE``) a l'integralite du dataset.
    Ecrit le resultat dans ``articles.resume``. Cette colonne devient
    ensuite la feature d'entrainement principale : c'est elle qui est
    concatenee au titre et envoyee au classifieur Qwen3-4B + LoRA.

    Idempotent par defaut : saute les articles ayant deja un resume.
    L'appel se fait en direct (import + await) plutot qu'en subprocess
    pour eviter le freeze systemique observe en subprocess sous Windows
    (saturation PIPE par les logs SQLAlchemy verbose, incident 16 avril
    2026). Le Qwen local est charge une seule fois et reste disponible
    pour l'etape ``summarize-green`` qui suit, evitant un cold start de
    ~5 minutes entre les deux etapes.
    """
    logger.info("=" * 70)
    logger.info("  RESUMES DE CLASSIFICATION (tous articles)")
    logger.info("=" * 70)
    from greentech.ai.services.summarizer import (
        summarize_all_articles_for_classification,
    )

    try:
        stats = await summarize_all_articles_for_classification()
    except Exception as exc:
        logger.exception(f"Echec resumes classification : {exc}")
        return False

    total = stats.get("total", 0)
    succes = stats.get("succes", 0)
    echecs = stats.get("echecs", 0)
    logger.info(
        f"Resumes de classification termines : {succes}/{total} succes, {echecs} echecs"
    )
    # Seuil d'acceptation : on tolere jusqu'a 5% d'echecs (articles au
    # contenu degenere, API ponctuellement indisponible). Au-dela on
    # considere que quelque chose cloche structurellement et on stoppe
    # le pipeline pour investigation avant de propager des resumes
    # incomplets aux etapes en aval.
    if total > 0 and echecs / total > 0.05:
        logger.error(
            f"Taux d'echec trop eleve ({echecs / total:.1%}), arret pipeline"
        )
        return False
    return True


async def step_summarize_green() -> bool:
    """Genere le resume ecologique pour les articles Green IT confirmes.

    Applique le prompt specialise Green IT aux articles dont
    ``est_green_it = True`` et qui n'ont pas encore de
    ``resume_ecologique``. Ce resume complete le ``resume`` general et
    alimente la section "aspects ecologiques" de la page detail UI.

    Appel direct Python dans le process courant.
    """
    logger.info("=" * 70)
    logger.info("  RESUMES ECOLOGIQUES (Green IT uniquement)")
    logger.info("=" * 70)
    from greentech.ai.services.summarizer import summarize_green_only

    try:
        stats = await summarize_green_only()
    except Exception as exc:
        logger.exception(f"Echec resumes ecologiques : {exc}")
        return False

    logger.info(f"Resumes ecologiques termines : {stats}")
    return True


async def step_summarize_all() -> bool:
    """Enchaine les deux passes de resume : classification puis ecologique.

    Utile quand on lance l'etape alias ``summarize`` (ancienne terminologie)
    ou quand on veut regenerer completement les deux colonnes ``resume`` et
    ``resume_ecologique`` en un seul appel du pipeline. Les nouveaux scripts
    preferent invoquer les deux etapes separement (``summarize-classif``
    puis ``summarize-green``) pour un controle plus fin.

    Returns:
        True si les deux passes ont reussi.
    """
    if not await step_summarize_classification():
        return False
    return await step_summarize_green()


async def step_export_golden() -> bool:
    """Regenere `data/golden_dataset.csv` depuis l'etat final de la DB.

    A executer apres la classification hybride pour produire un golden
    dataset qui reflete les decisions du LLM (et non plus du pre-filtre
    seul). Ce CSV est la source de verite pour l'entrainement Qwen3-4B.

    Appel direct Python dans le process courant.
    """
    logger.info("=" * 70)
    logger.info("  EXPORT GOLDEN DATASET (DB -> CSV)")
    logger.info("=" * 70)
    from export_golden_dataset import export_from_db

    try:
        stats = await export_from_db()
    except Exception as exc:
        logger.exception(f"Echec export golden dataset : {exc}")
        return False

    logger.info(f"Export termine : {stats}")
    return True


# =============================================================================
# ARCHIVAGE
# =============================================================================


def step_archive_current_production() -> str | None:
    """Archive le modele de production actuel dans models/versions/."""
    if not PRODUCTION_DIR.exists() or not (PRODUCTION_DIR / "adapter_config.json").exists():
        logger.info("Pas de modele de production a archiver")
        return None

    version_tag = _generate_version_tag()
    archive_dir = VERSIONS_DIR / version_tag
    archive_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Archivage du modele actuel vers {archive_dir}")
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, archive_dir / f.name)

    best = _load_json(BEST_METRICS_FILE)
    metadata = {
        "version": version_tag,
        "date_archive": datetime.now(UTC).isoformat(),
        "fichiers": [f.name for f in archive_dir.iterdir() if f.is_file()],
        "metrics": best["metrics"] if best else None,
    }
    (archive_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(f"Archive creee : {version_tag}")
    return version_tag


# =============================================================================
# ENTRAINEMENT
# =============================================================================


def step_train() -> bool:
    """Re-entraine Qwen3-4B + LoRA sur le dataset courant (split 80/20)."""
    logger.info("=" * 70)
    logger.info("  RE-ENTRAINEMENT QWEN3-4B + LoRA (split 80/20)")
    logger.info("=" * 70)
    return _run_module("greentech.ai.models.training", TRAIN_MODEL_TYPE)


# Fichier ou stocker le rapport K-fold complet (folds + moyennes + metriques globales)
CV_REPORT_FILE = MODELS_DIR / "cv_report.json"


async def step_train_cv(
    n_splits: int = 5,
    random_state: int = 42,
    train_final: bool = True,
) -> dict | None:
    """Entraine Qwen3-4B + LoRA via K-fold stratifie (K=5 par defaut).

    Chaque fold est entraine avec oversampling de la classe minoritaire, puis
    evalue sur le test du fold (sans oversampling). Les metriques agregees
    (moyenne + ecart-type sur K folds) servent de reference pour la promotion.

    Un modele final est re-entraine sur l'integralite des donnees et sauvegarde
    dans `models/challenger-qwen3/`, pret pour la promotion en production si
    les criteres sont satisfaits.

    Args:
        n_splits: Nombre de folds (defaut 5).
        random_state: Seed pour la reproductibilite du split.
        train_final: Si True, entraine un modele final sur tout le dataset apres le K-fold.

    Returns:
        Rapport complet du K-fold, ou None en cas d'erreur.
    """
    from greentech.ai.models.training import train_challenger_with_cv

    logger.info("=" * 70)
    logger.info(f"  RE-ENTRAINEMENT QWEN3-4B + LoRA (K-fold K={n_splits})")
    logger.info("=" * 70)

    try:
        rapport = await train_challenger_with_cv(
            model_type=TRAIN_MODEL_TYPE,
            n_splits=n_splits,
            random_state=random_state,
            train_final=train_final,
        )
    except Exception as exc:
        logger.exception(f"K-fold echoue : {exc}")
        return None

    rapport["date"] = datetime.now(UTC).isoformat()
    CV_REPORT_FILE.write_text(json.dumps(rapport, indent=2, default=str))
    logger.info(f"Rapport K-fold sauvegarde : {CV_REPORT_FILE}")

    _log_cv_report(rapport)
    return rapport


def _is_cv_report_fresh() -> bool:
    """Verifie que le rapport CV est plus recent que le modele entraine.

    Un rapport obsolete (anterieur a un entrainement 80/20 ulterieur) ne doit
    pas etre utilise pour la decision de promotion.
    """
    if not CV_REPORT_FILE.exists():
        return False
    adapter_file = TRAIN_DIR / "adapter_config.json"
    if not adapter_file.exists():
        return False
    return CV_REPORT_FILE.stat().st_mtime >= adapter_file.stat().st_mtime - 60


def _log_cv_variability(aggregated: dict) -> None:
    """Affiche la stabilite des metriques entre folds."""
    logger.info("")
    logger.info("  Stabilite entre folds (+/- ecart-type) :")
    for key in ("mcc", "f1", "recall"):
        if key in aggregated:
            stats = aggregated[key]
            logger.info(
                f"    {key:<8}: moy={stats['mean']:.4f}  std={stats['std']:.4f}  "
                f"min={stats['min']:.4f}  max={stats['max']:.4f}"
            )


def _log_cv_report(rapport: dict) -> None:
    """Affiche un resume lisible du rapport K-fold dans les logs."""
    aggregated = rapport["aggregated"]
    global_m = rapport["global"]
    folds = rapport["folds"]

    logger.info("")
    logger.info("=" * 72)
    logger.info(f"  RAPPORT K-FOLD ({rapport['n_splits']} folds)")
    logger.info("=" * 72)
    logger.info("  Metriques par fold :")
    logger.info(f"    {'Fold':<6}{'n_test':<10}{'n_green':<10}{'MCC':<12}{'F1':<12}{'Recall':<12}")
    for f in folds:
        logger.info(
            f"    {f['fold']:<6}{f['n_test']:<10}{f['n_green_test']:<10}"
            f"{f['mcc']:<12.4f}{f['f1']:<12.4f}{f['recall']:<12.4f}"
        )

    logger.info("")
    logger.info("  Moyennes sur les folds (+/- ecart-type) :")
    for key in ("mcc", "f1", "balanced_accuracy", "precision", "recall", "specificite"):
        stats = aggregated[key]
        logger.info(
            f"    {key:<20}: {stats['mean']:.4f} "
            f"(+/- {stats['std']:.4f}, "
            f"min={stats['min']:.4f}, max={stats['max']:.4f})"
        )

    logger.info("")
    logger.info("  Metriques globales (concatenation des predictions des folds) :")
    logger.info(f"    MCC global         : {global_m['mcc']:.4f}")
    logger.info(f"    F1 global          : {global_m['f1']:.4f}")
    logger.info(f"    Balanced accuracy  : {global_m['balanced_accuracy']:.4f}")
    logger.info(f"    Recall Green IT    : {global_m['recall']:.4f}")
    logger.info(f"    Precision          : {global_m['precision']:.4f}")
    logger.info(f"    Specificite        : {global_m['specificite']:.4f}")
    logger.info("")
    logger.info("  Matrice de confusion globale (K folds agreges) :")
    logger.info(f"    TP = {global_m['vrais_positifs']}  |  FN = {global_m['faux_negatifs']}")
    logger.info(f"    FP = {global_m['faux_positifs']}  |  TN = {global_m['vrais_negatifs']}")

    if rapport.get("final_model_trained"):
        logger.info("")
        logger.info("  Modele final entraine sur l'integralite du dataset : OK")
    logger.info("=" * 72)


# =============================================================================
# BASELINE (modele de base sans fine-tuning)
# =============================================================================


async def step_baseline(force: bool = False) -> dict[str, float | int]:
    """Calcule les metriques du modele de base SANS fine-tuning (reference permanente).

    Delegue au module `greentech.ai.models.baseline` qui factorise la logique
    de chargement + inference + calcul de metriques pour tout modele
    Hugging Face compatible `AutoModelForSequenceClassification`. Par defaut,
    evalue `Qwen/Qwen3-4B` defini dans ``settings.huggingface_model_baseline``.

    La baseline est recalculee si :
    - Aucune baseline n'a ete sauvegardee, OU
    - Le format de la baseline existante est obsolete (ancien split 20% ou
      metriques incompletes), OU
    - ``force=True`` est explicitement passe, OU
    - Le modele de la baseline existante differe du modele configure (par
      exemple apres une migration Llama -> Qwen3-4B).

    Args:
        force: Si True, recalcule meme si une baseline existe deja.

    Returns:
        Dictionnaire complet des metriques (MCC, F1, accuracy, precision,
        recall, specificite, matrice de confusion, distribution des
        predictions, latence moyenne).
    """
    from greentech.ai.models.baseline import evaluate_baseline
    from greentech.config import get_settings

    settings = get_settings()
    expected_model = settings.huggingface_model_baseline

    existing = _load_json(BASELINE_METRICS_FILE)
    required_keys = {"mcc", "specificite", "balanced_accuracy"}
    correct_scope = existing.get("evaluation_scope") == "full_dataset" if existing else False
    correct_model = existing.get("model") == expected_model if existing else False
    legacy_format = bool(existing) and (
        not required_keys.issubset(existing.get("metrics", {}).keys())
        or not correct_scope
        or not correct_model
    )

    if existing and not force and not legacy_format:
        logger.info(
            f"Baseline deja calculee (model={existing['model']}, "
            f"MCC={existing['metrics'].get('mcc', 0.0):.4f}, "
            f"F1={existing['metrics']['f1']:.4f}), reutilisation"
        )
        _log_detailed_metrics(existing["metrics"], "Baseline (rechargee)")
        return existing["metrics"]
    if legacy_format:
        logger.info(
            "Baseline en ancien format ou modele different detecte "
            f"(attendu : {expected_model}), recalcul complet"
        )

    logger.info("=" * 70)
    logger.info(f"  CALCUL BASELINE : {expected_model} SANS fine-tuning")
    logger.info("  Portee            : TOUT le dataset (aucun data leakage possible)")
    logger.info("=" * 70)

    result = evaluate_baseline(model_name=expected_model)

    _log_detailed_metrics(
        dict(result.metrics),
        f"Baseline : {result.model_name} (sans fine-tuning, dataset complet)",
    )

    # Tracking triple : JSON local + MLflow (run tagge baseline) + Pushgateway
    # Prometheus (metriques greentech_baseline_* visibles dans Grafana).
    from greentech.ai.mlops.baseline_tracking import track_baseline

    track_baseline(result, BASELINE_METRICS_FILE)
    logger.info(
        f"Baseline sauvegardee : MCC={result.metrics['mcc']:.4f}, "
        f"F1={result.metrics['f1']:.4f}, Recall={result.metrics['recall']:.4f} "
        f"(n={result.n_articles})"
    )

    return dict(result.metrics)


# =============================================================================
# BENCHMARK (nouveau vs meilleur historique vs baseline)
# =============================================================================


async def _evaluate_new_model(
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, float | int]:
    """Evalue le modele fraichement entraine sur le test set.

    Charge l'adaptateur LoRA depuis ``TRAIN_DIR`` (par defaut
    ``models/challenger-qwen3/``) et execute l'inference. Si le dossier
    contient un modele d'une autre famille (par exemple un ancien challenger
    Llama), la sous-classe adequate est instanciee via l'`adapter_config.json`
    pour reconstruire exactement le setup d'entrainement.
    """
    import json

    import torch
    from sklearn.metrics import classification_report

    from greentech.ai.models.classifier import TrainingConfig
    from greentech.ai.models.training import (
        ChallengerClassifier,
        ChallengerQwen3Classifier,
    )
    from greentech.config import get_settings

    adapter_file = TRAIN_DIR / "adapter_config.json"
    if adapter_file.exists():
        adapter_meta = json.loads(adapter_file.read_text())
        base_model_name = adapter_meta.get("base_model_name_or_path") or get_settings().huggingface_model_trainer_base
    else:
        base_model_name = get_settings().huggingface_model_trainer_base

    # Match strict sur `qwen3-4b` / `qwen3_4b` pour ne pas capturer par erreur
    # les anciens adaptateurs `qwen3.5-4b` si jamais il en reste en cache.
    name_lower = base_model_name.lower()
    is_qwen3 = "qwen3-4b" in name_lower or "qwen3_4b" in name_lower
    config = TrainingConfig(nom_modele=base_model_name, output_dir=TRAIN_DIR)
    classifier = (
        ChallengerQwen3Classifier(config=config)
        if is_qwen3
        else ChallengerClassifier(config=config)
    )
    classifier.load(TRAIN_DIR)

    preds: list[int] = []
    latencies: list[float] = []
    for text in test_texts:
        pred = await classifier.predict(text)
        preds.append(pred.label.value)
        latencies.append(pred.temps_ms)

    metrics = _compute_detailed_metrics(test_labels, preds, latencies)
    _log_detailed_metrics(metrics, "Nouveau modele (post-entrainement)")

    logger.info("")
    logger.info("  Rapport de classification detaille (sklearn) :")
    logger.info(
        "\n"
        + classification_report(
            test_labels,
            preds,
            target_names=["Non Green IT", "Green IT"],
            zero_division=0,
        )
    )

    del classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _evaluate_promotion_criteria(
    new_metrics: dict,
    best_metrics: dict | None,
) -> dict:
    """Evalue si le nouveau modele doit etre promu selon 3 criteres combines.

    Le critere composite est concu pour un dataset fortement desequilibre :
    - Le MCC est plus stable que le F1 face au desequilibre (litterature
      Chicco & Jurman 2020).
    - Le garde-fou recall evite qu'un modele "tricheur" (qui predit tout en
      Non Green IT et obtient un bon score global) soit retenu.
    - La tolerance F1 accepte un leger recul si le MCC progresse, mais refuse
      une regression majeure.

    Args:
        new_metrics: Metriques du nouveau modele (avec cles 'mcc', 'f1', 'recall').
        best_metrics: Metriques du meilleur modele historique, ou None si
            aucun modele n'a encore ete promu.

    Returns:
        Dictionnaire avec :
        - 'retenu' (bool) : True si le nouveau modele doit etre promu
        - 'criteres' (list) : detail de chaque critere evalue
        - 'raison' (str) : explication du verdict
    """
    criteres: list[dict] = []

    # Critere 1 : MCC (robuste au desequilibre)
    mcc_nouveau = float(new_metrics.get("mcc", 0.0))
    if best_metrics is not None and "mcc" in best_metrics:
        mcc_ancien = float(best_metrics["mcc"])
        mcc_pass = mcc_nouveau >= (mcc_ancien - MCC_EPSILON)
        criteres.append(
            {
                "nom": "MCC >= MCC_ancien - epsilon",
                "valeur_nouveau": mcc_nouveau,
                "valeur_reference": mcc_ancien,
                "seuil": mcc_ancien - MCC_EPSILON,
                "pass": mcc_pass,
                "description": (
                    f"MCC nouveau={mcc_nouveau:.4f} vs ancien={mcc_ancien:.4f} "
                    f"(tolerance={MCC_EPSILON})"
                ),
            }
        )
    else:
        mcc_pass = mcc_nouveau > 0.0  # Premier modele : au moins mieux que l'aleatoire
        criteres.append(
            {
                "nom": "MCC > 0 (premier modele)",
                "valeur_nouveau": mcc_nouveau,
                "valeur_reference": None,
                "seuil": 0.0,
                "pass": mcc_pass,
                "description": (
                    f"Premier modele : MCC={mcc_nouveau:.4f} doit etre > 0 (mieux que l'aleatoire)"
                ),
            }
        )

    # Critere 2 : Recall Green IT (garde-fou metier)
    recall_nouveau = float(new_metrics.get("recall", 0.0))
    recall_pass = recall_nouveau >= MIN_RECALL_GREEN_IT
    criteres.append(
        {
            "nom": f"Recall Green IT >= {MIN_RECALL_GREEN_IT}",
            "valeur_nouveau": recall_nouveau,
            "valeur_reference": None,
            "seuil": MIN_RECALL_GREEN_IT,
            "pass": recall_pass,
            "description": (
                f"Recall sur Green IT={recall_nouveau:.4f} "
                f"(plancher={MIN_RECALL_GREEN_IT}, evite les faux negatifs)"
            ),
        }
    )

    # Critere 3 : Non-regression F1 (seulement si on a une reference)
    f1_nouveau = float(new_metrics.get("f1", 0.0))
    if best_metrics is not None and "f1" in best_metrics:
        f1_ancien = float(best_metrics["f1"])
        f1_seuil = f1_ancien * F1_REGRESSION_TOLERANCE
        f1_pass = f1_nouveau >= f1_seuil
        criteres.append(
            {
                "nom": f"F1 >= F1_ancien * {F1_REGRESSION_TOLERANCE}",
                "valeur_nouveau": f1_nouveau,
                "valeur_reference": f1_ancien,
                "seuil": f1_seuil,
                "pass": f1_pass,
                "description": (
                    f"F1 nouveau={f1_nouveau:.4f} vs ancien={f1_ancien:.4f} "
                    f"(tolerance={int((1 - F1_REGRESSION_TOLERANCE) * 100)}%, "
                    f"seuil={f1_seuil:.4f})"
                ),
            }
        )
    else:
        f1_pass = True
        criteres.append(
            {
                "nom": "Non-regression F1 (non applicable)",
                "valeur_nouveau": f1_nouveau,
                "valeur_reference": None,
                "seuil": None,
                "pass": True,
                "description": "Pas de modele de reference, critere ignore",
            }
        )

    # Critere 4 (optionnel) : stabilite entre folds si on dispose des metriques CV
    cv_aggregated = new_metrics.get("cv_aggregated")
    if cv_aggregated and "mcc" in cv_aggregated:
        mcc_std = float(cv_aggregated["mcc"].get("std", 0.0))
        # Le plafond depend du nombre de Green IT : avec peu de positifs, la
        # variance inter-folds est mecanique, on tolere donc un seuil plus large.
        n_green = int(new_metrics.get("vrais_positifs", 0) + new_metrics.get("faux_negatifs", 0))
        max_std = _max_mcc_std(n_green)
        stability_pass = mcc_std <= max_std
        criteres.append(
            {
                "nom": f"Stabilite CV (std MCC <= {max_std})",
                "valeur_nouveau": mcc_std,
                "valeur_reference": None,
                "seuil": max_std,
                "pass": stability_pass,
                "description": (
                    f"Ecart-type du MCC sur les folds = {mcc_std:.4f} "
                    f"(plafond={max_std} ajuste pour {n_green} Green IT, "
                    "au-dela = performance instable)"
                ),
            }
        )

    retenu = all(c["pass"] for c in criteres)
    if retenu:
        raison = "Tous les criteres de promotion sont satisfaits"
    else:
        echecs = [c["nom"] for c in criteres if not c["pass"]]
        raison = f"Critere(s) echoue(s) : {', '.join(echecs)}"

    return {"retenu": retenu, "criteres": criteres, "raison": raison}


def _log_promotion_verdict(verdict: dict) -> None:
    """Affiche le detail de l'evaluation des criteres de promotion."""
    logger.info("")
    logger.info("-" * 80)
    logger.info("  EVALUATION DES CRITERES DE PROMOTION")
    logger.info("-" * 80)

    for i, critere in enumerate(verdict["criteres"], 1):
        statut = "OK   " if critere["pass"] else "ECHEC"
        logger.info(f"  [{statut}] {i}. {critere['nom']}")
        logger.info(f"          {critere['description']}")

    logger.info("")
    if verdict["retenu"]:
        logger.info("  VERDICT : NOUVEAU MODELE RETENU")
        logger.info(f"           {verdict['raison']}")
    else:
        logger.warning("  VERDICT : ANCIEN MODELE CONSERVE")
        logger.warning(f"           {verdict['raison']}")


async def step_benchmark() -> dict | None:
    """Compare le nouveau modele au meilleur historique et au baseline.

    Applique un critere composite (MCC + recall Green IT + non-regression F1
    + stabilite CV si disponible) adapte au desequilibre extreme du dataset.

    Si un rapport K-fold recent (`cv_report.json`) est present et plus recent
    que le modele entraine, ses metriques agregees sont utilisees au lieu
    d'une evaluation sur un split 80/20. Cela donne une comparaison plus
    robuste basee sur 5 folds de validation.

    Returns:
        Resultats avec verdict (nouveau_est_meilleur: bool), ou None si erreur.
    """
    from greentech.ai.models.training import load_golden_dataset

    logger.info("=" * 70)
    logger.info("  BENCHMARK : NOUVEAU vs MEILLEUR HISTORIQUE vs BASELINE")
    logger.info("=" * 70)

    if not TRAIN_DIR.exists() or not (TRAIN_DIR / "adapter_config.json").exists():
        logger.error(f"Modele entraine introuvable dans {TRAIN_DIR}")
        return None

    # Si un rapport K-fold recent existe, on l'utilise en priorite (beaucoup plus fiable)
    cv_report = _load_json(CV_REPORT_FILE)
    new_metrics: dict

    if cv_report and _is_cv_report_fresh():
        logger.info(f"Utilisation du rapport K-fold ({cv_report.get('n_splits')} folds)")
        new_metrics = dict(cv_report["global"])
        new_metrics["cv_aggregated"] = cv_report["aggregated"]
        new_metrics["cv_folds"] = cv_report["folds"]
        test_set_size = sum(f["n_test"] for f in cv_report["folds"])
        new_metrics["total_echantillons"] = test_set_size
        _log_detailed_metrics(new_metrics, "Nouveau modele (K-fold CV)")
        _log_cv_variability(cv_report["aggregated"])
    else:
        if cv_report:
            logger.warning(
                "Rapport CV present mais obsolete vs le modele entraine, "
                "basculement sur evaluation test-split 80/20"
            )
        _, _, test_texts, test_labels = load_golden_dataset(oversample=False)
        test_set_size = len(test_texts)
        logger.info(f"Test set : {test_set_size} articles")
        new_metrics = await _evaluate_new_model(test_texts, test_labels)

    best_ref = _load_json(BEST_METRICS_FILE)
    baseline_ref = _load_json(BASELINE_METRICS_FILE)

    logger.info("")
    logger.info("=" * 80)
    logger.info("  RAPPORT COMPARATIF DETAILLE")
    logger.info("=" * 80)

    if baseline_ref:
        baseline_model_name = baseline_ref.get("model", "modele de base")
        _log_detailed_metrics(
            baseline_ref["metrics"],
            f"Baseline ({baseline_model_name} sans fine-tuning)",
        )
        baseline_metrics = baseline_ref["metrics"]
        gain_f1 = new_metrics["f1"] - baseline_metrics["f1"]
        gain_mcc = new_metrics["mcc"] - baseline_metrics.get("mcc", 0.0)
        logger.info(
            f"  >>> Gain du fine-tuning vs baseline : "
            f"{'+' if gain_mcc >= 0 else ''}{gain_mcc:.4f} MCC, "
            f"{'+' if gain_f1 >= 0 else ''}{gain_f1:.4f} F1"
        )

    if best_ref:
        _log_detailed_metrics(
            best_ref["metrics"],
            f"Meilleur precedent ({best_ref['version']})",
        )
        old_metrics = best_ref["metrics"]
        delta_mcc = new_metrics["mcc"] - old_metrics.get("mcc", 0.0)
        delta_f1 = new_metrics["f1"] - old_metrics["f1"]
        logger.info(
            f"  >>> Delta vs meilleur precedent : "
            f"{'+' if delta_mcc >= 0 else ''}{delta_mcc:.4f} MCC, "
            f"{'+' if delta_f1 >= 0 else ''}{delta_f1:.4f} F1"
        )

    verdict = _evaluate_promotion_criteria(
        new_metrics,
        best_ref["metrics"] if best_ref else None,
    )
    _log_promotion_verdict(verdict)
    logger.info("=" * 80)

    report = {
        "date": datetime.now(UTC).isoformat(),
        "test_set_size": test_set_size,
        "nouveau": new_metrics,
        "meilleur_precedent": best_ref["metrics"] if best_ref else None,
        "baseline": baseline_ref["metrics"] if baseline_ref else None,
        "verdict": "nouveau_retenu" if verdict["retenu"] else "ancien_conserve",
        "criteres_promotion": verdict["criteres"],
        "raison": verdict["raison"],
    }
    report_path = BASE_DIR / "data" / "benchmark_versions.json"
    report_path.write_text(json.dumps(report, indent=2))

    return {"nouveau": new_metrics, "nouveau_est_meilleur": verdict["retenu"]}


# =============================================================================
# PROMOTION CONDITIONNELLE
# =============================================================================


async def step_auto_promote() -> bool:
    """Benchmark + promotion UNIQUEMENT si le nouveau modele est meilleur."""
    results = await step_benchmark()
    if results is None:
        logger.error("Benchmark echoue, pas de promotion")
        return False

    if not results["nouveau_est_meilleur"]:
        logger.info("ANCIEN MODELE CONSERVE EN PRODUCTION (pas de regression)")
        return True

    new_metrics = results["nouveau"]
    version_tag = _generate_version_tag()

    step_archive_current_production()

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file() and f.name != "README.md":
            f.unlink()

    fichiers = []
    for f in TRAIN_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, PRODUCTION_DIR / f.name)
            fichiers.append(f.name)

    _save_best_metrics(new_metrics, version_tag)

    meta = {
        "promoted_at": datetime.now(UTC).isoformat(),
        "version": version_tag,
        "metrics": new_metrics,
        "fichiers": fichiers,
    }
    (PRODUCTION_DIR / "promotion_info.json").write_text(json.dumps(meta, indent=2))

    logger.info(
        f"NOUVEAU MODELE PROMU : {version_tag} "
        f"(MCC={new_metrics.get('mcc', 0.0):.4f}, "
        f"F1={new_metrics['f1']:.4f}, "
        f"Recall Green IT={new_metrics.get('recall', 0.0):.4f})"
    )
    logger.info("Redemarrez l'API pour charger le nouveau modele.")
    return True


def step_force_promote() -> bool:
    """Promotion manuelle forcee, sans benchmark."""
    logger.info("=" * 70)
    logger.info("  PROMOTION FORCEE")
    logger.info("=" * 70)

    source = TRAIN_DIR
    if not source.exists() or not (source / "adapter_config.json").exists():
        logger.error(f"Modele introuvable dans {source}")
        return False

    step_archive_current_production()

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for f in PRODUCTION_DIR.iterdir():
        if f.is_file() and f.name != "README.md":
            f.unlink()

    fichiers = []
    for f in source.iterdir():
        if f.is_file():
            shutil.copy2(f, PRODUCTION_DIR / f.name)
            fichiers.append(f.name)

    (PRODUCTION_DIR / "promotion_info.json").write_text(
        json.dumps(
            {
                "promoted_at": datetime.now(UTC).isoformat(),
                "forced": True,
                "fichiers": fichiers,
            },
            indent=2,
        )
    )

    logger.info(f"Modele promu de force : {len(fichiers)} fichiers")
    return True


# =============================================================================
# ORCHESTRATION
# =============================================================================


async def run_pipeline(steps: list[str]) -> None:
    """Execute les etapes du pipeline dans l'ordre demande."""
    start = time.perf_counter()
    logger.info("")
    logger.info("#" * 70)
    logger.info("#  PIPELINE DE RE-ENTRAINEMENT — GreenTech Intelligence")
    logger.info(f"#  Etapes : {', '.join(steps)}")
    logger.info(f"#  Date   : {datetime.now(UTC).isoformat()}")
    logger.info("#" * 70)
    logger.info("")

    for step_name in steps:
        if step_name.startswith("ingest-file:"):
            file_path = step_name.split(":", 1)[1]
            if not step_ingest_file(file_path):
                logger.error(f"Interrompu a ingest-file ({file_path})")
                return
            continue

        if step_name == "benchmark":
            await step_benchmark()
            continue
        if step_name == "auto-promote":
            await step_auto_promote()
            continue
        if step_name == "baseline":
            await step_baseline()
            continue
        if step_name == "train-cv":
            logger.info("\n>>> Re-entrainement Qwen3-4B (K-fold CV)...")
            if await step_train_cv() is None:
                logger.error("Interrompu a : train-cv")
                return
            continue

        # Etapes executees en appel Python direct (async) au lieu de
        # subprocess pour eviter la saturation du PIPE stdout/stderr sous
        # Windows (incident 16 avril 2026 : freeze systematique apres ~4
        # articles du fait des logs SQLAlchemy verbose qui remplissent le
        # buffer PIPE 4-8 KB). Le chargement du modele Qwen est mutualise
        # entre summarize-classif et summarize-green (economie ~5 min de
        # cold start).
        async_steps = {
            "clean": (
                "Nettoyage articles inexploitables",
                step_clean,
            ),
            "summarize-classif": (
                "Resumes de classification (tous articles)",
                step_summarize_classification,
            ),
            "annotate": ("Pre-filtre mots-cles (etage 1)", step_annotate),
            "classify": ("LLM judge - verification candidats (etage 2)", step_classify),
            "summarize-green": (
                "Resumes ecologiques (Green IT uniquement)",
                step_summarize_green,
            ),
            # Alias historique : "summarize" declenche les deux passes
            # (classification puis green) pour compatibilite avec les anciennes
            # commandes. A eviter dans les nouveaux scripts qui preferent les
            # deux noms explicites.
            "summarize": ("Resumes (classification + Green IT)", step_summarize_all),
            "export-golden": ("Export golden_dataset.csv", step_export_golden),
        }

        # Etapes qui restent en subprocess : `collect` (Spark a besoin d'un
        # process Python dedie pour sa JVM), `train`, `promote`. Le volume
        # de logs de ces etapes est bien plus faible que celui de
        # summarize-classif, donc pas de risque de saturation PIPE.
        sync_steps = {
            "collect": ("Collecte des donnees", step_collect),
            "train": ("Re-entrainement Qwen3-4B (split 80/20)", step_train),
            "promote": ("Promotion forcee", step_force_promote),
        }

        if step_name in async_steps:
            label, async_func = async_steps[step_name]
            logger.info(f"\n>>> {label}...")
            if not await async_func():
                logger.error(f"Interrompu a : {step_name}")
                return
            continue

        if step_name not in sync_steps:
            valid = (
                list(async_steps.keys())
                + list(sync_steps.keys())
                + ["benchmark", "auto-promote", "baseline", "train-cv"]
            )
            logger.error(f"Etape inconnue : {step_name}")
            logger.info(f"Valides : {', '.join(valid)}, ingest-file <path>")
            return

        label, func = sync_steps[step_name]
        logger.info(f"\n>>> {label}...")
        if not func():
            logger.error(f"Interrompu a : {step_name}")
            return

    elapsed = time.perf_counter() - start
    logger.info("")
    logger.info("#" * 70)
    logger.info(f"#  PIPELINE TERMINE en {int(elapsed // 60)}m {int(elapsed % 60)}s")
    logger.info("#" * 70)


def main() -> None:
    """Point d'entree CLI."""
    # Activer le logging persistant : console + fichier rotatif logs/greentech_<date>.log
    # (+ Loki si accessible). Sans cet appel, loguru n'ecrit que sur stderr.
    from greentech.utils.logger import setup_logging

    # Loki active : les logs de l'orchestrateur sont centralises dans Grafana
    # pour suivre l'avancement du pipeline en direct via l'Explore Loki.
    setup_logging(level="INFO", enable_loki=True)

    # Ajoute le dossier scripts/ au sys.path pour permettre les imports
    # directs des modules `auto_annotate_dataset`, `classify_candidates` et
    # `export_golden_dataset` depuis les `step_*` async. Ces modules ne
    # sont pas dans le package `greentech` (ils sont a la racine de
    # scripts/) donc on leur donne acces via le sys.path.
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    args = sys.argv[1:]

    processed: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "ingest-file" and i + 1 < len(args):
            processed.append(f"ingest-file:{args[i + 1]}")
            i += 2
        else:
            processed.append(args[i])
            i += 1

    if not processed:
        # Pipeline complet recommande :
        # 1. collect            : collecte depuis toutes les sources
        # 2. clean              : supprime les articles inexploitables
        #                         (contenu NULL, < 50 chars, placeholder
        #                         NewsData "ONLY AVAILABLE IN PAID PLANS")
        # 3. summarize-classif  : resume LLM style abstract pour TOUS les
        #                         articles (source unique de verite de la
        #                         feature d'entrainement)
        # 4. annotate           : pre-filtre mots-cles (etage 1, binaire
        #                         CANDIDATE/NON_GREEN) sur le contenu brut
        # 5. classify           : LLM judge sur les candidats (etage 2,
        #                         decision finale) sur le contenu brut
        # 6. summarize-green    : resume ecologique pour les Green IT confirmes
        # 7. export-golden      : regenere golden_dataset.csv depuis la DB
        #                         post-classification avec `resume_classification`
        # 8. train-cv           : re-entraine Qwen3-4B avec K-fold CV sur le
        #                         nouveau golden (feature = titre + resume)
        # 9. auto-promote       : benchmark et promotion conditionnelle
        processed = [
            "collect",
            "clean",
            "summarize-classif",
            "annotate",
            "classify",
            "summarize-green",
            "export-golden",
            "train-cv",
            "auto-promote",
        ]

    asyncio.run(run_pipeline(processed))


if __name__ == "__main__":
    main()
