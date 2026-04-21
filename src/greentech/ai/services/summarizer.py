"""Services de resume d'articles avec persistance en base de donnees.

Ce module orchestre les deux types de resumes utilises dans l'application
et leur persistance dans la table ``articles`` :

- **Resume de classification** (colonne ``articles.resume``) : synthese
  dense style abstract scientifique, generee pour **tous** les articles
  via le prompt centralise dans ``classification_summarizer.py``. Ce
  resume sert simultanement (1) d'entree canonique au classifieur
  Qwen3-4B + LoRA et (2) de contenu affiche dans l'UI cote utilisateur.

- **Resume ecologique** (colonne ``articles.resume_ecologique``) : extrait
  les aspects Green IT saillants (efficacite energetique, sobriete
  numerique, e-waste, data centers durables, IA frugale, etc.) via un
  prompt systeme specialise. Genere **uniquement** pour les articles
  confirmes Green IT (``est_green_it = True``), en complement du resume
  general.

Les fonctions ``summarize_article`` et ``summarize_green_for_article``
forment le contrat public utilise par ``routes/analyze.py`` pour chaque
analyse interactive. Les fonctions batch ``summarize_all_articles_for_classification``
et ``summarize_green_only`` sont utilisees par le pipeline de
ré-entrainement pour peupler la base avant l'entrainement.

Les appels LLM sont factorises via ``llm_dispatcher`` qui bascule
automatiquement sur le fallback local Qwen (GPU AMD ROCm) lorsque le
quota HF Serverless mensuel est epuise.

"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import or_, select, update

from greentech.ai.services.classification_summarizer import (
    summarize_for_classification,
)
from greentech.ai.services.llm_dispatcher import (
    chat_completion,
    is_hf_quota_exhausted,
)
from greentech.config import get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# =============================================================================
# PROMPT DEDIE AU RESUME ECOLOGIQUE (Green IT uniquement)
# =============================================================================
#
# Ce prompt est volontairement distinct du prompt de classification defini
# dans `classification_summarizer.py` : il cible une information plus
# specifique (aspects ecologiques saillants) et s'adresse a un public plus
# avise qui consulte la page detail d'un article deja classifie Green IT.

GREEN_SYSTEM_PROMPT = (
    "Tu es un analyste specialise en Green IT et en informatique eco-responsable. "
    "Ta mission est de lire un article technologique identifie comme Green IT "
    "et d'en extraire les aspects ecologiques les plus saillants, en restant "
    "concis, factuel et fidele au texte source."
)

GREEN_USER_PROMPT_TEMPLATE = (
    "Analyse l'article ci-dessous et redige un court resume (3 a 5 phrases, "
    "environ 80 a 120 mots) en francais, centre exclusivement sur les aspects "
    "ecologiques / Green IT qu'il aborde. Mentionne : les technologies durables, "
    "les pratiques d'efficacite energetique, la reduction d'empreinte carbone, "
    "l'eco-conception logicielle, ou l'usage du numerique au service de la "
    "transition ecologique, selon ce qui est effectivement evoque.\n\n"
    "Ne mentionne que ce qui est explicitement dans l'article. Si un aspect "
    "n'est pas aborde, ne l'invente pas.\n\n"
    "Article :\n{texte}\n\n"
    "Resume des aspects ecologiques :"
)

GREEN_MAX_NEW_TOKENS = 250
GREEN_TEMPERATURE = 0.3

# Borne du contexte envoye au LLM pour le resume ecologique. Ce resume
# etant affiche en UI (pas utilise pour l'entrainement), on peut se
# permettre un contexte plus genereux que pour la classification.
GREEN_MAX_INPUT_CHARS = 10_000
GREEN_MIN_INPUT_CHARS = 50


@dataclass(frozen=True)
class SummaryResult:
    """Resultat d'un appel de resume attache a un article en base.

    Attributes:
        id_article: Identifiant de l'article en base. Mis a 0 pour les
            appels "purs" (sans article en base, ex. texte ad-hoc).
        resume: Texte du resume genere, ou None en cas d'echec.
        temps_ms: Temps d'inference en millisecondes.
        modele: Nom du backend utilise (modele HF ou ``"qwen_local"``).
        succes: True si le LLM a retourne un resume non vide.
        erreur: Message d'erreur eventuel (None si succes).
    """

    id_article: int
    resume: str | None
    temps_ms: int
    modele: str
    succes: bool
    erreur: str | None = None


def _truncate_green_input(text: str) -> str:
    """Tronque le texte pour le resume ecologique en preservant les phrases.

    Args:
        text: Texte brut de l'article.

    Returns:
        Texte eventuellement tronque a ``GREEN_MAX_INPUT_CHARS``.
    """
    if len(text) <= GREEN_MAX_INPUT_CHARS:
        return text

    truncated = text[:GREEN_MAX_INPUT_CHARS]
    last_period = truncated.rfind(".")
    if last_period > GREEN_MAX_INPUT_CHARS // 2:
        truncated = truncated[: last_period + 1]

    logger.debug(f"Texte tronque pour resume ecologique : {len(text)} -> {len(truncated)} chars")
    return truncated


def _backend_tag(model_name: str) -> str:
    """Retourne l'identifiant du backend actif (HF ou local)."""
    return "qwen_local" if is_hf_quota_exhausted() else model_name


# =============================================================================
# RESUME DE CLASSIFICATION (general, pour tous les articles)
# =============================================================================


async def summarize_text(text: str) -> SummaryResult:
    """Genere un resume de classification a partir d'un texte brut.

    Alias de compatibilite : delegue a ``summarize_for_classification``
    du module ``classification_summarizer`` et enrobe le resultat dans
    le ``SummaryResult`` attendu par les anciens appelants.

    Les nouveaux appelants sont encourages a importer directement
    ``summarize_for_classification`` pour eviter le detour par ce wrapper.

    Args:
        text: Contenu brut de l'article a resumer.

    Returns:
        Resultat contenant le resume en francais et les metriques.
    """
    pure = await summarize_for_classification(text)
    return SummaryResult(
        id_article=0,
        resume=pure.resume,
        temps_ms=pure.temps_ms,
        modele=pure.modele,
        succes=pure.succes,
        erreur=pure.erreur,
    )


async def summarize_article(
    article_id: int, *, session: AsyncSession | None = None
) -> SummaryResult:
    """Genere un resume de classification pour un article en base et le persiste.

    Lit le contenu de l'article, appelle le LLM via le prompt de
    classification centralise, puis met a jour la colonne ``articles.resume``.
    C'est cette colonne qui sert ensuite d'entree au classifieur
    Qwen3-4B + LoRA (a l'entrainement comme a l'inference).

    Args:
        article_id: Identifiant de l'article en base.
        session: Session SQLAlchemy optionnelle. Si ``None``, une nouvelle
            session est creee automatiquement.

    Returns:
        Resultat du resume avec metriques.

    Raises:
        ValueError: Si l'article n'existe pas en base.
    """
    if session is not None:
        return await _summarize_article_impl(article_id, session)

    async with async_session_factory() as own_session:
        return await _summarize_article_impl(article_id, own_session)


async def _summarize_article_impl(article_id: int, session: AsyncSession) -> SummaryResult:
    """Implementation interne : lit l'article, genere le resume, persiste."""
    stmt = select(Article).where(Article.id_article == article_id)
    result = await session.execute(stmt)
    article = result.scalar_one_or_none()

    if article is None:
        msg = f"Article id={article_id} introuvable"
        raise ValueError(msg)

    if not article.contenu:
        logger.warning(f"Article id={article_id} sans contenu, resume impossible")
        return SummaryResult(
            id_article=article_id,
            resume=None,
            temps_ms=0,
            modele=get_settings().huggingface_model_summarizer,
            succes=False,
            erreur="Article sans contenu",
        )

    pure = await summarize_for_classification(article.contenu)
    summary_result = SummaryResult(
        id_article=article_id,
        resume=pure.resume,
        temps_ms=pure.temps_ms,
        modele=pure.modele,
        succes=pure.succes,
        erreur=pure.erreur,
    )

    if summary_result.succes and summary_result.resume:
        stmt_update = (
            update(Article)
            .where(Article.id_article == article_id)
            .values(resume=summary_result.resume)
        )
        await session.execute(stmt_update)
        await session.commit()
        logger.info(
            f"Article id={article_id} : resume stocke en base ({len(summary_result.resume)} chars)"
        )

    return summary_result


# =============================================================================
# RESUME ECOLOGIQUE (Green IT confirmes uniquement)
# =============================================================================


async def summarize_green_aspects(text: str) -> SummaryResult:
    """Genere un resume oriente Green IT via le LLM instructif.

    N'a de sens que pour des articles identifies comme Green IT : extrait
    les aspects ecologiques saillants (efficacite energetique, sobriete
    numerique, e-waste, data centers durables, IA frugale, etc.) via un
    prompt systeme specialise.

    Comme le resume de classification, passe par le dispatcher qui bascule
    automatiquement sur le modele Qwen local si le quota HF est epuise.

    Args:
        text: Texte complet de l'article a analyser.

    Returns:
        Resultat contenant le resume ecologique et les metriques d'inference.
    """
    settings = get_settings()
    modele = settings.huggingface_model_green_summarizer

    if not text or len(text.strip()) < GREEN_MIN_INPUT_CHARS:
        logger.warning(f"Texte trop court pour le resume ecologique ({len(text)} chars)")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=0,
            modele=modele,
            succes=False,
            erreur="Texte trop court pour generer un resume",
        )

    truncated = _truncate_green_input(text.strip())
    messages = [
        {"role": "system", "content": GREEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GREEN_USER_PROMPT_TEMPLATE.format(texte=truncated),
        },
    ]

    start = time.perf_counter()
    try:
        completion = await chat_completion(
            messages=messages,
            max_tokens=GREEN_MAX_NEW_TOKENS,
            temperature=GREEN_TEMPERATURE,
            model_hf=modele,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        contenu = completion.choices[0].message.content if completion.choices else ""
        summary = contenu.strip()
        backend = _backend_tag(modele)
        logger.info(
            f"Resume ecologique genere en {elapsed_ms}ms ({len(summary)} chars, backend={backend})"
        )

        if not summary:
            return SummaryResult(
                id_article=0,
                resume=None,
                temps_ms=elapsed_ms,
                modele=backend,
                succes=False,
                erreur="Le LLM a retourne un resume vide",
            )

        return SummaryResult(
            id_article=0,
            resume=summary,
            temps_ms=elapsed_ms,
            modele=backend,
            succes=True,
        )

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.error(f"Echec resume ecologique ({elapsed_ms}ms) : {exc}")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=elapsed_ms,
            modele=_backend_tag(modele),
            succes=False,
            erreur=str(exc),
        )


async def summarize_green_for_article(
    article_id: int, *, session: AsyncSession | None = None
) -> SummaryResult:
    """Resume les aspects ecologiques d'un article en base et persiste.

    Complement de ``summarize_article`` : genere un resume oriente Green IT
    et le stocke dans la colonne ``articles.resume_ecologique``. A n'appeler
    que pour des articles classifies Green IT.

    Args:
        article_id: Identifiant de l'article en base.
        session: Session SQLAlchemy optionnelle.

    Returns:
        Resultat du resume avec metriques.
    """
    if session is not None:
        return await _summarize_green_impl(article_id, session)

    async with async_session_factory() as own_session:
        return await _summarize_green_impl(article_id, own_session)


async def _summarize_green_impl(article_id: int, session: AsyncSession) -> SummaryResult:
    """Implementation interne : lit l'article, genere le resume ecologique, persiste."""
    stmt = select(Article).where(Article.id_article == article_id)
    result = await session.execute(stmt)
    article = result.scalar_one_or_none()

    if article is None:
        msg = f"Article id={article_id} introuvable"
        raise ValueError(msg)

    if not article.contenu:
        return SummaryResult(
            id_article=article_id,
            resume=None,
            temps_ms=0,
            modele=get_settings().huggingface_model_green_summarizer,
            succes=False,
            erreur="Article sans contenu",
        )

    green_result = await summarize_green_aspects(article.contenu)
    green_result = SummaryResult(
        id_article=article_id,
        resume=green_result.resume,
        temps_ms=green_result.temps_ms,
        modele=green_result.modele,
        succes=green_result.succes,
        erreur=green_result.erreur,
    )

    if green_result.succes and green_result.resume:
        stmt_update = (
            update(Article)
            .where(Article.id_article == article_id)
            .values(resume_ecologique=green_result.resume)
        )
        await session.execute(stmt_update)
        await session.commit()
        logger.info(
            f"Article id={article_id} : resume ecologique persiste "
            f"({len(green_result.resume)} chars)"
        )

    return green_result


# =============================================================================
# ROUTINES BATCH (pipeline de re-entrainement)
# =============================================================================


async def summarize_all_articles_for_classification(
    *,
    limit: int | None = None,
    force: bool = False,
    delay_seconds: float = 0.3,
    shuffle: bool = False,
) -> dict[str, int]:
    """Genere le resume de classification pour les articles en base.

    Point d'entree recommande dans le pipeline de re-classification :
    parcourt tous les articles ayant un ``contenu`` non vide et peuple
    leur ``resume``. Par defaut, n'ecrase pas les resumes existants
    (comportement idempotent, reprise en cas d'interruption). Utiliser
    ``force=True`` pour re-generer integralement le dataset, par exemple
    apres une mise a jour du prompt de classification.

    Args:
        limit: Plafond optionnel sur le nombre d'articles traites. Si None,
            traite la totalite.
        force: Si True, ecrase aussi les resumes deja presents. Sinon, ne
            traite que les articles ou ``resume IS NULL``.
        delay_seconds: Pause entre deux articles pour respecter le
            fair-use HF Serverless et eviter les erreurs 429.
        shuffle: Si True, selectionne les articles dans un ordre aleatoire
            (``ORDER BY RANDOM()``) au lieu de l'ordre ``id_article``. Utile
            pour les smoke tests sur un echantillon representatif de toutes
            les sources plutot que les premiers articles inseres (qui sont
            presque tous de la meme source dominante). Non recommande pour
            un run complet sans ``limit`` car le plan de requete est moins
            efficace qu'un tri par cle primaire.

    Returns:
        Statistiques ``{total, succes, echecs}``.
    """
    from sqlalchemy import func

    stats = {"total": 0, "succes": 0, "echecs": 0}

    from greentech.ai.services.classification_summarizer import (
        CLASSIFICATION_MIN_INPUT_CHARS,
    )

    async with async_session_factory() as session:
        # On pre-filtre les articles dont le contenu est trop court pour le
        # LLM (< CLASSIFICATION_MIN_INPUT_CHARS = 50 chars). Ces articles
        # echoueront systematiquement au resume et ne changeront jamais
        # (abstracts arXiv corrompus/tronques dans le dump Kaggle). Les
        # exclure evite de les re-tenter a chaque run et de gonfler le
        # compteur d'echecs au-dela du seuil d'arret du pipeline.
        stmt = (
            select(Article.id_article)
            .where(Article.contenu.isnot(None))
            .where(func.length(Article.contenu) >= CLASSIFICATION_MIN_INPUT_CHARS)
        )
        if not force:
            stmt = stmt.where(Article.resume.is_(None))
        stmt = stmt.order_by(func.random() if shuffle else Article.id_article)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        article_ids = [row[0] for row in result.all()]

    stats["total"] = len(article_ids)
    if not article_ids:
        logger.info("Aucun article en attente de resume de classification")
        return stats

    logger.info(
        f"Resume de classification : {len(article_ids)} articles a traiter "
        f"(force={force}, delay={delay_seconds}s)"
    )

    for index, id_article in enumerate(article_ids, start=1):
        logger.info(f"[{index}/{len(article_ids)}] Article {id_article}")
        try:
            result_summary = await summarize_article(id_article)
            if result_summary.succes:
                stats["succes"] += 1
            else:
                stats["echecs"] += 1
        except Exception as exc:
            logger.error(f"Echec resume classification article {id_article} : {exc}")
            stats["echecs"] += 1

        if index < len(article_ids) and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    logger.info("Bilan resumes de classification :")
    logger.info(f"  Total articles : {stats['total']}")
    logger.info(f"  Succes         : {stats['succes']}")
    logger.info(f"  Echecs         : {stats['echecs']}")
    return stats


async def summarize_green_only(
    *,
    limit: int | None = None,
    force: bool = False,
    delay_seconds: float = 0.3,
) -> dict[str, int]:
    """Genere le resume ecologique pour les articles Green IT confirmes.

    Parcourt les articles ou ``est_green_it = True`` et peuple la colonne
    ``articles.resume_ecologique`` si elle est NULL. Complementaire a
    ``summarize_all_articles_for_classification`` qui, lui, peuple la
    colonne ``resume``.

    Args:
        limit: Plafond optionnel sur le nombre d'articles traites.
        force: Si True, regenere meme les resumes ecologiques existants.
        delay_seconds: Pause entre deux articles (fair-use HF).

    Returns:
        Statistiques ``{total, succes, echecs}``.
    """
    stats = {"total": 0, "succes": 0, "echecs": 0}

    async with async_session_factory() as session:
        stmt = (
            select(Article.id_article)
            .where(Article.contenu.isnot(None))
            .where(Article.est_green_it.is_(True))
        )
        if not force:
            stmt = stmt.where(Article.resume_ecologique.is_(None))
        stmt = stmt.order_by(Article.id_article)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        article_ids = [row[0] for row in result.all()]

    stats["total"] = len(article_ids)
    if not article_ids:
        logger.info("Aucun article Green IT en attente de resume ecologique")
        return stats

    logger.info(f"Resume ecologique : {len(article_ids)} articles Green IT a traiter")

    for index, id_article in enumerate(article_ids, start=1):
        logger.info(f"[{index}/{len(article_ids)}] Article {id_article}")
        try:
            result_summary = await summarize_green_for_article(id_article)
            if result_summary.succes:
                stats["succes"] += 1
            else:
                stats["echecs"] += 1
        except Exception as exc:
            logger.error(f"Echec resume ecologique article {id_article} : {exc}")
            stats["echecs"] += 1

        if index < len(article_ids) and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    logger.info("Bilan resumes ecologiques :")
    logger.info(f"  Total articles : {stats['total']}")
    logger.info(f"  Succes         : {stats['succes']}")
    logger.info(f"  Echecs         : {stats['echecs']}")
    return stats


async def summarize_green_it_articles(
    *,
    limit: int = 200,
    include_ecological: bool = True,
    delay_seconds: float = 0.3,
) -> dict[str, int]:
    """Genere en parallele les deux resumes pour les articles Green IT.

    Conservee pour compatibilite avec les anciens scripts CLI qui appelaient
    cette routine unique. Pour les nouveaux flux, prefere
    ``summarize_all_articles_for_classification`` suivie de
    ``summarize_green_only`` qui separent clairement les deux etapes.

    Args:
        limit: Plafond optionnel sur le nombre d'articles traites.
        include_ecological: Si True, genere aussi le resume ecologique
            (``resume_ecologique``) en parallele du resume de classification.
        delay_seconds: Pause entre deux articles.

    Returns:
        Statistiques ``{total, succes_general, succes_ecologique, echecs}``.
    """
    stats = {
        "total": 0,
        "succes_general": 0,
        "succes_ecologique": 0,
        "echecs": 0,
    }

    async with async_session_factory() as session:
        missing_any = or_(Article.resume.is_(None), Article.resume_ecologique.is_(None))
        stmt = (
            select(Article.id_article)
            .where(Article.contenu.isnot(None))
            .where(Article.est_green_it.is_(True))
            .where(missing_any)
            .order_by(Article.id_article)
            .limit(limit)
        )
        result = await session.execute(stmt)
        article_ids = [row[0] for row in result.all()]

    stats["total"] = len(article_ids)
    if not article_ids:
        logger.info("Aucun article Green IT en attente de resume")
        return stats

    logger.info(f"Resumes Green IT : {len(article_ids)} articles (ecological={include_ecological})")

    for index, id_article in enumerate(article_ids, start=1):
        logger.info(f"[{index}/{len(article_ids)}] Article {id_article}")
        try:
            if include_ecological:
                general, ecological = await asyncio.gather(
                    summarize_article(id_article),
                    summarize_green_for_article(id_article),
                )
                if general.succes:
                    stats["succes_general"] += 1
                else:
                    stats["echecs"] += 1
                if ecological.succes:
                    stats["succes_ecologique"] += 1
            else:
                general = await summarize_article(id_article)
                if general.succes:
                    stats["succes_general"] += 1
                else:
                    stats["echecs"] += 1
        except Exception as exc:
            logger.error(f"Echec resume article {id_article} : {exc}")
            stats["echecs"] += 1

        if index < len(article_ids) and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    logger.info("Bilan resumes Green IT :")
    logger.info(f"  Total articles         : {stats['total']}")
    logger.info(f"  Resumes generaux OK    : {stats['succes_general']}")
    logger.info(f"  Resumes ecologiques OK : {stats['succes_ecologique']}")
    logger.info(f"  Echecs                 : {stats['echecs']}")
    return stats


async def run_summarizer() -> None:
    """Point d'entree CLI : genere les resumes manquants en deux passes.

    1. Peuple ``articles.resume`` pour tous les articles sans resume
       (y compris Non Green IT).
    2. Peuple ``articles.resume_ecologique`` pour les Green IT confirmes.

    Utilise en ligne de commande via
    ``uv run python -m greentech.ai.services.summarizer``.
    """
    logger.info("Demarrage du service de resume automatique")
    settings = get_settings()
    logger.info(f"Modele classification : {settings.huggingface_model_summarizer}")
    logger.info(f"Modele ecologique     : {settings.huggingface_model_green_summarizer}")

    stats_classif = await summarize_all_articles_for_classification()
    stats_green = await summarize_green_only()

    if stats_classif["total"] == 0 and stats_green["total"] == 0:
        logger.info("Aucun article en attente de resume")


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_summarizer())
