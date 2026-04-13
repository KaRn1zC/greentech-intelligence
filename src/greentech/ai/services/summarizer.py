"""Service de résumé automatique via Hugging Face Serverless Inference API.

Connecte l'application à l'API d'inférence Hugging Face pour générer
deux types de résumés d'articles technologiques :

- **Résumé général** (toujours généré) : synthèse neutre et informative en français.
- **Résumé orienté Green IT** (uniquement si l'article est classifié Green IT) :
  extraction des aspects écologiques / durabilité mis en avant par l'article.

Les deux résumés utilisent le même LLM instructif (`Qwen/Qwen2.5-7B-Instruct`)
via des prompts système distincts. Cette architecture mono-modèle simplifie
l'infrastructure (un seul service SaaS), garantit une cohérence linguistique
(les deux résumés sont en français) et maintient une qualité homogène entre
le résumé général et le résumé spécialisé.

Les résultats sont persistés dans `articles.resume` et `articles.resume_ecologique`.

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select, update

from greentech.ai.services.llm_dispatcher import (
    chat_completion,
    is_hf_quota_exhausted,
)
from greentech.config import get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Bornes du texte en entree. Qwen2.5-7B-Instruct supporte 32k tokens mais on
# limite a 3000 caracteres pour garder une latence raisonnable et eviter les
# coupures excessives dans les articles tres longs.
MAX_INPUT_CHARS = 3000
MIN_INPUT_CHARS = 50

# === Prompts pour le resume general (toujours genere) ===

GENERAL_SYSTEM_PROMPT = (
    "Tu es un redacteur technique specialise dans la synthese d'articles "
    "technologiques. Tu produis des resumes neutres, factuels et concis, "
    "fideles au contenu source, destines a un public francophone curieux "
    "de technologie."
)

GENERAL_USER_PROMPT_TEMPLATE = (
    "Redige un resume clair et synthetique de l'article ci-dessous, "
    "en francais, en 3 a 5 phrases (environ 80 a 120 mots). Le resume doit "
    "restituer les informations les plus importantes : le sujet, les acteurs "
    "cles, les chiffres ou donnees notables, et la conclusion ou l'enjeu "
    "principal. Reste neutre, ne donne pas ton avis, ne reformule pas "
    "l'article par une analyse personnelle.\n\n"
    "Si l'article est en anglais, traduis-le en francais pour le resume.\n\n"
    "Article :\n{texte}\n\n"
    "Resume en francais :"
)

GENERAL_MAX_NEW_TOKENS = 250
GENERAL_TEMPERATURE = 0.3  # Faible creativite : on reste fidele au texte

# === Prompts pour le resume "aspects ecologiques" (si Green IT) ===

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


@dataclass(frozen=True)
class SummaryResult:
    """Résultat d'un appel de résumé sur un article.

    Attributes:
        id_article: Identifiant de l'article traité.
        resume: Texte du résumé généré.
        temps_ms: Temps d'inférence en millisecondes.
        modele: Nom du modèle utilisé.
        succes: Indique si le résumé a été généré avec succès.
        erreur: Message d'erreur éventuel.
    """

    id_article: int
    resume: str | None
    temps_ms: int
    modele: str
    succes: bool
    erreur: str | None = None


def _truncate_text(text: str) -> str:
    """Tronque le texte d'entrée à la taille maximale pour l'inférence.

    Qwen2.5-7B-Instruct accepte 32k tokens mais on limite a 3000 caracteres
    pour garder une latence raisonnable (~2s) et eviter les surcharges cote
    API. On coupe a la derniere phrase complete pour preserver le sens.

    Args:
        text: Texte brut de l'article.

    Returns:
        Texte tronqué prêt pour l'inférence.
    """
    if len(text) <= MAX_INPUT_CHARS:
        return text

    truncated = text[:MAX_INPUT_CHARS]
    # Couper à la dernière phrase complète (point, point d'exclamation, etc.)
    last_period = truncated.rfind(".")
    if last_period > MAX_INPUT_CHARS // 2:
        truncated = truncated[: last_period + 1]

    logger.debug(f"Texte tronqué : {len(text)} → {len(truncated)} caractères")
    return truncated


def _backend_tag(model_name: str) -> str:
    """Renvoie le nom de backend a tracer en base apres un appel LLM reussi."""
    return "qwen_local" if is_hf_quota_exhausted() else model_name


async def summarize_text(text: str) -> SummaryResult:
    """Genere un resume general en francais via le LLM instructif Qwen.

    Utilise d'abord l'API Hugging Face Serverless. Si le quota mensuel HF
    est epuise (HTTP 402), le dispatcher bascule automatiquement sur le
    modele Qwen local (GPU AMD ROCm) pour la suite de la session, sans
    interrompre la generation.

    Args:
        text: Texte de l'article a resumer (toute langue).

    Returns:
        Resultat contenant le resume en francais et les metriques d'inference.
    """
    settings = get_settings()
    modele = settings.huggingface_model_summarizer

    if not text or len(text.strip()) < MIN_INPUT_CHARS:
        logger.warning(f"Texte trop court pour le resume ({len(text)} chars)")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=0,
            modele=modele,
            succes=False,
            erreur="Texte trop court pour generer un resume",
        )

    truncated = _truncate_text(text.strip())
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERAL_USER_PROMPT_TEMPLATE.format(texte=truncated),
        },
    ]

    start = time.perf_counter()
    try:
        completion = await chat_completion(
            messages=messages,
            max_tokens=GENERAL_MAX_NEW_TOKENS,
            temperature=GENERAL_TEMPERATURE,
            model_hf=modele,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        contenu = completion.choices[0].message.content if completion.choices else ""
        summary = contenu.strip()
        backend = _backend_tag(modele)
        logger.info(
            f"Resume general genere en {elapsed_ms}ms "
            f"({len(summary)} chars, backend={backend})"
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
        logger.error(f"Echec resume ({elapsed_ms}ms) : {exc}")
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=elapsed_ms,
            modele=_backend_tag(modele),
            succes=False,
            erreur=str(exc),
        )


async def summarize_green_aspects(text: str) -> SummaryResult:
    """Genere un resume oriente Green IT via le LLM instructif Qwen.

    N'a de sens que pour des articles identifies comme Green IT par le
    classifieur : extrait les aspects ecologiques saillants (efficacite
    energetique, sobriete numerique, e-waste, data center durables,
    IA frugale, etc.) via un prompt systeme specialise.

    Comme `summarize_text`, passe par le dispatcher qui bascule
    automatiquement sur le modele Qwen local si le quota HF est epuise.

    Args:
        text: Texte complet de l'article a analyser.

    Returns:
        Resultat contenant le resume ecologique et les metriques d'inference.
    """
    settings = get_settings()
    modele = settings.huggingface_model_green_summarizer

    if not text or len(text.strip()) < MIN_INPUT_CHARS:
        logger.warning(
            f"Texte trop court pour le resume ecologique ({len(text)} chars)"
        )
        return SummaryResult(
            id_article=0,
            resume=None,
            temps_ms=0,
            modele=modele,
            succes=False,
            erreur="Texte trop court pour generer un resume",
        )

    truncated = _truncate_text(text.strip())
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
            f"Resume ecologique genere en {elapsed_ms}ms "
            f"({len(summary)} chars, backend={backend})"
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
    """Resume les aspects ecologiques d'un article en base et persiste le resultat.

    Complement de `summarize_article` : genere un resume orientes Green IT et
    le stocke dans la colonne `articles.resume_ecologique`. A n'appeler que
    pour des articles classifies Green IT.

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


async def _summarize_green_impl(
    article_id: int, session: AsyncSession
) -> SummaryResult:
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


async def summarize_article(
    article_id: int, *, session: AsyncSession | None = None
) -> SummaryResult:
    """Résume un article depuis la base de données et stocke le résultat.

    Lit le contenu de l'article, appelle l'API de résumé, puis met à jour
    la colonne `resume` dans la table `articles`.

    Args:
        article_id: Identifiant de l'article en base.
        session: Session SQLAlchemy optionnelle (créée automatiquement si absente).

    Returns:
        Résultat du résumé avec les métriques.

    Raises:
        ValueError: Si l'article n'existe pas ou n'a pas de contenu.
    """
    if session is not None:
        return await _summarize_article_impl(article_id, session)

    async with async_session_factory() as own_session:
        return await _summarize_article_impl(article_id, own_session)


async def _summarize_article_impl(
    article_id: int, session: AsyncSession
) -> SummaryResult:
    """Implementation interne du resume d'article avec session geree par l'appelant.

    Args:
        article_id: Identifiant de l'article en base.
        session: Session SQLAlchemy active.

    Returns:
        Resultat du resume avec les metriques.
    """
    stmt = select(Article).where(Article.id_article == article_id)
    result = await session.execute(stmt)
    article = result.scalar_one_or_none()

    if article is None:
        msg = f"Article id={article_id} introuvable"
        raise ValueError(msg)

    if not article.contenu:
        logger.warning(f"Article id={article_id} sans contenu, résumé impossible")
        return SummaryResult(
            id_article=article_id,
            resume=None,
            temps_ms=0,
            modele=get_settings().huggingface_model_summarizer,
            succes=False,
            erreur="Article sans contenu",
        )

    summary_result = await summarize_text(article.contenu)
    summary_result = SummaryResult(
        id_article=article_id,
        resume=summary_result.resume,
        temps_ms=summary_result.temps_ms,
        modele=summary_result.modele,
        succes=summary_result.succes,
        erreur=summary_result.erreur,
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
            f"Article id={article_id} : résumé stocké en base "
            f"({len(summary_result.resume)} chars)"
        )

    return summary_result


async def summarize_batch(
    *,
    limit: int = 50,
    force: bool = False,
    green_it_only: bool = True,
) -> list[SummaryResult]:
    """Résume un lot d'articles qui n'ont pas encore de résumé.

    Sélectionne les articles sans résumé (colonne `resume` IS NULL)
    et les traite séquentiellement pour respecter les limites de l'API.

    Par défaut, le batch ne traite que les articles **confirmés Green IT**
    (`est_green_it = True`) afin d'éviter de saturer l'API Hugging Face
    sur l'ensemble du corpus et de cibler uniquement les articles pertinents
    pour l'usage du projet.

    Args:
        limit: Nombre maximum d'articles à traiter.
        force: Si True, re-résume aussi les articles avec un résumé existant.
        green_it_only: Si True (défaut), restreint aux articles Green IT
            confirmés. Passer `False` pour traiter tout le corpus (non
            recommandé sur de gros volumes).

    Returns:
        Liste des résultats de résumé pour chaque article traité.
    """
    async with async_session_factory() as session:
        stmt = select(Article).where(Article.contenu.isnot(None))
        if green_it_only:
            stmt = stmt.where(Article.est_green_it.is_(True))
        if not force:
            stmt = stmt.where(Article.resume.is_(None))
        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        articles = result.scalars().all()

        if not articles:
            logger.info("Aucun article à résumer")
            return []

        logger.info(
            f"Résumé de {len(articles)} articles en cours "
            f"(green_it_only={green_it_only})..."
        )
        results: list[SummaryResult] = []

        for article in articles:
            summary = await summarize_article(article.id_article, session=session)
            results.append(summary)

        succes = sum(1 for r in results if r.succes)
        echecs = len(results) - succes
        temps_total = sum(r.temps_ms for r in results)

        logger.info(
            f"Batch terminé : {succes} résumés / {echecs} échecs "
            f"({temps_total}ms total, ~{temps_total // max(len(results), 1)}ms/article)"
        )

        return results


async def summarize_green_it_articles(
    *,
    limit: int = 200,
    include_ecological: bool = True,
    delay_seconds: float = 0.3,
) -> dict[str, int]:
    """Pipeline de résumé pour les articles Green IT confirmés.

    Point d'entrée recommandé dans le pipeline de re-classification :
    parcourt tous les articles avec `est_green_it = True` et sans résumé
    (`resume IS NULL`), génère en parallèle le résumé général et le résumé
    écologique, puis persiste les deux en base.

    Args:
        limit: Plafond optionnel sur le nombre d'articles traités.
        include_ecological: Si True, génère aussi le résumé écologique
            (`resume_ecologique`) en parallèle. Si False, seul le résumé
            général est produit.
        delay_seconds: Pause entre deux articles pour respecter le fair-use HF.

    Returns:
        Statistiques {total, succes_general, succes_ecologique, echecs}.
    """
    import asyncio as _asyncio

    stats = {
        "total": 0,
        "succes_general": 0,
        "succes_ecologique": 0,
        "echecs": 0,
    }

    from sqlalchemy import or_

    async with async_session_factory() as session:
        # On cible les articles Green IT dont l'un au moins des deux resumes
        # manque. Cela permet de reessayer automatiquement les cas ou l'appel
        # parallele a reussi pour un resume mais echoue pour l'autre.
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
        logger.info("Aucun article Green IT en attente de résumé")
        return stats

    logger.info(
        f"Résumé Green IT : {len(article_ids)} articles "
        f"(ecological={include_ecological})"
    )

    for index, id_article in enumerate(article_ids, start=1):
        logger.info(f"[{index}/{len(article_ids)}] Article {id_article}")
        try:
            if include_ecological:
                general, ecological = await _asyncio.gather(
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
            await _asyncio.sleep(delay_seconds)

    logger.info("Bilan résumés Green IT :")
    logger.info(f"  Total articles        : {stats['total']}")
    logger.info(f"  Résumés généraux OK   : {stats['succes_general']}")
    logger.info(f"  Résumés écologiques OK : {stats['succes_ecologique']}")
    logger.info(f"  Échecs                : {stats['echecs']}")
    return stats


async def run_summarizer() -> None:
    """Point d'entrée principal pour le résumé en lot.

    Traite tous les articles Green IT confirmés sans résumé en base.
    Utilisé en ligne de commande via
    `uv run python -m greentech.ai.services.summarizer`.
    """
    logger.info("Démarrage du service de résumé automatique")
    settings = get_settings()
    logger.info(f"Modèle : {settings.huggingface_model_summarizer}")

    stats = await summarize_green_it_articles()
    if stats["total"] == 0:
        logger.info("Aucun article en attente de résumé")


if __name__ == "__main__":
    import asyncio

    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_summarizer())
