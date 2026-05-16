"""Re-classification LLM v2 enrichie pour les borderline des sources mixtes (B2.10).

Ce script applique une seconde passe LLM, plus rigoureuse, aux 300 articles
borderline (score_confiance 0.3-0.7) restants apres l'auto-correction des
sources pures (cf. ``auto_correct_pure_sources.py``). Cible : les sources
mixtes scientifiques (Crossref, arXiv) et generalistes (Guardian, Dev.to,
TechCrunch Climate) ou la decision Green IT n'est pas triviale.

Strategie de re-classification
------------------------------

Pour chaque article borderline, le script effectue **3 passes LLM** avec
des temperatures differentes (0.1, 0.3, 0.5). Le prompt systeme est enrichi
par rapport a la passe initiale :

1. Ajout du **contexte de la source** : "L'article provient de Crossref,
   base de donnees de publications peer-reviewed..." aide le LLM a calibrer
   son seuil (un paper Crossref qui parle d'optimisation energetique est
   typiquement Green IT, alors qu'un paper Crossref purement medical ne
   l'est pas).
2. Ajout d'**exemples francais explicites** dans le few-shot : le biais
   linguistique anglais du LLM justifie l'observation empirique sur GreenIT.fr.
3. **3 inferences** + vote majoritaire = robustesse aux variations
   stochastiques (chaque passe seule peut hesiter, le triple-vote stabilise).

Decision finale
---------------

- **Vote majoritaire** sur les 3 verdicts booleens (au moins 2/3 d'accord).
- **Confiance moyenne** = moyenne des 3 scores.
- Si confiance moyenne >= 0.8 ET vote unanime (3/3), **decision automatique** :
  ``annotation_source='llm_judge_v2'``, ``annotated_by='system_llm_v2'``.
- Sinon, l'article **reste en l'etat** (borderline) pour annotation manuelle
  dans P2.2. Plus pragmatique que de risquer un faux positif/negatif sur
  un cas vraiment incertain.

Les articles deja annotes manuellement (``annotation_source='manual'``) ou
deja auto-corriges (``annotation_source='auto_source_pure'``) ne sont JAMAIS
re-classifies.

Usage
-----

Preview (dry-run, aucune modification BDD, aucun appel LLM) ::

    uv run python scripts/reclassify_borderline_v2.py --dry-run

Lancement reel (~75 min sur GPU local, 300 articles x 3 passes ~ 5s/passe) ::

    uv run python scripts/reclassify_borderline_v2.py

Restreindre a une seule source ::

    uv run python scripts/reclassify_borderline_v2.py --source Crossref

Rollback (annule toutes les corrections LLM v2) ::

    uv run python scripts/reclassify_borderline_v2.py --rollback

"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import and_, or_, select, update

from greentech.ai.services.classifier_llm import _parse_verdict, _truncate_content
from greentech.ai.services.llm_dispatcher import chat_completion, is_hf_quota_exhausted
from greentech.config import get_settings
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

# Sources mixtes ciblees par la re-classification LLM v2.
MIXED_SOURCES: tuple[str, ...] = (
    "Crossref",
    "arXiv Dataset",
    "arXiv API",
    "The Guardian",
    "Dev.to",
    "TechCrunch Climate",
)

# Contexte editorial injecte dans le prompt selon la source.
SOURCE_CONTEXT: dict[str, str] = {
    "Crossref": (
        "Crossref, base de donnees de publications scientifiques peer-reviewed. "
        "Les abstracts sont denses et techniques. Un article qui mentionne "
        "explicitement l'efficacite energetique, la sobriete numerique, "
        "l'empreinte carbone des modeles ou des infrastructures, ou "
        "l'eco-conception logicielle est typiquement Green IT, meme si "
        "ce n'est pas son sujet principal."
    ),
    "arXiv Dataset": (
        "arXiv (corpus historique Kaggle), preprints scientifiques toutes "
        "categories CS, EESS et stat.ML confondues. Beaucoup d'articles de "
        "ML purs (sans angle energetique). Ne classer Green IT que si "
        "l'angle energetique ou environnemental est explicite dans l'abstract."
    ),
    "arXiv API": (
        "arXiv (preprints recents), focus sur les requetes Green IT (carbon-aware, "
        "energy-efficient ML, green computing). Forte probabilite a priori que "
        "l'article soit Green IT - on cherche surtout a eliminer les hors-sujet."
    ),
    "The Guardian": (
        "The Guardian, sections environment et technology. Journalisme generaliste "
        "anglophone. Beaucoup d'articles environnementaux purs (climat, biodiversite) "
        "qui NE sont PAS Green IT si l'angle numerique manque. Etre prudent."
    ),
    "Dev.to": (
        "Dev.to, blog tech communautaire. Tags Green IT/sustainability filtres. "
        "Articles plutot pratiques (tutoriels, retours d'experience). Forte "
        "probabilite a priori que l'article soit Green IT."
    ),
    "TechCrunch Climate": (
        "TechCrunch section Climate. Journalisme tech ciblant les startups climat. "
        "Articles souvent business (levee de fonds, accord commercial). Un article "
        "n'est Green IT que si l'angle technologique numerique est clairement present."
    ),
}

# Fenetre de score consideree borderline.
SCORE_MIN = 0.3
SCORE_MAX = 0.7

# Marqueurs de tracabilite.
ANNOTATION_SOURCE_TAG = "llm_judge_v2"
ANNOTATED_BY_TAG = "system_llm_v2"
MODELE_CLASSIFICATION_TAG = "keyword_filter+qwen_llm_judge_v2"

# Seuil de confiance pour accepter une decision automatique.
CONFIDENCE_THRESHOLD = 0.8

# Temperatures successives pour les 3 passes (de tres deterministe a explorant).
PASS_TEMPERATURES: tuple[float, float, float] = (0.1, 0.3, 0.5)

# Limites d'entree LLM.
MAX_NEW_TOKENS = 220

# Pause entre articles (HF rate limit). Neutralisee en mode local.
DELAY_BETWEEN_ARTICLES = 0.5

# Taille des batchs de commit (kill-safe : perte max 20 articles ~ 5 min).
BATCH_COMMIT_SIZE = 20


# === Prompt systeme enrichi v2 ===
# Ajout par rapport au prompt v1 :
# - Contexte explicite de la source (injecte dynamiquement)
# - Exemples francais en few-shot
# - Insistance sur la regle du doute raisonnable
SYSTEM_PROMPT_V2 = (
    "Tu es un expert en Green IT charge de classifier des articles "
    "technologiques avec rigueur. Ta mission : determiner si un article aborde "
    "le Green IT de maniere significative.\n\n"
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
    "Sont EXCLUS (Non Green IT) :\n"
    "  - recherche IA/ML purement theorique portant uniquement sur la "
    "precision ou la complexite algorithmique, sans consideration energetique ;\n"
    "  - cryptomonnaies, rapports boursiers et previsions de marche ;\n"
    "  - cybersecurite pure, gaming, metaverse, reseaux sociaux, "
    "smartphones grand public (meme si tech) ;\n"
    "  - energies renouvelables ou vehicules electriques SANS lien avec "
    "le numerique ;\n"
    "  - sujets sante ou sciences appliquees qui utilisent simplement de l'IA "
    "sans aborder son impact environnemental.\n\n"
    "Exemples illustratifs (decisions correctes) :\n"
    "  - 'Comment j'ai reduit l'empreinte carbone de mon site WordPress de 40 %' "
    "-> Green IT (eco-conception logicielle concrete).\n"
    "  - 'Efficient Transformer Training via Dynamic Quantization' "
    "-> Green IT (optimisation energetique implicite des modeles ML).\n"
    "  - 'Top 10 Cryptocurrencies to Watch in 2025' "
    "-> Non Green IT (crypto pur sans angle energetique).\n"
    "  - 'Deep Learning for Early Diagnosis of Cardiac Arrhythmias' "
    "-> Non Green IT (IA en sante, pas d'impact environnemental).\n"
    "  - 'Apple Silicon M4 vs Intel Core Ultra : efficacite energetique' "
    "-> Green IT (efficacite materiel IT).\n"
    "  - 'Smart Grids and the Future of Renewable Energy' (sans angle data/AI) "
    "-> Non Green IT (energie sans angle numerique explicite).\n\n"
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

USER_PROMPT_V2_TEMPLATE = (
    "Source editoriale : {source_context}\n\n"
    "Classifie l'article suivant comme Green IT ou non, selon la definition "
    "donnee. Rappelle-toi : en cas de doute raisonnable, penche vers Green IT. "
    "Renvoie uniquement le JSON demande.\n\n"
    "Titre : {titre}\n\n"
    "Contenu :\n{contenu}\n\n"
    "JSON de classification :"
)


@dataclass(frozen=True)
class V2Decision:
    """Resultat agrege des 3 passes LLM pour un article."""

    est_green_it: bool | None
    confidence_mean: float
    pass_decisions: tuple[bool | None, ...]
    pass_confidences: tuple[float, ...]
    raison: str
    unanimous: bool
    accepted: bool


async def _fetch_borderline_candidates(
    source_filter: str | None,
) -> list[tuple[int, str, str, str]]:
    """Recupere les articles borderline a re-classifier.

    Args:
        source_filter: Si fourni, restreint a une seule source par nom.

    Returns:
        Liste de tuples (id_article, titre, contenu, nom_source).
    """
    sources_to_query = (source_filter,) if source_filter else MIXED_SOURCES
    async with async_session_factory() as session:
        stmt = (
            select(Article.id_article, Article.titre, Article.contenu, Source.nom)
            .join(Source, Source.id_source == Article.id_source)
            .where(
                and_(
                    Source.nom.in_(sources_to_query),
                    Article.score_confiance >= SCORE_MIN,
                    Article.score_confiance <= SCORE_MAX,
                    or_(
                        Article.annotation_source.is_(None),
                        Article.annotation_source.notin_(
                            ("manual", "auto_source_pure", ANNOTATION_SOURCE_TAG)
                        ),
                    ),
                    Article.contenu.isnot(None),
                )
            )
            .order_by(Article.id_article)
        )
        result = await session.execute(stmt)
        return [(row[0], row[1], row[2] or "", row[3]) for row in result.all()]


async def _single_pass(
    titre: str,
    contenu: str,
    source_context: str,
    temperature: float,
) -> tuple[bool | None, float, str]:
    """Une passe LLM unique. Renvoie (est_green_it, confiance, raison).

    Retourne (None, 0.0, "") en cas d'echec.
    """
    settings = get_settings()
    contenu_t = _truncate_content(contenu)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {
            "role": "user",
            "content": USER_PROMPT_V2_TEMPLATE.format(
                source_context=source_context,
                titre=titre or "(titre manquant)",
                contenu=contenu_t,
            ),
        },
    ]
    try:
        response = await chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            model_hf=settings.huggingface_model_classifier_llm,
        )
        raw = (response.choices[0].message.content or "").strip()
        est_green, confiance, raison = _parse_verdict(raw)
        return est_green, confiance, raison
    except Exception as exc:
        logger.debug(f"Passe T={temperature} echouee : {exc!r}")
        return None, 0.0, ""


async def _three_passes(
    titre: str,
    contenu: str,
    source_name: str,
) -> V2Decision:
    """Trois passes LLM avec temperatures differentes + agregation."""
    source_context = SOURCE_CONTEXT.get(source_name, source_name)
    decisions: list[bool | None] = []
    confidences: list[float] = []
    reasons: list[str] = []

    for temp in PASS_TEMPERATURES:
        d, c, r = await _single_pass(titre, contenu, source_context, temp)
        decisions.append(d)
        confidences.append(c)
        if r:
            reasons.append(r)

    # Vote majoritaire en ignorant les None (echecs de passe).
    valid_decisions = [d for d in decisions if d is not None]
    if not valid_decisions:
        return V2Decision(
            est_green_it=None,
            confidence_mean=0.0,
            pass_decisions=tuple(decisions),
            pass_confidences=tuple(confidences),
            raison="(3 passes echouees)",
            unanimous=False,
            accepted=False,
        )

    n_green = sum(1 for d in valid_decisions if d is True)
    n_non_green = len(valid_decisions) - n_green
    majority = n_green > n_non_green
    unanimous = (n_green == len(valid_decisions)) or (n_non_green == len(valid_decisions))

    valid_conf = [c for d, c in zip(decisions, confidences, strict=True) if d is not None]
    confidence_mean = sum(valid_conf) / len(valid_conf) if valid_conf else 0.0

    # Decision acceptee si : unanimite des passes valides ET confiance moyenne suffisante.
    accepted = unanimous and confidence_mean >= CONFIDENCE_THRESHOLD

    # Premiere raison non vide.
    raison = reasons[0] if reasons else ""

    return V2Decision(
        est_green_it=majority,
        confidence_mean=confidence_mean,
        pass_decisions=tuple(decisions),
        pass_confidences=tuple(confidences),
        raison=raison,
        unanimous=unanimous,
        accepted=accepted,
    )


async def _persist_decision(id_article: int, decision: V2Decision) -> None:
    """Ecrit la decision v2 en base de donnees."""
    now_utc = datetime.now(UTC)
    async with async_session_factory() as session:
        stmt = (
            update(Article)
            .where(Article.id_article == id_article)
            .values(
                est_green_it=decision.est_green_it,
                score_confiance=decision.confidence_mean,
                raison_llm_judge=decision.raison or None,
                modele_classification=MODELE_CLASSIFICATION_TAG,
                annotation_source=ANNOTATION_SOURCE_TAG,
                annotated_at=now_utc,
                annotated_by=ANNOTATED_BY_TAG,
            )
        )
        await session.execute(stmt)
        await session.commit()


async def _rollback() -> int:
    """Annule toutes les corrections llm_judge_v2 et restaure llm_judge."""
    async with async_session_factory() as session:
        stmt = (
            update(Article)
            .where(Article.annotation_source == ANNOTATION_SOURCE_TAG)
            .values(
                annotation_source="llm_judge",
                annotated_at=None,
                annotated_by=None,
                modele_classification="keyword_filter+qwen_llm_judge",
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        return result.rowcount or 0


async def run(dry_run: bool, rollback: bool, source_filter: str | None) -> None:
    """Point d'entree principal."""
    if rollback:
        logger.warning("Mode ROLLBACK : annulation des corrections llm_judge_v2")
        n_restored = await _rollback()
        logger.success(f"{n_restored} articles restaures vers annotation_source='llm_judge'")
        return

    candidates = await _fetch_borderline_candidates(source_filter)
    n_total = len(candidates)

    logger.info("=" * 70)
    logger.info("Re-classification LLM v2 des borderline (sources mixtes)")
    logger.info("=" * 70)
    logger.info(f"  Source filtree : {source_filter or 'toutes les sources mixtes'}")
    logger.info(f"  Articles a traiter : {n_total}")
    logger.info(f"  Passes par article : {len(PASS_TEMPERATURES)} (T={PASS_TEMPERATURES})")
    logger.info(f"  Seuil de confiance : >= {CONFIDENCE_THRESHOLD}")
    logger.info("=" * 70)

    if dry_run:
        # En dry-run, ventiler par source pour transparence.
        by_source: dict[str, int] = {}
        for _, _, _, src in candidates:
            by_source[src] = by_source.get(src, 0) + 1
        for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
            logger.info(f"  {src:25s} {count:>4d} articles")
        logger.warning("Mode DRY-RUN : aucun appel LLM, aucune modification BDD")
        return

    if n_total == 0:
        logger.info("Rien a faire : aucun article borderline restant sur les sources mixtes")
        return

    n_accepted_green = 0
    n_accepted_non_green = 0
    n_left_borderline = 0
    start = time.perf_counter()

    for index, (id_article, titre, contenu, source_name) in enumerate(candidates, start=1):
        decision = await _three_passes(titre, contenu, source_name)

        if decision.accepted:
            await _persist_decision(id_article, decision)
            if decision.est_green_it:
                n_accepted_green += 1
            else:
                n_accepted_non_green += 1
            symbol = "G" if decision.est_green_it else "N"
        else:
            n_left_borderline += 1
            symbol = "?"

        logger.info(
            f"[{index}/{n_total}] {symbol} id={id_article} src={source_name} "
            f"votes={decision.pass_decisions} conf={decision.confidence_mean:.2f} "
            f"{'ACCEPT' if decision.accepted else 'LEFT'}"
        )

        if (
            index < n_total
            and DELAY_BETWEEN_ARTICLES > 0
            and not is_hf_quota_exhausted()
        ):
            await asyncio.sleep(DELAY_BETWEEN_ARTICLES)

    elapsed = time.perf_counter() - start
    logger.info("=" * 70)
    logger.success(
        f"Re-classification v2 terminee en {elapsed / 60:.1f} min "
        f"({elapsed / max(n_total, 1):.1f} s/article)"
    )
    logger.info(f"  Accepted Green IT : {n_accepted_green}")
    logger.info(f"  Accepted Non Green IT : {n_accepted_non_green}")
    logger.info(f"  Laisses borderline (pour annotation manuelle) : {n_left_borderline}")
    logger.info("=" * 70)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-classification LLM v2 enrichie des borderline sources mixtes (B2.10)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apercu sans appel LLM ni modification BDD.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Annule toutes les corrections llm_judge_v2 precedentes.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=f"Restreindre a une source unique (parmi {MIXED_SOURCES}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run and args.rollback:
        logger.error("--dry-run et --rollback sont mutuellement exclusifs.")
        raise SystemExit(1)
    asyncio.run(run(dry_run=args.dry_run, rollback=args.rollback, source_filter=args.source))
