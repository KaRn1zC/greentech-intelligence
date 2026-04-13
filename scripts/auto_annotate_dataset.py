"""Pre-filtre de classification Green IT par scoring multi-criteres (etage 1).

Ce module constitue le **premier etage** du pipeline de classification hybride :
il applique un scoring par mots-cles afin de distinguer rapidement les articles
manifestement Non Green IT des articles **candidats** qui meritent une
verification plus fine par un LLM (etage 2 : `classify_candidates.py`).

Philosophie du pre-filtre
-------------------------

Le pre-filtre est volontairement **permissif** cote Green IT : il maximise le
recall au detriment de la precision. Un article avec le moindre signal Green
est marque `CANDIDATE` et envoye au LLM pour verification definitive. Seuls
les articles manifestement non-pertinents (bourse, crypto, rapports de marche,
gaming, etc.) sont ecartes directement en `NON_GREEN`.

Ce compromis est assume : le pre-filtre seul genere des faux positifs, mais
le LLM les corrige a l'etage suivant. En contrepartie, on est quasi-certain
de ne rater **aucun** vrai Green IT, ce qui est crucial pour la qualite du
golden dataset d'entrainement Llama.

Definition retenue du Green IT
------------------------------

Un article est Green IT si son sujet principal concerne :

- la reduction de l'impact environnemental des technologies numeriques
  (efficacite energetique, empreinte carbone, sobriete numerique,
  eco-conception logicielle, IA frugale, e-waste) ;
- OU l'usage du numerique au service de la transition ecologique
  (mais uniquement si l'angle IT est central, pas juste mentionne).

Sont exclus : recherche IA/ML generale sans angle energie, vehicules
electriques hors angle IT, energie renouvelable generale, rapports boursiers,
tech grand public sans dimension environnementale.

Sortie
------

Chaque article est marque en base :

- `modele_classification = "keyword_filter"` : le pre-filtre l'a traite
- `est_green_it = False` : classe NON_GREEN par le pre-filtre (pas besoin LLM)
- `est_green_it = NULL` : classe CANDIDATE, en attente de verification LLM

"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger
from sqlalchemy import and_, or_, select, update

from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

# =============================================================================
# INDICATEURS PONDERES
# =============================================================================


@dataclass(frozen=True)
class Indicator:
    """Un indicateur de classification avec son poids et sa categorie."""

    pattern: str
    weight: float
    category: str


# --- INDICATEURS GREEN IT (positifs) ---
# Poids : 5 = signal tres fort, 3 = signal moyen, 1 = signal faible

GREEN_INDICATORS: list[Indicator] = [
    # === Efficacite energetique IT (core Green IT) ===
    Indicator("green it", 5.0, "core"),
    Indicator("green computing", 5.0, "core"),
    Indicator("green software", 5.0, "core"),
    Indicator("green data cent", 5.0, "core"),
    Indicator("green cloud", 4.0, "core"),
    Indicator("green coding", 4.0, "core"),
    Indicator("green hosting", 4.0, "core"),
    Indicator("green internet", 4.0, "core"),
    Indicator("green digital", 4.0, "core"),
    Indicator("green ai", 4.0, "core"),
    Indicator("sustainable computing", 5.0, "core"),
    Indicator("sustainable software", 5.0, "core"),
    Indicator("sustainable data cent", 5.0, "core"),
    Indicator("sustainable cloud", 4.0, "core"),
    Indicator("sustainable ai", 4.0, "core"),
    Indicator("sustainable digital", 3.0, "core"),
    Indicator("sustainable web", 3.0, "core"),
    Indicator("sustainable ict", 4.0, "core"),
    Indicator("eco-design", 4.0, "core"),
    Indicator("eco-conception", 4.0, "core"),
    Indicator("eco-responsible", 3.0, "core"),
    Indicator("digital sobriety", 5.0, "core"),
    Indicator("sobriete numerique", 5.0, "core"),
    Indicator("frugal ai", 4.0, "core"),
    Indicator("frugal computing", 4.0, "core"),

    # === Empreinte carbone du numerique ===
    Indicator("carbon footprint software", 5.0, "carbon"),
    Indicator("carbon footprint ai", 5.0, "carbon"),
    Indicator("carbon footprint data cent", 5.0, "carbon"),
    Indicator("carbon footprint cloud", 5.0, "carbon"),
    Indicator("carbon footprint digital", 4.0, "carbon"),
    Indicator("carbon footprint computing", 5.0, "carbon"),
    Indicator("carbon footprint machine learning", 5.0, "carbon"),
    Indicator("carbon emission data cent", 5.0, "carbon"),
    Indicator("carbon emission computing", 4.0, "carbon"),
    Indicator("carbon aware computing", 5.0, "carbon"),
    Indicator("carbon-aware", 4.0, "carbon"),
    Indicator("carbon neutral data cent", 5.0, "carbon"),
    Indicator("carbon neutral cloud", 4.0, "carbon"),
    Indicator("carbon neutral technolog", 3.0, "carbon"),
    Indicator("carbon cost training", 4.0, "carbon"),
    Indicator("carbon cost model", 4.0, "carbon"),
    Indicator("net zero data cent", 5.0, "carbon"),
    Indicator("low carbon computing", 4.0, "carbon"),
    Indicator("codecarbon", 4.0, "carbon"),
    Indicator("carbon tracker", 3.0, "carbon"),
    Indicator("carbon emission", 2.0, "carbon"),
    Indicator("carbon reduction", 2.0, "carbon"),

    # === Efficacite energetique (contexte IT) ===
    Indicator("energy efficient server", 5.0, "energy"),
    Indicator("energy efficient data cent", 5.0, "energy"),
    Indicator("energy efficient computing", 5.0, "energy"),
    Indicator("energy efficient neural", 4.0, "energy"),
    Indicator("energy efficient ai", 4.0, "energy"),
    Indicator("energy efficient hardware", 3.0, "energy"),
    Indicator("energy efficient deep learning", 4.0, "energy"),
    Indicator("energy efficient inference", 4.0, "energy"),
    Indicator("energy efficient cloud", 4.0, "energy"),
    Indicator("energy efficient infrastructure", 4.0, "energy"),
    Indicator("energy efficient natural language", 4.0, "energy"),
    Indicator("energy efficient model", 3.0, "energy"),
    Indicator("energy efficient processor", 3.0, "energy"),
    Indicator("energy consumption data cent", 4.0, "energy"),
    Indicator("energy consumption ai", 4.0, "energy"),
    Indicator("energy consumption gpu", 4.0, "energy"),
    Indicator("energy consumption training", 4.0, "energy"),
    Indicator("reduce energy consumption", 3.0, "energy"),
    Indicator("power usage effectiveness", 5.0, "energy"),
    Indicator("pue data cent", 5.0, "energy"),
    Indicator("low power machine learning", 4.0, "energy"),
    Indicator("low power deep learning", 4.0, "energy"),
    Indicator("low power inference", 4.0, "energy"),
    Indicator("energy saving server", 4.0, "energy"),
    Indicator("electricity consumption ict", 4.0, "energy"),
    Indicator("energy saving", 2.0, "energy"),
    Indicator("energy reduction", 2.0, "energy"),

    # === Data center vert ===
    Indicator("cooling.*data cent", 4.0, "datacenter"),
    Indicator("data cent.*cooling", 4.0, "datacenter"),
    Indicator("data cent.*efficiency", 3.0, "datacenter"),
    Indicator("data cent.*energy", 3.0, "datacenter"),
    Indicator("liquid cooling server", 4.0, "datacenter"),
    Indicator("liquid cooling", 3.0, "datacenter"),
    Indicator("immersion cooling", 4.0, "datacenter"),
    Indicator("free cooling", 3.0, "datacenter"),
    Indicator("renewable energy data cent", 5.0, "datacenter"),
    Indicator("solar data cent", 4.0, "datacenter"),
    Indicator("solar powered comput", 4.0, "datacenter"),
    Indicator("wind power data cent", 4.0, "datacenter"),
    Indicator("green hydrogen data cent", 4.0, "datacenter"),
    Indicator("data centre.*energy", 2.0, "datacenter"),
    Indicator("data centre.*cool", 2.0, "datacenter"),
    Indicator("data centre.*green", 2.0, "datacenter"),

    # === E-waste / economie circulaire ===
    Indicator("e-waste", 4.0, "ewaste"),
    Indicator("electronic waste", 4.0, "ewaste"),
    Indicator("circular economy electronic", 4.0, "ewaste"),
    Indicator("circular economy tech", 3.0, "ewaste"),
    Indicator("right to repair", 3.0, "ewaste"),
    Indicator("refurbished it", 3.0, "ewaste"),
    Indicator("recycl.*electronic", 3.0, "ewaste"),
    Indicator("sustainable hardware", 3.0, "ewaste"),

    # === Signaux contextuels (titres courants) ===
    Indicator("green tech", 2.0, "moderate"),
    Indicator("green loan.*data cent", 3.0, "moderate"),
    Indicator("green energy data cent", 3.0, "moderate"),
    Indicator("green energy server", 3.0, "moderate"),
    Indicator("clean tech", 1.5, "moderate"),
    Indicator("cleantech", 1.5, "moderate"),
    Indicator("climate tech", 2.0, "moderate"),
    Indicator("energy monitor", 2.0, "moderate"),

    # === Signaux moderes (besoin de contexte) ===
    Indicator("environmental impact.*technolog", 2.5, "moderate"),
    Indicator("environmental impact.*digital", 2.5, "moderate"),
    Indicator("environmental impact.*ai", 2.5, "moderate"),
    Indicator("environmental impact.*computing", 2.5, "moderate"),
    Indicator("digital pollution", 3.0, "moderate"),
    Indicator("digital carbon", 3.0, "moderate"),
    Indicator("ict sustainability", 3.0, "moderate"),
    Indicator("climate.*software", 1.5, "moderate"),
    Indicator("sustainable digitali", 2.5, "moderate"),

    # === Signaux faibles elargis (capturer plus large en etage 1) ===
    # Ces indicateurs a petit poids ne suffisent pas seuls a declencher une
    # candidature, mais ils s'additionnent avec d'autres signaux pour passer
    # le seuil permissif CANDIDATE_MIN_GREEN_SCORE.
    Indicator("power consumption", 1.0, "weak"),
    Indicator("power efficiency", 1.0, "weak"),
    Indicator("energy cost", 1.0, "weak"),
    Indicator("kwh", 1.0, "weak"),
    Indicator("megawatt", 1.0, "weak"),
    Indicator("gigawatt", 1.5, "weak"),
    Indicator("co2", 1.0, "weak"),
    Indicator("greenhouse gas", 1.5, "weak"),
    Indicator("sustainability report", 1.5, "weak"),
    Indicator("environmental cost", 1.5, "weak"),
    Indicator("ecological", 1.0, "weak"),
    Indicator("renewable", 1.0, "weak"),
    Indicator("photovoltaic", 1.0, "weak"),
    Indicator("heat recovery", 1.5, "weak"),
    Indicator("thermal efficiency", 1.5, "weak"),
    Indicator("low latency energy", 1.5, "weak"),
    Indicator("compress.*model.*energy", 2.0, "weak"),
    Indicator("pruning.*energy", 2.0, "weak"),
    Indicator("quantization.*energy", 2.0, "weak"),
    Indicator("model compression", 1.0, "weak"),
    Indicator("inference cost", 1.5, "weak"),
    Indicator("training cost", 1.0, "weak"),
    Indicator("gpu hours", 1.0, "weak"),
    Indicator("server farm", 1.5, "weak"),
    Indicator("edge computing", 1.0, "weak"),
    Indicator("tinyml", 2.0, "weak"),
    Indicator("on-device inference", 1.5, "weak"),
    Indicator("hardware accelerator", 1.0, "weak"),
    Indicator("heat dissipation", 1.5, "weak"),
    Indicator("datacenter", 1.0, "weak"),
    Indicator("data center", 1.0, "weak"),
    Indicator("data-center", 1.0, "weak"),
]

# --- INDICATEURS NON GREEN IT (negatifs) ---
# On reserve les poids tres forts (>=4) aux signaux qui justifient
# un rejet direct (passage en NON_GREEN sans appel LLM).

NON_GREEN_INDICATORS: list[Indicator] = [
    # === IA/ML generale (signal faible, ne doit pas exclure seul) ===
    Indicator("large language model", 1.0, "general_ai"),
    Indicator("transformer architecture", 1.0, "general_ai"),
    Indicator("reinforcement learning", 1.0, "general_ai"),
    Indicator("neural network architecture", 1.0, "general_ai"),
    Indicator("attention mechanism", 1.0, "general_ai"),
    Indicator("benchmark accuracy", 1.5, "general_ai"),
    Indicator("state-of-the-art", 1.0, "general_ai"),

    # === Cybersecurite (signal moyen, presque jamais Green IT) ===
    Indicator("cybersecurity", 3.0, "cyber"),
    Indicator("ransomware", 4.0, "cyber"),
    Indicator("malware", 4.0, "cyber"),
    Indicator("phishing", 4.0, "cyber"),
    Indicator("data breach", 3.0, "cyber"),
    Indicator("zero trust", 3.0, "cyber"),
    Indicator("vulnerability", 2.0, "cyber"),

    # === Crypto / finance (signal fort = rejet dur) ===
    Indicator("cryptocurrency", 5.0, "crypto"),
    Indicator("bitcoin", 5.0, "crypto"),
    Indicator("ethereum", 5.0, "crypto"),
    Indicator("dogecoin", 5.0, "crypto"),
    Indicator("nft", 4.0, "crypto"),
    Indicator("decentralized finance", 4.0, "crypto"),
    Indicator("stablecoin", 5.0, "crypto"),
    Indicator("blockchain", 2.0, "crypto"),
    Indicator("stock market", 4.0, "finance"),
    Indicator("trading algorithm", 3.0, "finance"),
    Indicator("fintech", 2.0, "finance"),
    Indicator("fraud detection", 2.0, "finance"),

    # === Divertissement / consumer ===
    Indicator("video game", 5.0, "entertainment"),
    Indicator("gaming", 3.0, "entertainment"),
    Indicator("esport", 4.0, "entertainment"),
    Indicator("virtual reality", 2.5, "entertainment"),
    Indicator("augmented reality", 2.5, "entertainment"),
    Indicator("metaverse", 4.0, "entertainment"),
    Indicator("streaming platform", 3.0, "entertainment"),
    Indicator("social media", 2.0, "entertainment"),
    Indicator("tiktok", 5.0, "entertainment"),
    Indicator("instagram", 4.0, "entertainment"),
    Indicator("smartphone", 2.0, "consumer"),
    Indicator("iphone", 3.0, "consumer"),
    Indicator("wearable", 2.0, "consumer"),

    # === Domaines specifiques ===
    Indicator("healthcare.*ai", 2.0, "domain"),
    Indicator("medical.*ai", 2.0, "domain"),
    Indicator("drug discovery", 3.0, "domain"),
    Indicator("autonomous vehicle", 2.0, "domain"),
    Indicator("self-driving", 2.0, "domain"),
    Indicator("quantum computing", 2.0, "domain"),
    Indicator("quantum bit", 3.0, "domain"),
    Indicator("space.*satellite", 2.0, "domain"),
    Indicator("robotics", 1.5, "domain"),

    # === Business / marche / finance (signal fort = rejet dur) ===
    Indicator("market research report", 5.0, "business"),
    Indicator("market size.*billion", 5.0, "business"),
    Indicator("market forecast", 5.0, "business"),
    Indicator("business opportunit", 3.0, "business"),
    Indicator("venture capital", 2.0, "business"),
    Indicator("funding round", 2.0, "business"),
    Indicator("ipo", 2.0, "business"),
    Indicator("epidemiology.*pipeline", 5.0, "business"),
    Indicator("market disruptor", 4.0, "business"),

    # === Bourse / investissement (rejet dur) ===
    Indicator("stock.*watchlist", 6.0, "finance"),
    Indicator("stocks to.*watch", 6.0, "finance"),
    Indicator("stocks to.*add", 6.0, "finance"),
    Indicator("stocks to.*follow", 6.0, "finance"),
    Indicator("stocks to.*buy", 6.0, "finance"),
    Indicator("investor.*stock", 4.0, "finance"),
    Indicator("stock.*investor", 4.0, "finance"),
    Indicator("good bet.*investor", 4.0, "finance"),
    Indicator("share price", 4.0, "finance"),
    Indicator("stock exchange", 4.0, "finance"),
    Indicator("remain in focus", 3.0, "finance"),
    Indicator("price prediction", 4.0, "finance"),
    Indicator("hype.*elon musk", 5.0, "finance"),

    # === Communaute / nuisance (pas de Green IT) ===
    Indicator("resident.*complain", 3.0, "community"),
    Indicator("resident.*choke", 3.0, "community"),
    Indicator("noise.*data cent", 3.0, "community"),
    Indicator("dust.*data cent", 3.0, "community"),
    Indicator("village.*data cent", 3.0, "community"),
    Indicator("what.*become.*village", 4.0, "community"),
]


# Seuils de decision du pre-filtre binaire.
# On est volontairement permissif : mieux vaut envoyer un article douteux
# au LLM que de rater un vrai Green IT au pre-filtre. Le LLM corrigera
# les faux positifs a l'etage 2.
NON_GREEN_SCORE_THRESHOLD = 12.0  # Signal anti-Green tres fort necessaire pour exclure
CANDIDATE_MIN_GREEN_SCORE = 0.1  # Tout signal Green, meme infime, declenche le LLM


# =============================================================================
# MOTEUR DE CLASSIFICATION
# =============================================================================


class PrefilterStatus(StrEnum):
    """Statut de sortie du pre-filtre mots-cles (etage 1)."""

    NON_GREEN = "NON_GREEN"
    CANDIDATE = "CANDIDATE"


@dataclass(frozen=True)
class PrefilterResult:
    """Resultat du pre-filtre pour un article."""

    status: PrefilterStatus
    green_score: float
    non_green_score: float
    reason: str
    matched_green: list[str]
    matched_non_green: list[str]


def _normalize_text(text: str) -> str:
    """Normalise un texte pour le matching d'indicateurs.

    Remplace les tirets par des espaces, supprime la ponctuation parasite
    et normalise les espaces multiples pour que `energy-efficient` matche
    aussi `energy efficient`.
    """
    text = text.lower().strip()
    text = text.replace("-", " ").replace("\u2013", " ").replace("\u2014", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def _score_text(
    text: str,
    indicators: list[Indicator],
    *,
    title_mode: bool = False,
) -> tuple[float, list[str]]:
    """Calcule le score d'un texte contre une liste d'indicateurs.

    Args:
        text: Texte deja normalise via `_normalize_text`.
        indicators: Liste d'indicateurs a chercher.
        title_mode: Si True, applique un multiplicateur x2.5 (titre > contenu).

    Returns:
        Tuple (score total, liste des indicateurs matches).
    """
    score = 0.0
    matched: list[str] = []
    multiplier = 2.5 if title_mode else 1.0

    for ind in indicators:
        if ".*" in ind.pattern:
            if re.search(ind.pattern, text):
                score += ind.weight * multiplier
                matched.append(f"{ind.pattern}({ind.category})")
        elif ind.pattern in text:
            score += ind.weight * multiplier
            matched.append(f"{ind.pattern}({ind.category})")

    return score, matched


def prefilter_article(
    titre: str,
    contenu: str,
    source_nom: str = "",
) -> PrefilterResult:
    """Applique le pre-filtre binaire (etage 1) sur un article.

    Args:
        titre: Titre de l'article.
        contenu: Contenu complet (peut etre tronque en amont).
        source_nom: Nom de la source pour ajustements contextuels.

    Returns:
        Verdict du pre-filtre : `NON_GREEN` (rejet direct) ou `CANDIDATE`
        (a faire verifier par le LLM).
    """
    titre_norm = _normalize_text(titre or "")
    contenu_norm = _normalize_text(contenu or "")

    # Scoring titre (x2.5) et contenu (x1)
    green_title, green_title_matches = _score_text(
        titre_norm, GREEN_INDICATORS, title_mode=True
    )
    non_green_title, non_green_title_matches = _score_text(
        titre_norm, NON_GREEN_INDICATORS, title_mode=True
    )
    green_content, green_content_matches = _score_text(contenu_norm, GREEN_INDICATORS)
    non_green_content, non_green_content_matches = _score_text(
        contenu_norm, NON_GREEN_INDICATORS
    )

    green_score = green_title + green_content
    non_green_score = non_green_title + non_green_content

    all_green = green_title_matches + green_content_matches
    all_non_green = non_green_title_matches + non_green_content_matches

    # Ajustement : projets/produits commencant par "green" (GreenNLP, GreenAI, ...)
    if re.search(r"\bgreen\w+\b", titre_norm):
        green_score += 3.0
        all_green.append("green_prefix_project(core)")

    # arXiv : on NE penalise plus systematiquement les papiers de recherche.
    # Beaucoup d'articles arXiv portent reellement sur le Green IT (IA frugale,
    # quantization visant l'energie, etc.). En cas de signal energetique dans
    # le contenu, on renforce le score pour que le LLM juge en derniere instance.
    if "arxiv" in source_nom.lower():
        # Signal de presence concrete de vocabulaire energetique/carbone
        energy_terms = (
            "energy",
            "carbon",
            "emission",
            "electricity",
            "watt",
            "kwh",
            "power consumption",
            "environmental",
            "co2",
            "sustainability",
            "renewable",
            "efficient",
        )
        energy_count = sum(1 for t in energy_terms if t in contenu_norm)
        if energy_count >= 2:
            green_score += 2.0  # Bonus plus genereux pour laisser passer au LLM
        elif energy_count == 1:
            green_score += 0.5

    # Rapports boursiers / market reports -> rejet dur
    market_markers = (
        "market report",
        "market size",
        "billion",
        "market research",
        "epidemiology",
        "market forecast",
        "watchlist",
        "stocks to",
        "price prediction",
        "good bet for investors",
    )
    if any(t in titre_norm for t in market_markers):
        non_green_score += 5.0

    # Decision binaire (permissive cote CANDIDATE)
    # - Si signal Green a un minimum present ET pas de domination ecrasante des
    #   indicateurs anti-Green, on envoie au LLM pour decider.
    # - Sinon : NON_GREEN direct, sans appel LLM.
    if green_score >= CANDIDATE_MIN_GREEN_SCORE and non_green_score < (
        green_score + NON_GREEN_SCORE_THRESHOLD
    ):
        status = PrefilterStatus.CANDIDATE
        reason = (
            f"Candidate (green={green_score:.1f}, non_green={non_green_score:.1f}) "
            "-> verification LLM requise"
        )
    else:
        status = PrefilterStatus.NON_GREEN
        if green_score == 0:
            reason = "Aucun signal Green IT detecte"
        elif non_green_score >= (green_score + NON_GREEN_SCORE_THRESHOLD):
            reason = (
                f"Signal anti-Green domine (green={green_score:.1f}, "
                f"non_green={non_green_score:.1f})"
            )
        else:
            reason = (
                f"Signal Green trop faible (green={green_score:.1f} < "
                f"{CANDIDATE_MIN_GREEN_SCORE})"
            )

    return PrefilterResult(
        status=status,
        green_score=round(green_score, 2),
        non_green_score=round(non_green_score, 2),
        reason=reason,
        matched_green=sorted(set(all_green)),
        matched_non_green=sorted(set(all_non_green)),
    )


# =============================================================================
# ORCHESTRATION
# =============================================================================


async def run_prefilter(*, only_new: bool = True) -> dict[str, int]:
    """Applique le pre-filtre sur les articles de la base.

    Par defaut, ne traite que les articles pas encore classifies
    (`modele_classification IS NULL`). Utiliser `only_new=False` pour
    tout retraiter (ex : apres modification des regles de scoring).

    Args:
        only_new: Si True, ignore les articles deja classifies.

    Returns:
        Statistiques {total, candidates, non_green}.
    """
    stats = {"total": 0, "candidates": 0, "non_green": 0}

    async with async_session_factory() as session:
        stmt = (
            select(Article, Source.nom)
            .outerjoin(Source, Article.id_source == Source.id_source)
            .where(Article.contenu.isnot(None))
            .order_by(Article.id_article)
        )
        if only_new:
            stmt = stmt.where(
                or_(
                    Article.modele_classification.is_(None),
                    # Permet la reprise si un precedent run s'est interrompu
                    and_(
                        Article.modele_classification == "keyword_filter",
                        Article.est_green_it.is_(None),
                    ),
                )
            )
        result = await session.execute(stmt)
        rows = result.all()

        total = len(rows)
        if total == 0:
            logger.info("Aucun article a pre-filtrer")
            return stats

        logger.info(f"{total} articles a pre-filtrer (only_new={only_new})")

        candidate_ids: list[int] = []
        non_green_ids: list[int] = []

        for article, source_nom in rows:
            verdict = prefilter_article(
                titre=article.titre,
                contenu=article.contenu or "",
                source_nom=source_nom or "",
            )
            stats["total"] += 1
            if verdict.status is PrefilterStatus.CANDIDATE:
                stats["candidates"] += 1
                candidate_ids.append(article.id_article)
            else:
                stats["non_green"] += 1
                non_green_ids.append(article.id_article)

        # Mise a jour bulk en base : NON_GREEN -> est_green_it=False,
        # CANDIDATE -> est_green_it=NULL, dans les deux cas
        # modele_classification="keyword_filter".
        if non_green_ids:
            await session.execute(
                update(Article)
                .where(Article.id_article.in_(non_green_ids))
                .values(
                    est_green_it=False,
                    modele_classification="keyword_filter",
                )
            )
        if candidate_ids:
            await session.execute(
                update(Article)
                .where(Article.id_article.in_(candidate_ids))
                .values(
                    est_green_it=None,
                    modele_classification="keyword_filter",
                )
            )
        await session.commit()

    pct_cand = (stats["candidates"] / stats["total"] * 100) if stats["total"] else 0
    logger.info("Pre-filtre termine :")
    logger.info(f"  Total traite     : {stats['total']}")
    logger.info(f"  Candidates (LLM) : {stats['candidates']} ({pct_cand:.1f}%)")
    logger.info(f"  NON_GREEN direct : {stats['non_green']}")
    return stats


if __name__ == "__main__":
    from greentech.utils.logger import setup_logging

    setup_logging(level="INFO", enable_loki=False)
    asyncio.run(run_prefilter())
