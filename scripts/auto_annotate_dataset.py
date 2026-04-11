"""Annotation automatique du Golden Dataset par scoring multi-critères.

Classifie chaque article comme Green IT (1) ou Non Green IT (0)
en combinant plusieurs signaux pondérés :
  - Indicateurs positifs (Green IT) dans le titre et le contenu
  - Indicateurs négatifs (Non Green IT) dans le titre et le contenu
  - Règles contextuelles par source (arXiv vs actualités)
  - Score de confiance pour identifier les cas ambigus

La définition de "Green IT" utilisée :
  Un article est Green IT si son SUJET PRINCIPAL concerne la réduction
  de l'impact environnemental des technologies numériques, OU l'usage
  du numérique au service de la transition écologique.

  Cela inclut : efficacité énergétique des infrastructures IT, empreinte
  carbone du numérique, sobriété numérique, éco-conception logicielle,
  e-waste, énergies renouvelables pour les data centers, IA frugale.

  Cela exclut : recherche IA/ML générale (même si "efficient"),
  véhicules électriques (sauf angle IT), énergie renouvelable générale
  (sans lien avec l'IT), rapports de marché financier, tech grand public.

"""

from __future__ import annotations

import asyncio
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from sqlalchemy import select

from greentech.config import BASE_DIR
from greentech.data.storage.database import async_session_factory
from greentech.data.storage.models import Article, Source

OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "golden_dataset.csv"

CSV_HEADERS = [
    "id_article",
    "titre",
    "url",
    "contenu_extrait",
    "source_nom",
    "date_publication",
    "label_green_it",
    "score_confiance",
    "raison",
]

EXTRAIT_MAX_CHARS = 500


# =============================================================================
# INDICATEURS PONDÉRÉS
# =============================================================================

@dataclass(frozen=True)
class Indicator:
    """Un indicateur de classification avec son poids et sa catégorie."""
    pattern: str
    weight: float
    category: str


# --- INDICATEURS GREEN IT (positifs) ---
# Poids : 5 = signal très fort, 3 = signal moyen, 1 = signal faible

GREEN_INDICATORS: list[Indicator] = [
    # === Efficacité énergétique IT (core Green IT) ===
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
    Indicator("sobriété numérique", 5.0, "core"),
    Indicator("frugal ai", 4.0, "core"),
    Indicator("frugal computing", 4.0, "core"),

    # === Empreinte carbone du numérique ===
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
    Indicator("net-zero technology", 3.0, "carbon"),
    Indicator("low carbon computing", 4.0, "carbon"),
    Indicator("codecarbon", 4.0, "carbon"),
    Indicator("carbon tracker", 3.0, "carbon"),

    # === Efficacité énergétique (contexte IT) ===
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

    # === E-waste / économie circulaire ===
    Indicator("e-waste", 4.0, "ewaste"),
    Indicator("electronic waste", 4.0, "ewaste"),
    Indicator("circular economy electronic", 4.0, "ewaste"),
    Indicator("circular economy tech", 3.0, "ewaste"),
    Indicator("right to repair", 3.0, "ewaste"),
    Indicator("refurbished it", 3.0, "ewaste"),
    Indicator("recycl.*electronic", 3.0, "ewaste"),
    Indicator("sustainable hardware", 3.0, "ewaste"),

    # === Signaux contextuels (titres courants) ===
    Indicator("green tech", 3.0, "moderate"),
    Indicator("green loan.*data cent", 3.0, "moderate"),
    Indicator("green energy data cent", 3.0, "moderate"),
    Indicator("green energy server", 3.0, "moderate"),
    Indicator("clean tech", 2.0, "moderate"),
    Indicator("cleantech", 2.0, "moderate"),
    Indicator("climate tech", 3.0, "moderate"),
    Indicator("energy monitor", 3.0, "moderate"),
    Indicator("energy saving", 3.0, "energy"),
    Indicator("energy reduction", 3.0, "energy"),
    Indicator("heat and energy", 2.0, "energy"),
    Indicator("cuts heat", 2.0, "energy"),
    Indicator("reduce heat", 2.0, "energy"),
    Indicator("reduce power", 2.0, "energy"),
    Indicator("solar powered", 3.0, "datacenter"),
    Indicator("wind powered", 3.0, "datacenter"),
    Indicator("data centre.*energy", 2.0, "datacenter"),
    Indicator("data centre.*cool", 2.0, "datacenter"),
    Indicator("data centre.*green", 2.0, "datacenter"),

    # === Signaux modérés (besoin de contexte) ===
    Indicator("environmental impact.*technolog", 3.0, "moderate"),
    Indicator("environmental impact.*digital", 3.0, "moderate"),
    Indicator("environmental impact.*ai", 3.0, "moderate"),
    Indicator("environmental impact.*computing", 3.0, "moderate"),
    Indicator("digital pollution", 3.0, "moderate"),
    Indicator("digital carbon", 3.0, "moderate"),
    Indicator("ict sustainability", 3.0, "moderate"),
    Indicator("climate.*software", 2.0, "moderate"),
    Indicator("sustainable digitali", 3.0, "moderate"),
    Indicator("carbon emission", 3.0, "carbon"),
    Indicator("carbon reduction", 3.0, "carbon"),
    Indicator("reduce.*carbon", 2.0, "carbon"),
    Indicator("net zero", 2.0, "carbon"),
]

# --- INDICATEURS NON GREEN IT (négatifs) ---
NON_GREEN_INDICATORS: list[Indicator] = [
    # === IA/ML générale ===
    Indicator("large language model", 2.0, "general_ai"),
    Indicator("transformer architecture", 2.0, "general_ai"),
    Indicator("reinforcement learning", 2.0, "general_ai"),
    Indicator("neural network architecture", 2.0, "general_ai"),
    Indicator("attention mechanism", 2.0, "general_ai"),
    Indicator("benchmark accuracy", 2.0, "general_ai"),
    Indicator("state-of-the-art", 1.5, "general_ai"),
    Indicator("fine-tuning", 1.0, "general_ai"),
    Indicator("pre-training", 1.0, "general_ai"),

    # === Cybersécurité ===
    Indicator("cybersecurity", 3.0, "cyber"),
    Indicator("ransomware", 4.0, "cyber"),
    Indicator("malware", 4.0, "cyber"),
    Indicator("phishing", 4.0, "cyber"),
    Indicator("data breach", 3.0, "cyber"),
    Indicator("zero trust", 3.0, "cyber"),
    Indicator("vulnerability", 2.0, "cyber"),

    # === Crypto / finance ===
    Indicator("cryptocurrency", 4.0, "crypto"),
    Indicator("bitcoin", 4.0, "crypto"),
    Indicator("ethereum", 4.0, "crypto"),
    Indicator("nft", 3.0, "crypto"),
    Indicator("decentralized finance", 3.0, "crypto"),
    Indicator("stock market", 3.0, "finance"),
    Indicator("trading algorithm", 3.0, "finance"),
    Indicator("fintech", 2.0, "finance"),
    Indicator("fraud detection", 2.0, "finance"),

    # === Divertissement / consumer ===
    Indicator("video game", 4.0, "entertainment"),
    Indicator("gaming", 2.0, "entertainment"),
    Indicator("esport", 3.0, "entertainment"),
    Indicator("virtual reality", 2.0, "entertainment"),
    Indicator("augmented reality", 2.0, "entertainment"),
    Indicator("metaverse", 3.0, "entertainment"),
    Indicator("streaming platform", 3.0, "entertainment"),
    Indicator("social media", 2.0, "entertainment"),
    Indicator("tiktok", 4.0, "entertainment"),
    Indicator("instagram", 3.0, "entertainment"),
    Indicator("smartphone", 2.0, "consumer"),
    Indicator("iphone", 3.0, "consumer"),
    Indicator("wearable", 2.0, "consumer"),

    # === Domaines spécifiques ===
    Indicator("healthcare.*ai", 2.0, "domain"),
    Indicator("medical.*ai", 2.0, "domain"),
    Indicator("drug discovery", 3.0, "domain"),
    Indicator("autonomous vehicle", 2.0, "domain"),
    Indicator("self-driving", 2.0, "domain"),
    Indicator("quantum computing", 2.0, "domain"),
    Indicator("quantum bit", 3.0, "domain"),
    Indicator("space.*satellite", 2.0, "domain"),
    Indicator("robotics", 1.5, "domain"),

    # === Business / marché / finance (bruit) ===
    Indicator("market research report", 4.0, "business"),
    Indicator("market size.*billion", 3.0, "business"),
    Indicator("market forecast", 4.0, "business"),
    Indicator("business opportunit", 3.0, "business"),
    Indicator("venture capital", 2.0, "business"),
    Indicator("funding round", 2.0, "business"),
    Indicator("ipo", 2.0, "business"),
    Indicator("epidemiology.*pipeline", 4.0, "business"),

    # === Bourse / investissement ===
    Indicator("stock.*watchlist", 5.0, "finance"),
    Indicator("stocks to.*watch", 5.0, "finance"),
    Indicator("stocks to.*add", 5.0, "finance"),
    Indicator("stocks to.*follow", 5.0, "finance"),
    Indicator("stocks to.*buy", 5.0, "finance"),
    Indicator("investor.*stock", 3.0, "finance"),
    Indicator("stock.*investor", 3.0, "finance"),
    Indicator("good bet.*investor", 3.0, "finance"),
    Indicator("share price", 3.0, "finance"),
    Indicator("stock exchange", 3.0, "finance"),
    Indicator("remain in focus", 3.0, "finance"),

    # === Communauté / nuisance (pas de Green IT) ===
    Indicator("resident.*complain", 3.0, "community"),
    Indicator("resident.*choke", 3.0, "community"),
    Indicator("noise.*data cent", 3.0, "community"),
    Indicator("dust.*data cent", 3.0, "community"),
    Indicator("village.*data cent", 3.0, "community"),
    Indicator("what.*become.*village", 4.0, "community"),
]


# =============================================================================
# MOTEUR DE CLASSIFICATION
# =============================================================================


@dataclass
class ClassificationResult:
    """Résultat de classification d'un article."""
    label: int  # 1 = Green IT, 0 = Non Green IT
    confidence: float  # 0.0 à 1.0
    green_score: float
    non_green_score: float
    reason: str  # Explication courte
    matched_green: list[str]
    matched_non_green: list[str]


def _normalize_text(text: str) -> str:
    """Normalise un texte pour le matching d'indicateurs.

    Remplace les tirets par des espaces, supprime la ponctuation
    et normalise les espaces multiples pour que "energy-efficient"
    matche "energy efficient".
    """
    text = text.lower().strip()
    # Tirets → espaces (energy-efficient → energy efficient)
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    # Supprimer apostrophes typographiques
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # Normaliser les espaces multiples
    text = re.sub(r"\s+", " ", text)
    return text


def _score_text(
    text: str,
    indicators: list[Indicator],
    title_mode: bool = False,
) -> tuple[float, list[str]]:
    """Calcule le score d'un texte contre une liste d'indicateurs.

    Args:
        text: Texte déjà normalisé via _normalize_text().
        indicators: Liste d'indicateurs à chercher.
        title_mode: Si True, applique un multiplicateur x2.5 (titre > contenu).

    Returns:
        Tuple (score, liste des indicateurs matchés).
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


def classify_article(
    titre: str,
    contenu: str,
    source_nom: str = "",
) -> ClassificationResult:
    """Classifie un article avec scoring multi-critères.

    Le processus :
    1. Score le titre (poids x2.5) et le contenu (poids x1) séparément
    2. Applique des ajustements contextuels par source
    3. Calcule un score net et une confiance

    Args:
        titre: Titre de l'article.
        contenu: Contenu complet.
        source_nom: Nom de la source (pour ajustements contextuels).

    Returns:
        Résultat détaillé de la classification.
    """
    titre_norm = _normalize_text(titre)
    contenu_norm = _normalize_text(contenu or "")

    # --- Scoring du titre ---
    green_title, green_title_matches = _score_text(
        titre_norm, GREEN_INDICATORS, title_mode=True
    )
    non_green_title, non_green_title_matches = _score_text(
        titre_norm, NON_GREEN_INDICATORS, title_mode=True
    )

    # --- Scoring du contenu ---
    green_content, green_content_matches = _score_text(
        contenu_norm, GREEN_INDICATORS, title_mode=False
    )
    non_green_content, non_green_content_matches = _score_text(
        contenu_norm, NON_GREEN_INDICATORS, title_mode=False
    )

    green_score = green_title + green_content
    non_green_score = non_green_title + non_green_content

    all_green = green_title_matches + green_content_matches
    all_non_green = non_green_title_matches + non_green_content_matches

    # --- Ajustements contextuels ---

    # Titres de projets/papiers commençant par "green" (GreenNLP, GreenAI, etc.)
    if re.search(r"\bgreen\w+\b", titre_norm):
        green_score += 5.0
        all_green.append("green_prefix_project(core)")

    # arXiv : les articles de recherche IA sont majoritairement Non Green IT
    # sauf si un signal Green fort est détecté
    if "arxiv" in source_nom.lower():
        # Pénalité de base pour arXiv (biais vers Non Green IT)
        non_green_score += 3.0

        # Mots qui indiquent de l'efficacité algorithmique (pas énergétique)
        algo_efficiency_terms = [
            "efficient algorithm", "efficient method", "efficient approach",
            "computationally efficient", "sample efficient", "parameter efficient",
            "communication efficient", "efficient training",
            "efficient representation", "efficient learning",
        ]
        for term in algo_efficiency_terms:
            if term in titre_norm or term in contenu_norm:
                # Ces termes parlent d'efficacité computationnelle, pas d'énergie
                # Réduire le score Green si on les a matchés par erreur
                non_green_score += 1.5
                break

        # Si le contenu mentionne spécifiquement l'énergie/carbone -> annuler la pénalité
        energy_terms = ["energy", "carbon", "emission", "electricity", "watt",
                        "kwh", "power consumption", "environmental"]
        energy_count = sum(1 for t in energy_terms if t in contenu_norm)
        if energy_count >= 2:
            green_score += 3.0  # Compensation : vraiment un sujet Green

    # Rapports de marché et articles boursiers : rarement du vrai Green IT
    if any(t in titre_norm for t in ["market report", "market size", "billion",
                                       "market research", "epidemiology",
                                       "market forecast", "watchlist",
                                       "stocks to", "stock", "investor"]):
        non_green_score += 5.0

    # --- Décision finale ---
    net_score = green_score - non_green_score

    if net_score >= 5.0:
        label = 1
        reason = f"Fort signal Green IT (net={net_score:.1f})"
    elif net_score >= 3.0:
        label = 1
        reason = f"Signal Green IT modéré (net={net_score:.1f})"
    elif net_score <= -5.0:
        label = 0
        reason = f"Fort signal Non Green IT (net={net_score:.1f})"
    elif net_score <= -1.0:
        label = 0
        reason = f"Signal Non Green IT modéré (net={net_score:.1f})"
    elif green_score > 0 and non_green_score == 0:
        label = 1
        reason = f"Signal Green IT uniquement (green={green_score:.1f})"
    else:
        label = 0
        reason = f"Pas de signal Green IT clair (net={net_score:.1f})"

    # Confiance : basée sur l'écart entre les scores
    total = green_score + non_green_score
    if total > 0:
        if label == 1:
            confidence = min(green_score / total, 1.0)
        else:
            confidence = min(non_green_score / total, 1.0)
    else:
        confidence = 0.5  # Aucun indicateur trouvé

    # Ajuster la confiance par la magnitude
    if abs(net_score) >= 10:
        confidence = min(confidence + 0.2, 1.0)
    elif abs(net_score) <= 2:
        confidence = max(confidence - 0.15, 0.3)

    return ClassificationResult(
        label=label,
        confidence=round(confidence, 3),
        green_score=round(green_score, 2),
        non_green_score=round(non_green_score, 2),
        reason=reason,
        matched_green=list(set(all_green)),
        matched_non_green=list(set(all_non_green)),
    )


# =============================================================================
# ORCHESTRATION
# =============================================================================


async def auto_annotate() -> dict[str, int]:
    """Annote tous les articles avec le scoring multi-critères.

    Returns:
        Statistiques {total, green, non_green, high_confidence, low_confidence}.
    """
    stats = {
        "total": 0, "green": 0, "non_green": 0,
        "high_confidence": 0, "low_confidence": 0,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    async with async_session_factory() as session:
        stmt = (
            select(Article, Source.nom)
            .outerjoin(Source, Article.id_source == Source.id_source)
            .where(Article.contenu.isnot(None))
            .order_by(Article.id_article)
        )
        result = await session.execute(stmt)
        rows = result.all()

    logger.info(f"{len(rows)} articles avec contenu à annoter")

    csv_rows: list[dict[str, str]] = []
    for article, source_nom in rows:
        cr = classify_article(
            titre=article.titre,
            contenu=article.contenu or "",
            source_nom=source_nom or "",
        )

        extrait = (article.contenu or "")[:EXTRAIT_MAX_CHARS]
        if len(article.contenu or "") > EXTRAIT_MAX_CHARS:
            extrait += "..."

        csv_rows.append({
            "id_article": str(article.id_article),
            "titre": article.titre,
            "url": article.url,
            "contenu_extrait": extrait,
            "source_nom": source_nom or "Inconnue",
            "date_publication": (
                article.date_publication.isoformat() if article.date_publication else ""
            ),
            "label_green_it": str(cr.label),
            "score_confiance": str(cr.confidence),
            "raison": cr.reason,
        })

        stats["total"] += 1
        if cr.label == 1:
            stats["green"] += 1
        else:
            stats["non_green"] += 1
        if cr.confidence >= 0.7:
            stats["high_confidence"] += 1
        else:
            stats["low_confidence"] += 1

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(csv_rows)

    g_pct = stats["green"] / stats["total"] * 100 if stats["total"] > 0 else 0
    hc_pct = stats["high_confidence"] / stats["total"] * 100 if stats["total"] > 0 else 0
    logger.info(f"Annotation terminée : {OUTPUT_FILE}")
    logger.info(
        f"Total: {stats['total']} | "
        f"Green IT: {stats['green']} ({g_pct:.1f}%) | "
        f"Non Green IT: {stats['non_green']} ({100 - g_pct:.1f}%)"
    )
    logger.info(
        f"Confiance haute (>=0.7): {stats['high_confidence']} ({hc_pct:.1f}%) | "
        f"Confiance basse (<0.7): {stats['low_confidence']}"
    )
    logger.info(
        "IMPORTANT : Vérifier manuellement les cas à faible confiance "
        "avant d'utiliser pour l'entraînement."
    )
    return stats


if __name__ == "__main__":
    asyncio.run(auto_annotate())
