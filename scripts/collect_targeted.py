"""Collecte ciblée avec les 120 crédits API restants.

Utilise des requêtes très spécifiques pour maximiser la qualité et la
diversité des articles collectés, avec un délai entre chaque requête
pour respecter les limites de l'API NewsData.io (free tier).

60 requêtes Green IT + 60 requêtes Non Green IT = dataset équilibré.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from greentech.config import get_settings
from greentech.data.storage.minio_client import (
    generate_raw_path,
    upload_json_to_minio,
)

NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"
HTTP_TIMEOUT = 30.0
MAX_RESULTS = 10

# Délai entre les requêtes (secondes) pour respecter le rate limit
REQUEST_DELAY = 3.0

# === MOTS-CLÉS GREEN IT (60) ===
# Articles dont le SUJET PRINCIPAL est la durabilité environnementale
# dans le secteur technologique / numérique.
GREEN_IT_QUERIES: list[str] = [
    # --- Efficacité énergétique des data centers ---
    "data center energy efficiency",
    "data center cooling technology",
    "data center renewable energy",
    "data center carbon neutral",
    "liquid cooling server",
    "immersion cooling data center",
    "power usage effectiveness PUE",
    # --- Green IT / sustainable computing ---
    "green IT strategy company",
    "green computing initiative",
    "green software engineering",
    "sustainable software development",
    "eco-design software",
    "digital sobriety",
    "green coding practice",
    "sustainable web design",
    # --- Empreinte carbone du numérique ---
    "carbon footprint artificial intelligence",
    "carbon footprint cloud computing",
    "carbon footprint machine learning training",
    "carbon emissions data center",
    "carbon aware computing",
    "net zero technology company",
    "carbon offset technology",
    # --- IA éco-responsable ---
    "green AI research",
    "energy efficient neural network",
    "sustainable machine learning",
    "low power deep learning",
    "efficient AI model carbon",
    "reduce AI energy consumption",
    "carbon cost large language model",
    # --- Hardware durable ---
    "e-waste recycling technology",
    "circular economy electronics",
    "sustainable hardware design",
    "refurbished IT equipment",
    "right to repair electronics",
    "eco-friendly semiconductor",
    # --- Énergie renouvelable + tech ---
    "solar powered data center",
    "wind energy technology infrastructure",
    "renewable energy cloud provider",
    "green hydrogen data center",
    "clean energy artificial intelligence",
    # --- Réglementation Green IT ---
    "EU digital sustainability regulation",
    "green digital certification",
    "environmental reporting technology",
    "ESG technology compliance",
    "corporate sustainability digital",
    # --- Cloud & infra verte ---
    "green cloud migration",
    "sustainable cloud provider",
    "serverless computing environmental",
    "edge computing energy savings",
    # --- Métriques & monitoring ---
    "measure software carbon footprint",
    "green IT metrics dashboard",
    "sustainability reporting technology",
    "energy monitoring IT infrastructure",
    # --- Recherche / concepts ---
    "digital pollution environment",
    "technology environmental impact study",
    "ICT electricity consumption",
    "sustainable digitalization",
    "green internet infrastructure",
    "green telecom network",
]

# === MOTS-CLÉS NON GREEN IT (60) ===
# Articles technologiques sans rapport avec la durabilité environnementale.
NON_GREEN_QUERIES: list[str] = [
    # --- IA générale ---
    "artificial intelligence healthcare diagnosis",
    "machine learning stock prediction",
    "natural language processing chatbot",
    "computer vision facial recognition",
    "generative AI content creation",
    "AI drug discovery pharmaceutical",
    "reinforcement learning game",
    "AI hiring recruitment tool",
    "deepfake detection technology",
    "AI customer service automation",
    # --- Cybersécurité ---
    "cybersecurity data breach",
    "ransomware attack prevention",
    "zero trust security model",
    "phishing email detection",
    "cloud security vulnerability",
    # --- Blockchain / crypto ---
    "cryptocurrency market regulation",
    "bitcoin price prediction",
    "NFT digital art marketplace",
    "blockchain supply chain tracking",
    "DeFi decentralized finance",
    # --- Réseaux / télécom ---
    "5G network deployment city",
    "satellite internet Starlink",
    "WiFi 7 technology standard",
    "network slicing telecom",
    # --- Divertissement / média ---
    "video game industry revenue",
    "streaming platform technology",
    "virtual reality gaming headset",
    "augmented reality retail shopping",
    "social media algorithm recommendation",
    "content moderation platform",
    # --- Mobile / consumer tech ---
    "smartphone innovation feature",
    "wearable technology health tracker",
    "smart home automation device",
    "electric vehicle battery technology",
    "drone delivery logistics",
    # --- Business / fintech ---
    "fintech digital banking app",
    "e-commerce technology platform",
    "SaaS enterprise software",
    "digital transformation company",
    "low code no code development",
    # --- Recherche fondamentale ---
    "quantum computing qubit",
    "neuromorphic computing chip",
    "DNA data storage technology",
    "optical computing processor",
    # --- Robotique / automation ---
    "robotics warehouse automation",
    "autonomous vehicle regulation",
    "surgical robot AI",
    "industrial automation factory",
    # --- Données / analytics ---
    "big data analytics enterprise",
    "predictive analytics business",
    "data governance compliance",
    "real-time data processing",
    # --- Cloud / DevOps (sans angle green) ---
    "kubernetes container orchestration",
    "microservices architecture",
    "CI CD pipeline automation",
    "infrastructure as code Terraform",
    # --- Divers tech ---
    "biometric authentication technology",
    "digital twin manufacturing",
    "AI chip GPU performance benchmark",
    "tech startup funding venture capital",
]


async def _fetch_articles(
    client: httpx.AsyncClient,
    keyword: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """Interroge l'API NewsData.io pour un mot-clé."""
    params = {
        "apikey": api_key,
        "q": keyword,
        "language": "en",
        "size": MAX_RESULTS,
    }

    response = await client.get(NEWSDATA_BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "success":
        return []

    articles = []
    for article in data.get("results", []):
        titre = article.get("title")
        url = article.get("link")
        if not titre or not url:
            continue

        articles.append({
            "titre": titre,
            "url": url,
            "description": article.get("description", ""),
            "contenu": article.get("content", ""),
            "date_publication": article.get("pubDate"),
            "auteur": ", ".join(article.get("creator") or []) or None,
            "source_nom": article.get("source_id", "newsdata"),
            "langue": article.get("language", "en"),
            "categorie": article.get("category", []),
        })

    return articles


async def run_targeted_collection() -> dict[str, int]:
    """Lance la collecte ciblée avec délai entre les requêtes.

    Returns:
        Statistiques {total, green_queries, non_green_queries, errors}.
    """
    settings = get_settings()
    api_key = settings.api_news_key

    if not api_key or api_key.startswith("YOUR_"):
        logger.error("Clé API non configurée dans .env")
        return {"total": 0, "green_queries": 0, "non_green_queries": 0, "errors": 0}

    stats = {"total": 0, "green_queries": 0, "non_green_queries": 0, "errors": 0}

    all_queries = (
        [(kw, "green_it") for kw in GREEN_IT_QUERIES]
        + [(kw, "non_green_it") for kw in NON_GREEN_QUERIES]
    )

    logger.info(f"Collecte ciblée : {len(all_queries)} requêtes prévues")
    logger.info(f"Délai entre requêtes : {REQUEST_DELAY}s")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for i, (keyword, category) in enumerate(all_queries):
            # Délai anti rate-limit
            if i > 0:
                await asyncio.sleep(REQUEST_DELAY)

            try:
                articles = await _fetch_articles(client, keyword, api_key)

                if not articles:
                    logger.debug(f"[{i+1}/{len(all_queries)}] '{keyword}' : 0 articles")
                    continue

                raw_path = generate_raw_path("api", f"newsdata_targeted_{category}")
                payload = {
                    "keyword": keyword,
                    "category": category,
                    "source": "newsdata.io",
                    "articles_count": len(articles),
                    "articles": articles,
                }
                path = await upload_json_to_minio(
                    payload,
                    bucket=settings.minio_bucket_raw,
                    object_name=raw_path,
                )

                stats["total"] += len(articles)
                if category == "green_it":
                    stats["green_queries"] += 1
                else:
                    stats["non_green_queries"] += 1

                logger.info(
                    f"[{i+1}/{len(all_queries)}] [{category}] '{keyword}' : "
                    f"{len(articles)} articles -> {path}"
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(
                        f"[{i+1}/{len(all_queries)}] Rate limit atteint. "
                        f"Pause de 60s puis reprise..."
                    )
                    await asyncio.sleep(60)
                    stats["errors"] += 1
                else:
                    logger.error(f"Erreur HTTP {e.response.status_code} pour '{keyword}'")
                    stats["errors"] += 1
            except Exception as e:
                logger.error(f"Erreur pour '{keyword}' : {e}")
                stats["errors"] += 1

    logger.info("=" * 60)
    logger.info(
        f"COLLECTE TERMINÉE : {stats['total']} articles | "
        f"{stats['green_queries']} requêtes green | "
        f"{stats['non_green_queries']} requêtes non-green | "
        f"{stats['errors']} erreurs"
    )
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    asyncio.run(run_targeted_collection())
