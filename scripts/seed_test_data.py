"""Génère des données de test dans MinIO clean-data pour valider l'ingestion SQL.

Crée un fichier Parquet dans MinIO clean-data/articles/ avec le même
schéma que celui produit par le pipeline Spark (spark_cleaner.py).
Utile pour le développement et la validation du script d'ingestion.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from greentech.config import get_settings
from greentech.data.storage.minio_client import get_minio_client

# Schéma Parquet identique à celui produit par spark_cleaner.py
SCHEMA = pa.schema(
    [
        pa.field("titre", pa.string(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
        pa.field("contenu", pa.string(), nullable=True),
        pa.field("auteur", pa.string(), nullable=True),
        pa.field("date_publication", pa.string(), nullable=True),
        pa.field("source_nom", pa.string(), nullable=True),
        pa.field("langue", pa.string(), nullable=True),
    ]
)

# Articles de test simulant la sortie du pipeline Spark
TEST_ARTICLES = [
    # --- Source API (NewsData.io) ---
    {
        "titre": "Green IT Strategies for Modern Data Centers in 2026",
        "url": "https://example.com/articles/green-it-data-centers-2026",
        "contenu": (
            "As sustainability becomes a priority for tech companies, data centers "
            "are adopting innovative cooling systems and renewable energy sources. "
            "This article explores the latest Green IT strategies that reduce "
            "environmental impact while maintaining performance."
        ),
        "auteur": "J.S.",
        "date_publication": "2026-03-08T14:30:00Z",
        "source_nom": "techradar",
        "langue": "en",
    },
    {
        "titre": "Sustainable AI: Reducing the Carbon Footprint of Machine Learning",
        "url": "https://example.com/articles/sustainable-ai-carbon-footprint",
        "contenu": (
            "Machine learning models consume significant energy during training. "
            "Researchers propose new techniques including model pruning, "
            "quantization and efficient architectures to reduce the carbon "
            "footprint of AI systems."
        ),
        "auteur": "M.L.",
        "date_publication": "2026-03-07T09:15:00Z",
        "source_nom": "bbc_news",
        "langue": "en",
    },
    {
        "titre": "Europe Adopts New Digital Sustainability Regulations",
        "url": "https://example.com/articles/europe-digital-sustainability-2026",
        "contenu": (
            "The European Commission has introduced new regulations requiring "
            "tech companies to report their digital carbon emissions. The rules "
            "will take effect in January 2027."
        ),
        "auteur": "Auteur anonyme",
        "date_publication": "2026-03-06T16:45:00Z",
        "source_nom": "newsdata",
        "langue": "en",
    },
    # --- Source Scraping (TechCrunch) ---
    {
        "titre": "Startup Raises $50M to Build Energy-Efficient Cloud Infrastructure",
        "url": "https://techcrunch.com/2026/03/05/startup-energy-efficient-cloud",
        "contenu": (
            "A new startup has raised $50 million in Series B funding to develop "
            "cloud infrastructure that uses 40% less energy than traditional "
            "data centers. Their approach combines liquid cooling with AI-powered "
            "workload optimization."
        ),
        "auteur": "A.C.",
        "date_publication": "2026-03-05T11:00:00Z",
        "source_nom": "TechCrunch Climate",
        "langue": "en",
    },
    {
        "titre": "How Solar-Powered Edge Computing Could Transform Rural Connectivity",
        "url": "https://techcrunch.com/2026/03/04/solar-edge-computing-rural",
        "contenu": (
            "New solar-powered micro data centers are being deployed in rural "
            "areas to provide low-latency computing without relying on the "
            "traditional power grid."
        ),
        "auteur": "R.K.",
        "date_publication": "2026-03-04T08:30:00Z",
        "source_nom": "TechCrunch Climate",
        "langue": "en",
    },
    # --- Source Fichier (arXiv) ---
    {
        "titre": "Efficient Fine-Tuning of Large Language Models with LoRA",
        "url": "https://arxiv.org/abs/2403.12345",
        "contenu": (
            "We propose an improved LoRA method that achieves comparable "
            "performance to full fine-tuning while reducing GPU memory usage "
            "by 75% and training time by 60%. Our approach is evaluated on "
            "classification and generation tasks."
        ),
        "auteur": "W.Z., L.C., K.M.",
        "date_publication": "2026-02-28T00:00:00Z",
        "source_nom": "arXiv Dataset",
        "langue": "en",
    },
    {
        "titre": "Measuring and Reducing Carbon Emissions in Federated Learning",
        "url": "https://arxiv.org/abs/2403.67890",
        "contenu": (
            "Federated learning distributes training across devices but its "
            "carbon footprint remains poorly understood. We present a framework "
            "for measuring and optimizing energy consumption in federated "
            "learning systems."
        ),
        "auteur": "P.R., S.D.",
        "date_publication": "2026-02-25T00:00:00Z",
        "source_nom": "arXiv Dataset",
        "langue": "en",
    },
    {
        "titre": "GreenNLP: A Benchmark for Energy-Efficient Natural Language Processing",
        "url": "https://arxiv.org/abs/2403.11111",
        "contenu": (
            "We introduce GreenNLP, a comprehensive benchmark for evaluating "
            "the energy efficiency of NLP models. Our benchmark covers 12 tasks "
            "and includes metrics for accuracy, latency, and energy consumption."
        ),
        "auteur": "T.B., N.H., J.P.",
        "date_publication": "2026-02-20T00:00:00Z",
        "source_nom": "arXiv Dataset",
        "langue": "en",
    },
]


def seed_test_data() -> int:
    """Génère les données de test en Parquet dans MinIO clean-data/articles/.

    Returns:
        Nombre d'articles écrits.
    """
    settings = get_settings()
    client = get_minio_client()

    # Construire la table PyArrow
    arrays = {field.name: [] for field in SCHEMA}
    for article in TEST_ARTICLES:
        for field_name in arrays:
            arrays[field_name].append(article.get(field_name))

    table = pa.table(arrays, schema=SCHEMA)

    # Écrire en Parquet dans un buffer
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    parquet_bytes = buffer.getvalue()

    # Upload vers MinIO
    object_name = "articles/part-00000-test.parquet"
    stream = io.BytesIO(parquet_bytes)
    client.put_object(
        settings.minio_bucket_clean,
        object_name,
        stream,
        length=len(parquet_bytes),
        content_type="application/octet-stream",
    )

    logger.info(
        f"Données de test créées : {len(TEST_ARTICLES)} articles "
        f"→ {settings.minio_bucket_clean}/{object_name} "
        f"({len(parquet_bytes)} octets)"
    )
    return len(TEST_ARTICLES)


if __name__ == "__main__":
    count = seed_test_data()
    logger.info(f"Seed terminé : {count} articles de test")
