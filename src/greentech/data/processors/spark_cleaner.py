"""Module de nettoyage et agrégation Big Data via Apache Spark.

Lit les données brutes depuis MinIO (raw-data) via le protocole S3A,
applique les transformations de nettoyage (HTML, dates, encodages),
anonymise les auteurs (RGPD) et sauvegarde le résultat nettoyé
dans MinIO (clean-data) au format Parquet.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import re

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from greentech.config import get_settings

# Schéma unifié pour les articles nettoyés (toutes sources confondues)
CLEAN_SCHEMA = StructType(
    [
        StructField("titre", StringType(), nullable=False),
        StructField("url", StringType(), nullable=False),
        StructField("contenu", StringType(), nullable=True),
        StructField("auteur", StringType(), nullable=True),
        StructField("date_publication", StringType(), nullable=True),
        StructField("source_nom", StringType(), nullable=True),
        StructField("langue", StringType(), nullable=True),
    ]
)


def create_spark_session() -> SparkSession:
    """Crée et configure une session Spark connectée à MinIO via S3A.

    Configure les connecteurs Hadoop-AWS pour que Spark puisse lire
    et écrire directement dans les buckets MinIO en utilisant le
    protocole S3A (compatible Amazon S3).

    Returns:
        Session Spark configurée et prête à l'emploi.
    """
    settings = get_settings()

    # Extraire host et port depuis l'endpoint MinIO
    endpoint = settings.minio_endpoint
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"

    # Déterminer le chemin Python du venv pour les workers Spark
    import sys

    python_path = sys.executable

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("GreenTech-DataCleaner")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.4.2,com.amazonaws:aws-java-sdk-bundle:1.12.367",
        )
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.access.key", settings.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", settings.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .config("spark.pyspark.python", python_path)
        .config("spark.pyspark.driver.python", python_path)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Session Spark créée (version {spark.version})")
    return spark


# ---------------------------------------------------------------------------
# UDFs de nettoyage
# ---------------------------------------------------------------------------


def _strip_html(text: str | None) -> str | None:
    """Supprime les balises HTML d'un texte.

    Args:
        text: Texte potentiellement contenant du HTML.

    Returns:
        Texte nettoyé sans balises HTML, ou None si l'entrée est None.
    """
    if not text:
        return None
    # Supprimer les balises HTML
    clean = re.sub(r"<[^>]+>", " ", text)
    # Supprimer les entités HTML courantes
    clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
    # Normaliser les espaces multiples
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean if clean else None


def _anonymiser_auteur(nom_complet: str | None) -> str | None:
    """Transforme un nom complet en initiales pour la conformité RGPD.

    Applique la règle d'anonymisation définie dans le registre RGPD :
    "John Doe" → "J.D.", "Marie-Claire Dubois" → "M.D."

    Args:
        nom_complet: Nom complet de l'auteur.

    Returns:
        Initiales formatées, "Auteur anonyme" si vide, ou None.
    """
    if not nom_complet or nom_complet.strip() == "":
        return "Auteur anonyme"

    # Gestion des listes d'auteurs séparées par des virgules
    auteurs = nom_complet.split(",")
    initiales_list = []

    for auteur in auteurs[:3]:  # Limiter à 3 auteurs max
        auteur = auteur.strip()
        if not auteur:
            continue
        mots = auteur.split()
        initiales = [mot[0].upper() for mot in mots if mot]
        if initiales:
            initiales_list.append(".".join(initiales) + ".")

    return ", ".join(initiales_list) if initiales_list else "Auteur anonyme"


def _normaliser_date(date_str: str | None) -> str | None:
    """Normalise une date au format ISO 8601 (YYYY-MM-DDTHH:MM:SSZ).

    Gère les formats courants : ISO complet, date seule, timestamps Unix.

    Args:
        date_str: Chaîne représentant une date.

    Returns:
        Date au format ISO 8601 ou None si non parsable.
    """
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip()

    # Déjà au format ISO 8601 complet
    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", date_str):
        return date_str[:19] + "Z" if "Z" not in date_str and "+" not in date_str else date_str

    # Format date seule : YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return f"{date_str}T00:00:00Z"

    # Format avec espace : YYYY-MM-DD HH:MM:SS
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", date_str):
        return date_str[:19].replace(" ", "T") + "Z"

    return None


# ---------------------------------------------------------------------------
# Fonctions de lecture depuis MinIO
# ---------------------------------------------------------------------------


def read_raw_json_from_minio(spark: SparkSession, bucket: str, path: str) -> DataFrame:
    """Lit des fichiers JSON depuis un bucket MinIO via S3A.

    Valide la compétence "Connexion et extraction depuis un système Big Data".

    Args:
        spark: Session Spark active.
        bucket: Nom du bucket MinIO.
        path: Chemin du répertoire dans le bucket (lecture récursive).

    Returns:
        DataFrame Spark contenant les données brutes.
    """
    s3a_path = f"s3a://{bucket}/{path}"
    logger.info(f"Lecture depuis MinIO : {s3a_path}")

    df = spark.read.option("multiline", "true").option("recursiveFileLookup", "true").json(s3a_path)
    logger.info(f"Données lues : {df.count()} enregistrements")
    return df


# ---------------------------------------------------------------------------
# Pipeline de nettoyage
# ---------------------------------------------------------------------------


def clean_api_data(df: DataFrame) -> DataFrame:
    """Nettoie les données provenant de l'API NewsData.io.

    Extrait les articles depuis la structure JSON imbriquée,
    normalise les champs et supprime les entrées incomplètes.

    Args:
        df: DataFrame brut depuis MinIO (structure newsdata).

    Returns:
        DataFrame nettoyé au schéma unifié.
    """
    logger.info("Nettoyage données API...")

    # Exploser le tableau d'articles imbriqué
    articles_df = df.select(F.explode("articles").alias("article"))

    # Extraire et normaliser les champs
    clean_df = articles_df.select(
        F.col("article.titre").alias("titre"),
        F.col("article.url").alias("url"),
        F.col("article.contenu").alias("contenu"),
        F.col("article.auteur").alias("auteur"),
        F.col("article.date_publication").alias("date_publication"),
        F.col("article.source_nom").alias("source_nom"),
        F.col("article.langue").alias("langue"),
    )

    return clean_df


def clean_scraping_data(df: DataFrame) -> DataFrame:
    """Nettoie les données provenant du scraping TechCrunch.

    Supprime les balises HTML du contenu et normalise les champs.

    Args:
        df: DataFrame brut depuis MinIO (structure scraping).

    Returns:
        DataFrame nettoyé au schéma unifié.
    """
    logger.info("Nettoyage données scraping...")

    articles_df = df.select(F.explode("articles").alias("article"))

    clean_df = articles_df.select(
        F.col("article.titre").alias("titre"),
        F.col("article.url").alias("url"),
        F.col("article.contenu_html").alias("contenu"),
        F.col("article.auteur").alias("auteur"),
        F.col("article.date_publication").alias("date_publication"),
        F.col("article.source_nom").alias("source_nom"),
        F.lit("en").alias("langue"),
    )

    return clean_df


def clean_file_data(df: DataFrame) -> DataFrame:
    """Nettoie les données provenant du dataset arXiv.

    Normalise les champs pour correspondre au schéma unifié.

    Args:
        df: DataFrame brut depuis MinIO (structure arxiv).

    Returns:
        DataFrame nettoyé au schéma unifié.
    """
    logger.info("Nettoyage données fichier (arXiv)...")

    articles_df = df.select(F.explode("articles").alias("article"))

    clean_df = articles_df.select(
        F.col("article.titre").alias("titre"),
        F.col("article.url").alias("url"),
        F.col("article.contenu").alias("contenu"),
        F.col("article.auteur").alias("auteur"),
        F.col("article.date_publication").alias("date_publication"),
        F.col("article.source_nom").alias("source_nom"),
        F.lit("en").alias("langue"),
    )

    return clean_df


def apply_cleaning_pipeline(df: DataFrame) -> DataFrame:
    """Applique le pipeline complet de nettoyage sur un DataFrame unifié.

    Opérations effectuées :
    1. Suppression des balises HTML dans le contenu
    2. Anonymisation des auteurs (RGPD)
    3. Normalisation des dates (ISO 8601)
    4. Suppression des entrées corrompues (sans titre ou URL)
    5. Suppression des doublons (par URL)
    6. Normalisation des espaces et encodages

    Args:
        df: DataFrame agrégé contenant tous les articles.

    Returns:
        DataFrame nettoyé et prêt pour l'insertion en base.
    """
    logger.info(f"Pipeline de nettoyage : {df.count()} articles en entrée")

    # --- Toutes les transformations utilisent des fonctions Spark SQL natives ---
    # (plus performant que les UDFs Python et compatible Windows)

    # 1. Nettoyage HTML du contenu et du titre
    for col_name in ["contenu", "titre"]:
        df = df.withColumn(col_name, F.regexp_replace(F.col(col_name), r"<[^>]+>", " "))
        df = df.withColumn(col_name, F.regexp_replace(F.col(col_name), r"&[a-zA-Z]+;", " "))
        df = df.withColumn(col_name, F.trim(F.regexp_replace(F.col(col_name), r"\s+", " ")))
        # Remplacer les chaînes vides par null
        df = df.withColumn(
            col_name,
            F.when(F.length(F.trim(F.col(col_name))) == 0, None).otherwise(F.col(col_name)),
        )

    # 2. Anonymisation des auteurs (RGPD) — fonctions Spark SQL natives
    # Découpe par virgule, garde max 3 auteurs, extrait les initiales de chaque mot
    auteurs_array = F.slice(F.split(F.col("auteur"), ","), 1, 3)
    initiales_array = F.transform(
        auteurs_array,
        lambda auteur: F.concat(
            F.concat_ws(
                ".",
                F.transform(
                    F.split(F.trim(auteur), " "),
                    lambda mot: F.upper(F.substring(mot, 1, 1)),
                ),
            ),
            F.lit("."),
        ),
    )
    df = df.withColumn(
        "auteur",
        F.when(
            F.col("auteur").isNull() | (F.trim(F.col("auteur")) == ""),
            F.lit("Auteur anonyme"),
        ).otherwise(F.array_join(initiales_array, ", ")),
    )
    # Si le résultat est vide après transformation, mettre "Auteur anonyme"
    df = df.withColumn(
        "auteur",
        F.when(
            F.col("auteur").isNull() | (F.trim(F.col("auteur")) == ""),
            F.lit("Auteur anonyme"),
        ).otherwise(F.col("auteur")),
    )

    # 3. Normalisation des dates (ISO 8601)
    date_col = F.trim(F.col("date_publication"))
    df = df.withColumn(
        "date_publication",
        F.when(
            date_col.isNull() | (F.length(date_col) == 0),
            F.lit(None),
        )
        .when(
            # Déjà ISO complet : YYYY-MM-DDTHH:MM:SS...
            date_col.rlike(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"),
            F.when(
                date_col.contains("Z") | date_col.contains("+"),
                date_col,
            ).otherwise(F.concat(F.substring(date_col, 1, 19), F.lit("Z"))),
        )
        .when(
            # Date seule : YYYY-MM-DD
            date_col.rlike(r"^\d{4}-\d{2}-\d{2}$"),
            F.concat(date_col, F.lit("T00:00:00Z")),
        )
        .when(
            # Date avec espace : YYYY-MM-DD HH:MM:SS
            date_col.rlike(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"),
            F.concat(
                F.regexp_replace(F.substring(date_col, 1, 19), " ", "T"),
                F.lit("Z"),
            ),
        )
        .otherwise(F.lit(None)),
    )

    # 4. Suppression des entrées corrompues (titre ou URL manquant)
    before = df.count()
    df = df.filter(F.col("titre").isNotNull() & (F.trim(F.col("titre")) != ""))
    df = df.filter(F.col("url").isNotNull() & (F.trim(F.col("url")) != ""))
    after = df.count()
    removed = before - after
    if removed > 0:
        logger.warning(f"Entrées corrompues supprimées : {removed}")

    # 5. Suppression des doublons par URL
    before = df.count()
    df = df.dropDuplicates(["url"])
    after = df.count()
    dupes = before - after
    if dupes > 0:
        logger.info(f"Doublons supprimés : {dupes}")

    logger.info(f"Pipeline terminé : {df.count()} articles en sortie")
    return df


# ---------------------------------------------------------------------------
# Orchestration complète
# ---------------------------------------------------------------------------


def run_spark_cleaning() -> int:
    """Orchestre le pipeline complet de nettoyage Big Data.

    1. Crée la session Spark avec connecteur S3A/MinIO
    2. Lit les données brutes depuis les 3 sources dans MinIO raw-data
    3. Nettoie et agrège les DataFrames
    4. Sauvegarde le résultat en Parquet dans MinIO clean-data

    Returns:
        Nombre total d'articles nettoyés et sauvegardés.
    """
    settings = get_settings()
    spark = create_spark_session()

    try:
        dataframes: list[DataFrame] = []

        # --- Lecture des données brutes depuis MinIO ---
        # Source API
        try:
            api_df = read_raw_json_from_minio(spark, settings.minio_bucket_raw, "api")
            api_clean = clean_api_data(api_df)
            dataframes.append(api_clean)
            logger.info(f"API : {api_clean.count()} articles chargés")
        except Exception as e:
            logger.warning(f"Aucune donnée API trouvée ou erreur : {e}")

        # Source Scraping
        try:
            scraping_df = read_raw_json_from_minio(
                spark, settings.minio_bucket_raw, "scraping/techcrunch"
            )
            scraping_clean = clean_scraping_data(scraping_df)
            dataframes.append(scraping_clean)
            logger.info(f"Scraping : {scraping_clean.count()} articles chargés")
        except Exception as e:
            logger.warning(f"Aucune donnée scraping trouvée ou erreur : {e}")

        # Source Fichier (arXiv)
        try:
            file_df = read_raw_json_from_minio(
                spark, settings.minio_bucket_raw, "file"
            )
            file_clean = clean_file_data(file_df)
            dataframes.append(file_clean)
            logger.info(f"Fichier : {file_clean.count()} articles chargés")
        except Exception as e:
            logger.warning(f"Aucune donnée fichier trouvée ou erreur : {e}")

        if not dataframes:
            logger.warning("Aucune donnée brute trouvée dans MinIO raw-data")
            return 0

        # --- Agrégation des 3 sources ---
        logger.info(f"Agrégation de {len(dataframes)} sources...")
        aggregated_df = dataframes[0]
        for df in dataframes[1:]:
            aggregated_df = aggregated_df.unionByName(df, allowMissingColumns=True)

        logger.info(f"Total agrégé : {aggregated_df.count()} articles")

        # --- Nettoyage complet ---
        clean_df = apply_cleaning_pipeline(aggregated_df)

        # --- Sauvegarde en Parquet dans MinIO clean-data ---
        output_path = f"s3a://{settings.minio_bucket_clean}/articles"
        logger.info(f"Sauvegarde Parquet vers : {output_path}")

        clean_df.write.mode("overwrite").parquet(output_path)

        total = clean_df.count()
        logger.info(f"Nettoyage terminé : {total} articles sauvegardés en Parquet")
        return total

    finally:
        spark.stop()
        logger.info("Session Spark fermée")


if __name__ == "__main__":
    import os
    import sys

    # Configurer JAVA_HOME si nécessaire
    java_home = os.environ.get("JAVA_HOME", "")
    if not java_home:
        temurin_path = "C:/Program Files/Eclipse Adoptium/jdk-21.0.10.7-hotspot"
        if os.path.exists(temurin_path):
            os.environ["JAVA_HOME"] = temurin_path
            os.environ["PATH"] = f"{temurin_path}/bin;{os.environ['PATH']}"

    # Forcer l'IP locale pour éviter les problèmes de résolution IPv6 sur Windows
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    # Configurer HADOOP_HOME pour winutils.exe (requis par Spark sur Windows)
    hadoop_home = os.environ.get("HADOOP_HOME", "")
    if not hadoop_home:
        hadoop_path = "C:/hadoop"
        if os.path.exists(f"{hadoop_path}/bin/winutils.exe"):
            os.environ["HADOOP_HOME"] = hadoop_path
            os.environ["PATH"] = f"{hadoop_path}/bin;{os.environ['PATH']}"

    total = run_spark_cleaning()
    logger.info(f"Résultat final : {total} articles nettoyés")
    sys.exit(0 if total >= 0 else 1)
