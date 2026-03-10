"""Client MinIO pour le stockage objet S3-compatible.

Gère l'upload et le download de fichiers vers les buckets
raw-data et clean-data du Data Lake.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any

from loguru import logger
from minio import Minio

from greentech.config import get_settings


def get_minio_client() -> Minio:
    """Crée et retourne un client MinIO configuré.

    Returns:
        Instance du client MinIO prête à l'emploi.
    """
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=False,
    )


async def upload_json_to_minio(
    data: dict[str, Any] | list[dict[str, Any]],
    *,
    bucket: str,
    object_name: str,
) -> str:
    """Upload un objet JSON vers MinIO.

    Args:
        data: Données à sérialiser en JSON.
        bucket: Nom du bucket cible.
        object_name: Chemin de l'objet dans le bucket.

    Returns:
        Chemin complet de l'objet stocké (bucket/object_name).

    Raises:
        Exception: Si l'upload échoue.
    """
    client = get_minio_client()
    json_bytes = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    stream = io.BytesIO(json_bytes)

    client.put_object(
        bucket,
        object_name,
        stream,
        length=len(json_bytes),
        content_type="application/json",
    )

    path = f"{bucket}/{object_name}"
    logger.debug(f"Upload JSON vers MinIO : {path} ({len(json_bytes)} octets)")
    return path


async def upload_raw_to_minio(
    content: str | bytes,
    *,
    bucket: str,
    object_name: str,
    content_type: str = "text/html",
) -> str:
    """Upload du contenu brut (HTML, texte) vers MinIO.

    Args:
        content: Contenu à stocker (str ou bytes).
        bucket: Nom du bucket cible.
        object_name: Chemin de l'objet dans le bucket.
        content_type: Type MIME du contenu.

    Returns:
        Chemin complet de l'objet stocké.
    """
    client = get_minio_client()
    if isinstance(content, str):
        content = content.encode("utf-8")

    stream = io.BytesIO(content)
    client.put_object(
        bucket,
        object_name,
        stream,
        length=len(content),
        content_type=content_type,
    )

    path = f"{bucket}/{object_name}"
    logger.debug(f"Upload brut vers MinIO : {path} ({len(content)} octets)")
    return path


def download_from_minio(*, bucket: str, object_name: str) -> bytes:
    """Télécharge un objet depuis MinIO.

    Args:
        bucket: Nom du bucket source.
        object_name: Chemin de l'objet dans le bucket.

    Returns:
        Contenu brut de l'objet en bytes.
    """
    client = get_minio_client()
    response = client.get_object(bucket, object_name)
    try:
        data = response.read()
    finally:
        response.close()
        response.release_conn()

    logger.debug(f"Download depuis MinIO : {bucket}/{object_name} ({len(data)} octets)")
    return data


def generate_raw_path(source_type: str, source_name: str, extension: str = "json") -> str:
    """Génère un chemin unique pour stocker un fichier brut dans MinIO.

    Format : {source_type}/{source_name}/{YYYY-MM-DD}/{timestamp}.{ext}

    Args:
        source_type: Type de source (api, scraping, file).
        source_name: Nom de la source (ex: newsdata, techcrunch).
        extension: Extension du fichier.

    Returns:
        Chemin de l'objet formaté.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{source_type}/{source_name}/{date_str}/{timestamp}.{extension}"
