"""Module de base pour les collecteurs de données.

Fournit la classe abstraite BaseCollector et la fonction
get_config_from_db pour récupérer la configuration dynamique
depuis la table search_config de PostgreSQL.

Rédigé par KaRn1zC - 2026-03-10
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.data.storage.models import SearchConfig


@dataclass
class CollectResult:
    """Résultat d'une collecte de données.

    Attributes:
        source_name: Nom de la source (ex: newsdata, techcrunch).
        articles_count: Nombre d'articles collectés.
        raw_paths: Chemins des fichiers bruts stockés dans MinIO.
        errors: Liste des erreurs rencontrées pendant la collecte.
    """

    source_name: str
    articles_count: int = 0
    raw_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseCollector(ABC):
    """Classe abstraite pour tous les collecteurs de données.

    Chaque collecteur (API, scraping, fichier) hérite de cette classe
    et implémente la méthode collect().
    """

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    async def collect(
        self,
        keywords: list[str],
        **kwargs: Any,
    ) -> CollectResult:
        """Lance la collecte de données.

        Args:
            keywords: Liste de mots-clés pour filtrer les résultats.
            **kwargs: Paramètres additionnels spécifiques au collecteur.

        Returns:
            Résultat de la collecte avec nombre d'articles et chemins MinIO.
        """
        ...


async def get_config_from_db(
    session: AsyncSession,
    *,
    type_source: str | None = None,
    actif_seulement: bool = True,
) -> list[SearchConfig]:
    """Récupère la configuration de recherche depuis la base de données.

    Extrait les mots-clés, URLs et paramètres de la table search_config
    pour les injecter dynamiquement dans les modules de collecte.

    Args:
        session: Session SQLAlchemy asynchrone.
        type_source: Filtre optionnel par type (api, scraping, file).
        actif_seulement: Si True, ne retourne que les configs actives.

    Returns:
        Liste des configurations correspondant aux critères.

    Example:
        ```python
        async with async_session_factory() as session:
            configs = await get_config_from_db(session, type_source="api")
            keywords = [c.mot_cle for c in configs]
        ```
    """
    query = select(SearchConfig).order_by(SearchConfig.priorite)

    if type_source:
        query = query.where(SearchConfig.type_source == type_source)

    if actif_seulement:
        query = query.where(SearchConfig.actif.is_(True))

    result = await session.execute(query)
    configs = list(result.scalars().all())

    logger.info(
        f"Configuration chargée : {len(configs)} entrées"
        f"{f' (type={type_source})' if type_source else ''}"
    )
    for cfg in configs:
        logger.debug(f"  - [{cfg.priorite}] {cfg.mot_cle} ({cfg.type_source})")

    return configs
