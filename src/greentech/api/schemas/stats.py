"""Schemas Pydantic pour les statistiques.

Definit les modeles de reponse pour les endpoints /stats.

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class GlobalStats(BaseModel):
    """Statistiques globales de la plateforme."""

    total_articles: int = Field(..., description="Nombre total d'articles en base")
    articles_green_it: int = Field(..., description="Articles classifies Green IT")
    articles_non_green_it: int = Field(..., description="Articles classifies Non Green IT")
    en_attente_analyse: int = Field(..., description="Articles non encore analyses")
    score_confiance_moyen: float | None = Field(
        None, description="Score de confiance moyen (0.0-1.0)"
    )
    pourcentage_green_it: float | None = Field(
        None, description="Pourcentage d'articles Green IT parmi les analyses"
    )


class DailyStatsItem(BaseModel):
    """Statistiques pour une journee donnee."""

    date_stat: date
    total_articles: int
    articles_green_it: int
    articles_non_green_it: int
    score_confiance_moyen: float | None = None


class DailyStatsResponse(BaseModel):
    """Reponse pour les statistiques quotidiennes."""

    stats: list[DailyStatsItem]
    periode_debut: date
    periode_fin: date


class SourceStatsItem(BaseModel):
    """Statistiques par source de donnees."""

    id_source: int
    nom: str
    type: str
    total_articles: int
    articles_green_it: int
    articles_non_green_it: int
    derniere_collecte: datetime | None = None


class SourceStatsResponse(BaseModel):
    """Reponse pour les statistiques par source."""

    sources: list[SourceStatsItem]


class HealthResponse(BaseModel):
    """Reponse du health check."""

    status: str = Field(default="ok", description="Statut de l'application")
    database: bool = Field(..., description="Connexion PostgreSQL operationnelle")
    version: str = Field(..., description="Version de l'application")
