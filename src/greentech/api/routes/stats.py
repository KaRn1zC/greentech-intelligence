"""Routes pour les statistiques de la plateforme.

Endpoints publics exposant les metriques globales, quotidiennes
et par source de donnees.

"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.api.dependencies import get_db
from greentech.api.schemas.stats import (
    DailyStatsItem,
    DailyStatsResponse,
    GlobalStats,
    SourceStatsItem,
    SourceStatsResponse,
)
from greentech.data.storage.models import Article, Source

router = APIRouter(prefix="/stats", tags=["Statistiques"])


@router.get(
    "",
    response_model=GlobalStats,
    summary="Statistiques globales",
)
async def get_global_stats(
    db: AsyncSession = Depends(get_db),
) -> GlobalStats:
    """Retourne les statistiques globales de la plateforme.

    Calcule en temps reel le nombre total d'articles, la repartition
    Green IT / Non Green IT et le score de confiance moyen.

    Args:
        db: Session de base de donnees.

    Returns:
        Statistiques globales agregees.
    """
    stmt = select(
        func.count(Article.id_article).label("total"),
        func.count(Article.id_article).filter(Article.est_green_it.is_(True)).label("green"),
        func.count(Article.id_article).filter(Article.est_green_it.is_(False)).label("non_green"),
        func.count(Article.id_article).filter(Article.est_green_it.is_(None)).label("en_attente"),
        func.avg(Article.score_confiance).label("confiance_moy"),
    )
    result = await db.execute(stmt)
    row = result.one()

    total_classifies = row.green + row.non_green
    pourcentage = round(row.green / total_classifies * 100, 2) if total_classifies > 0 else None

    return GlobalStats(
        total_articles=row.total,
        articles_green_it=row.green,
        articles_non_green_it=row.non_green,
        en_attente_analyse=row.en_attente,
        score_confiance_moyen=round(float(row.confiance_moy), 3) if row.confiance_moy else None,
        pourcentage_green_it=pourcentage,
    )


@router.get(
    "/daily",
    response_model=DailyStatsResponse,
    summary="Statistiques par jour",
)
async def get_daily_stats(
    days: int = Query(30, ge=1, le=365, description="Nombre de jours a remonter"),
    db: AsyncSession = Depends(get_db),
) -> DailyStatsResponse:
    """Retourne les statistiques d'articles par jour sur une periode donnee.

    Agrege le nombre d'articles par jour de creation, avec la repartition
    Green IT / Non Green IT.

    Args:
        days: Nombre de jours a remonter (defaut 30, max 365).
        db: Session de base de donnees.

    Returns:
        Statistiques quotidiennes sur la periode demandee.
    """
    date_debut = date.today() - timedelta(days=days)
    date_col = func.date(Article.date_creation)

    stmt = (
        select(
            date_col.label("jour"),
            func.count(Article.id_article).label("total"),
            func.count(case((Article.est_green_it.is_(True), Article.id_article))).label("green"),
            func.count(case((Article.est_green_it.is_(False), Article.id_article))).label(
                "non_green"
            ),
            func.avg(Article.score_confiance).label("confiance_moy"),
        )
        .where(func.date(Article.date_creation) >= date_debut)
        .group_by(date_col)
        .order_by(date_col.asc())
    )

    result = await db.execute(stmt)
    rows = result.all()

    stats = [
        DailyStatsItem(
            date_stat=row.jour,
            total_articles=row.total,
            articles_green_it=row.green,
            articles_non_green_it=row.non_green,
            score_confiance_moyen=round(float(row.confiance_moy), 3) if row.confiance_moy else None,
        )
        for row in rows
    ]

    return DailyStatsResponse(
        stats=stats,
        periode_debut=date_debut,
        periode_fin=date.today(),
    )


@router.get(
    "/sources",
    response_model=SourceStatsResponse,
    summary="Statistiques par source",
)
async def get_source_stats(
    db: AsyncSession = Depends(get_db),
) -> SourceStatsResponse:
    """Retourne les statistiques d'articles par source de donnees.

    Pour chaque source, indique le nombre total d'articles et la
    repartition Green IT / Non Green IT.

    Args:
        db: Session de base de donnees.

    Returns:
        Statistiques par source de donnees.
    """
    stmt = (
        select(
            Source.id_source,
            Source.nom,
            Source.type,
            Source.derniere_collecte,
            func.count(Article.id_article).label("total"),
            func.count(case((Article.est_green_it.is_(True), Article.id_article))).label("green"),
            func.count(case((Article.est_green_it.is_(False), Article.id_article))).label(
                "non_green"
            ),
        )
        .outerjoin(Article, Source.id_source == Article.id_source)
        .group_by(Source.id_source, Source.nom, Source.type, Source.derniere_collecte)
        .order_by(Source.nom)
    )

    result = await db.execute(stmt)
    rows = result.all()

    sources = [
        SourceStatsItem(
            id_source=row.id_source,
            nom=row.nom,
            type=row.type,
            total_articles=row.total,
            articles_green_it=row.green,
            articles_non_green_it=row.non_green,
            derniere_collecte=row.derniere_collecte,
        )
        for row in rows
    ]

    return SourceStatsResponse(sources=sources)
