"""Routes pour la consultation des articles.

Endpoints publics de lecture des articles avec pagination,
filtres et recherche textuelle.

"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from greentech.api.dependencies import get_db
from greentech.api.schemas.article import (
    ArticleDetail,
    ArticleListItem,
    ArticleListResponse,
)
from greentech.data.storage.models import Article, Source

router = APIRouter(prefix="/articles", tags=["Articles"])


@router.get(
    "",
    response_model=ArticleListResponse,
    summary="Liste paginee des articles",
)
async def list_articles(
    page: int = Query(1, ge=1, description="Numero de page"),
    limit: int = Query(20, ge=1, le=100, description="Articles par page"),
    is_green_it: bool | None = Query(None, description="Filtre Green IT"),
    source_id: int | None = Query(None, description="Filtre par source"),
    date_from: datetime | None = Query(None, description="Date de publication minimale"),
    date_to: datetime | None = Query(None, description="Date de publication maximale"),
    db: AsyncSession = Depends(get_db),
) -> ArticleListResponse:
    """Retourne une liste paginee d'articles avec filtres optionnels.

    Supporte le filtrage par classification Green IT, source de donnees
    et plage de dates de publication.

    Args:
        page: Numero de page (commence a 1).
        limit: Nombre d'articles par page (max 100).
        is_green_it: Filtre par classification Green IT.
        source_id: Filtre par identifiant de source.
        date_from: Date de publication minimale (inclusive).
        date_to: Date de publication maximale (inclusive).
        db: Session de base de donnees.

    Returns:
        Liste paginee d'articles avec metadonnees de pagination.
    """
    # Construction de la requete de base
    base_query = select(Article).outerjoin(Source, Article.id_source == Source.id_source)

    # Application des filtres
    if is_green_it is not None:
        base_query = base_query.where(Article.est_green_it == is_green_it)
    if source_id is not None:
        base_query = base_query.where(Article.id_source == source_id)
    if date_from is not None:
        base_query = base_query.where(Article.date_publication >= date_from)
    if date_to is not None:
        base_query = base_query.where(Article.date_publication <= date_to)

    # Comptage total
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Pagination
    offset = (page - 1) * limit
    pages = (total + limit - 1) // limit if total > 0 else 0

    stmt = (
        base_query
        .add_columns(Source.nom.label("nom_source"))
        .order_by(Article.date_creation.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    articles = [
        ArticleListItem(
            id_article=article.id_article,
            uuid=str(article.uuid),
            titre=article.titre,
            url=article.url,
            resume=article.resume,
            auteur=article.auteur,
            date_publication=article.date_publication,
            est_green_it=article.est_green_it,
            score_confiance=article.score_confiance,
            nom_source=nom_source,
            date_creation=article.date_creation,
        )
        for article, nom_source in rows
    ]

    logger.debug(f"Articles listes : page={page}, limit={limit}, total={total}")

    return ArticleListResponse(
        articles=articles,
        total=total,
        page=page,
        limit=limit,
        pages=pages,
    )


@router.get(
    "/search",
    response_model=ArticleListResponse,
    summary="Recherche d'articles",
)
async def search_articles(
    q: str = Query(..., min_length=2, max_length=200, description="Terme de recherche"),
    page: int = Query(1, ge=1, description="Numero de page"),
    limit: int = Query(20, ge=1, le=100, description="Articles par page"),
    db: AsyncSession = Depends(get_db),
) -> ArticleListResponse:
    """Recherche d'articles par titre ou contenu.

    Effectue une recherche insensible a la casse dans le titre
    et le contenu des articles.

    Args:
        q: Terme de recherche (minimum 2 caracteres).
        page: Numero de page.
        limit: Nombre de resultats par page.
        db: Session de base de donnees.

    Returns:
        Liste paginee des articles correspondants.
    """
    search_pattern = f"%{q}%"

    base_query = (
        select(Article)
        .outerjoin(Source, Article.id_source == Source.id_source)
        .where(
            or_(
                Article.titre.ilike(search_pattern),
                Article.contenu.ilike(search_pattern),
            )
        )
    )

    # Comptage
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    offset = (page - 1) * limit
    pages = (total + limit - 1) // limit if total > 0 else 0

    stmt = (
        base_query
        .add_columns(Source.nom.label("nom_source"))
        .order_by(Article.date_creation.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    articles = [
        ArticleListItem(
            id_article=article.id_article,
            uuid=str(article.uuid),
            titre=article.titre,
            url=article.url,
            resume=article.resume,
            auteur=article.auteur,
            date_publication=article.date_publication,
            est_green_it=article.est_green_it,
            score_confiance=article.score_confiance,
            nom_source=nom_source,
            date_creation=article.date_creation,
        )
        for article, nom_source in rows
    ]

    logger.debug(f"Recherche '{q}' : {total} resultats")

    return ArticleListResponse(
        articles=articles,
        total=total,
        page=page,
        limit=limit,
        pages=pages,
    )


@router.get(
    "/{article_id}",
    response_model=ArticleDetail,
    summary="Detail d'un article",
)
async def get_article(
    article_id: int,
    db: AsyncSession = Depends(get_db),
) -> ArticleDetail:
    """Retourne le detail complet d'un article par son identifiant.

    Args:
        article_id: Identifiant numerique de l'article.
        db: Session de base de donnees.

    Returns:
        Detail complet de l'article avec les resultats d'analyse IA.

    Raises:
        HTTPException: 404 si l'article n'existe pas.
    """
    stmt = (
        select(Article, Source.nom.label("nom_source"))
        .outerjoin(Source, Article.id_source == Source.id_source)
        .where(Article.id_article == article_id)
    )
    result = await db.execute(stmt)
    row = result.one_or_none()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article id={article_id} introuvable",
        )

    article, nom_source = row

    return ArticleDetail(
        id_article=article.id_article,
        uuid=str(article.uuid),
        titre=article.titre,
        url=article.url,
        contenu=article.contenu,
        resume=article.resume,
        auteur=article.auteur,
        date_publication=article.date_publication,
        langue=article.langue,
        est_green_it=article.est_green_it,
        score_confiance=article.score_confiance,
        modele_classification=article.modele_classification,
        date_analyse=article.date_analyse,
        nom_source=nom_source,
        date_creation=article.date_creation,
        date_modification=article.date_modification,
    )
