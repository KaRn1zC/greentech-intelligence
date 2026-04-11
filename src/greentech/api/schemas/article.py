"""Schemas Pydantic pour les articles.

Definit les modeles de requete et de reponse pour les endpoints /articles.
Assure la validation des entrees et la serialisation des sorties.

"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ArticleBase(BaseModel):
    """Champs communs a tous les schemas d'article."""

    titre: str = Field(..., min_length=1, max_length=500, description="Titre de l'article")
    url: str = Field(..., min_length=1, description="URL de l'article")


class ArticleListItem(BaseModel):
    """Schema leger pour la liste paginee d'articles."""

    model_config = ConfigDict(from_attributes=True)

    id_article: int
    uuid: str
    titre: str
    url: str
    resume: str | None = None
    auteur: str | None = None
    date_publication: datetime | None = None
    est_green_it: bool | None = None
    score_confiance: float | None = None
    nom_source: str | None = None

    date_creation: datetime


class ArticleDetail(BaseModel):
    """Schema complet pour le detail d'un article."""

    model_config = ConfigDict(from_attributes=True)

    id_article: int
    uuid: str
    titre: str
    url: str
    contenu: str | None = None
    resume: str | None = None
    auteur: str | None = None
    date_publication: datetime | None = None
    langue: str = "en"
    est_green_it: bool | None = None
    score_confiance: float | None = None
    modele_classification: str | None = None
    date_analyse: datetime | None = None
    nom_source: str | None = None

    date_creation: datetime
    date_modification: datetime


class ArticleListResponse(BaseModel):
    """Reponse paginee pour la liste d'articles."""

    articles: list[ArticleListItem]
    total: int = Field(..., description="Nombre total d'articles correspondants")
    page: int = Field(..., ge=1, description="Page courante")
    limit: int = Field(..., ge=1, le=100, description="Nombre d'articles par page")
    pages: int = Field(..., ge=0, description="Nombre total de pages")


class ArticleSearchParams(BaseModel):
    """Parametres de recherche d'articles."""

    q: str = Field(..., min_length=2, max_length=200, description="Terme de recherche")
    page: int = Field(1, ge=1, description="Numero de page")
    limit: int = Field(20, ge=1, le=100, description="Nombre de resultats par page")
