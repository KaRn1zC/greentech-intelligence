"""Modèles ORM SQLAlchemy 2.0 pour la base de données GreenTech.

Réplique exacte du schéma défini dans scripts/sql/init.sql
avec la nomenclature française du MCD/MLD.

"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Classe de base pour tous les modèles ORM."""


class SearchConfig(Base):
    """Configuration des recherches (mots-clés, URLs, priorités)."""

    __tablename__ = "search_config"

    id_config: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mot_cle: Mapped[str] = mapped_column(String(100), nullable=False)
    url_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    type_source: Mapped[str | None] = mapped_column(String(20), nullable=True)
    priorite: Mapped[int] = mapped_column(Integer, default=1)
    actif: Mapped[bool] = mapped_column(Boolean, default=True)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    date_modification: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<SearchConfig(id={self.id_config}, mot_cle='{self.mot_cle}', type='{self.type_source}')>"


class Source(Base):
    """Sources de données (API, scraping, fichier)."""

    __tablename__ = "sources"

    id_source: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nom: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    url_base: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    est_active: Mapped[bool] = mapped_column(Boolean, default=True)
    derniere_collecte: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    articles: Mapped[list[Article]] = relationship(back_populates="source")

    def __repr__(self) -> str:
        return f"<Source(id={self.id_source}, nom='{self.nom}', type='{self.type}')>"


class Article(Base):
    """Articles collectés et analysés."""

    __tablename__ = "articles"

    id_article: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    id_source: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sources.id_source", ondelete="SET NULL"), nullable=True
    )

    # Contenu
    titre: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    contenu: Mapped[str | None] = mapped_column(Text, nullable=True)
    resume: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Métadonnées
    auteur: Mapped[str | None] = mapped_column(String(200), nullable=True)
    date_publication: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    langue: Mapped[str] = mapped_column(String(10), default="en")

    # Résultats IA
    est_green_it: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score_confiance: Mapped[float | None] = mapped_column(Float, nullable=True)
    modele_classification: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Audit
    chemin_donnees_brutes: Mapped[str | None] = mapped_column(Text, nullable=True)
    date_analyse: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    date_modification: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    source: Mapped[Source | None] = relationship(back_populates="articles")
    logs_analyse: Mapped[list[AnalysisLog]] = relationship(back_populates="article")

    def __repr__(self) -> str:
        return f"<Article(id={self.id_article}, titre='{self.titre[:50]}...')>"


class User(Base):
    """Utilisateurs de la plateforme."""

    __tablename__ = "users"

    id_utilisateur: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    mot_de_passe_hash: Mapped[str] = mapped_column(String(1024), nullable=False)
    est_actif: Mapped[bool] = mapped_column(Boolean, default=True)
    est_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    est_verifie: Mapped[bool] = mapped_column(Boolean, default=False)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    date_modification: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id_utilisateur}, email='{self.email}')>"


class DailyStats(Base):
    """Statistiques quotidiennes agrégées."""

    __tablename__ = "daily_stats"

    id_stats: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_stat: Mapped[datetime] = mapped_column(DateTime, nullable=False, unique=True)
    total_articles: Mapped[int] = mapped_column(Integer, default=0)
    articles_green_it: Mapped[int] = mapped_column(Integer, default=0)
    articles_non_green_it: Mapped[int] = mapped_column(Integer, default=0)
    score_confiance_moyen: Mapped[float | None] = mapped_column(Float, nullable=True)
    articles_par_source: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<DailyStats(date={self.date_stat}, total={self.total_articles})>"


class AnalysisLog(Base):
    """Logs d'analyse IA (traçabilité des inférences)."""

    __tablename__ = "analysis_logs"

    id_log: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_article: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("articles.id_article", ondelete="CASCADE"), nullable=True
    )
    nom_modele: Mapped[str] = mapped_column(String(100), nullable=False)
    version_modele: Mapped[str | None] = mapped_column(String(50), nullable=True)
    temps_inference_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    emissions_carbone_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confiance: Mapped[float | None] = mapped_column(Float, nullable=True)
    date_creation: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    article: Mapped[Article | None] = relationship(back_populates="logs_analyse")

    def __repr__(self) -> str:
        return f"<AnalysisLog(id={self.id_log}, modele='{self.nom_modele}')>"
