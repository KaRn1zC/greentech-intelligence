"""Schemas Pydantic pour l'analyse d'articles par IA.

Definit les modeles pour les endpoints /analyze (soumission et suivi).

Redige par KaRn1zC - 2026-03-13
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class AnalysisInput(BaseModel):
    """Schema de requete pour declencher une analyse IA.

    Exactement un des deux champs doit etre fourni : url ou texte.
    """

    url: str | None = Field(None, description="URL de l'article a analyser")
    texte: str | None = Field(None, min_length=50, description="Texte brut a analyser")

    @model_validator(mode="after")
    def valider_entree(self) -> AnalysisInput:
        """Verifie qu'exactement un champ est fourni (url ou texte)."""
        if not self.url and not self.texte:
            msg = "Fournissez une URL ou un texte a analyser."
            raise ValueError(msg)
        if self.url and self.texte:
            msg = "Fournissez soit une URL, soit un texte, pas les deux."
            raise ValueError(msg)
        return self


class AnalysisStatus(StrEnum):
    """Statuts possibles d'un job d'analyse."""

    EN_ATTENTE = "en_attente"
    EN_COURS = "en_cours"
    TERMINE = "termine"
    ERREUR = "erreur"


class AnalysisJobCreated(BaseModel):
    """Reponse apres soumission d'une analyse."""

    job_id: uuid.UUID = Field(..., description="Identifiant unique du job d'analyse")
    statut: AnalysisStatus = Field(
        default=AnalysisStatus.EN_ATTENTE, description="Statut initial"
    )
    message: str = Field(
        default="Analyse soumise avec succes",
        description="Message informatif",
    )


class AnalysisResult(BaseModel):
    """Resultat complet d'une analyse IA."""

    job_id: uuid.UUID
    statut: AnalysisStatus
    id_article: int | None = None
    titre: str | None = None
    est_green_it: bool | None = None
    score_confiance: float | None = None
    resume: str | None = None
    modele_classification: str | None = None
    temps_inference_ms: int | None = None
    date_analyse: datetime | None = None
    erreur: str | None = None
