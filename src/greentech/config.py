"""Configuration centralisée du projet via Pydantic Settings.

Charge automatiquement les variables depuis le fichier .env
situé à la racine du projet.

"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (deux niveaux au-dessus de src/greentech/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Paramètres globaux de l'application GreenTech Intelligence.

    Toutes les valeurs sont chargées depuis les variables d'environnement
    ou le fichier .env. Les valeurs par défaut conviennent au développement local.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "GreenTech Intelligence"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "CHANGE_THIS_TO_A_RANDOM_STRING_IN_PRODUCTION"

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "greentech"
    postgres_password: str = "greentech_dev_password"
    postgres_db: str = "greentech_db"
    postgres_app_user: str = "greentech_app"
    postgres_app_password: str = "greentech_app_password"

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket_raw: str = "raw-data"
    minio_bucket_clean: str = "clean-data"
    minio_bucket_models: str = "models"

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "greentech-classification"

    # --- Hugging Face ---
    huggingface_token: str = ""
    # Architecture mono-modele : un seul LLM instructif est utilise pour les
    # deux types de resumes (general et aspects ecologiques). Cela garantit
    # la coherence linguistique (tout en francais), la coherence qualitative
    # (meme niveau de generation entre les 2 blocs) et simplifie l'infrastructure.
    # `Qwen/Qwen2.5-7B-Instruct` est disponible via HF Inference Providers
    # sans activation ni demande d'acces (licence Apache-2.0).
    huggingface_model_summarizer: str = "Qwen/Qwen2.5-7B-Instruct"
    # Le modele de resume ecologique pointe vers le meme modele : deux appels
    # paralleles avec des prompts distincts (plutot qu'un seul appel JSON pour
    # conserver une implementation simple et robuste aux erreurs individuelles).
    huggingface_model_green_summarizer: str = "Qwen/Qwen2.5-7B-Instruct"
    huggingface_model_classifier: str = "microsoft/deberta-v3-base"
    # LLM judge pour l'etage 2 de classification Green IT : verifie les articles
    # marques CANDIDATE par le pre-filtre mots-cles. Meme modele Qwen instructif
    # que les summarizers pour limiter le nombre de services a maintenir.
    huggingface_model_classifier_llm: str = "Qwen/Qwen2.5-7B-Instruct"
    # Modele utilise en fallback local (GPU AMD ROCm) lorsque le quota mensuel
    # HF Inference Providers est epuise (erreur HTTP 402). On conserve
    # exactement le MEME modele que celui appele via HF (Qwen2.5-7B-Instruct)
    # pour garantir une continuite qualitative totale entre les deux backends :
    # les resumes et les verdicts de classification ont la meme qualite
    # selon que l'on passe par le cloud ou par le GPU local. En FP16 sur la
    # RX 7900 XTX (24 Go VRAM), le 7B occupe environ 14 Go, ce qui laisse
    # une marge confortable pour le contexte et les activations.
    huggingface_model_local_fallback: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Liste des origines CORS autorisees, parsee depuis la chaine.

        Accepte deux formats pour etre robuste aux erreurs de config :
        - CSV : "http://a.com,http://b.com"
        - JSON array : '["http://a.com","http://b.com"]' (format Pydantic historique)
        """
        import json

        raw = self.cors_origins.strip()
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(o).strip() for o in parsed if str(o).strip()]
            except json.JSONDecodeError:
                pass
        return [o.strip() for o in raw.split(",") if o.strip()]

    # --- JWT ---
    jwt_secret_key: str = "CHANGE_THIS_TO_A_RANDOM_STRING"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # --- Data Collection ---
    scraping_user_agent: str = "GreenTech-Bot/1.0"
    scraping_delay_seconds: int = 2
    api_news_key: str = ""

    # --- Monitoring ---
    loki_url: str = "http://localhost:3100"
    prometheus_port: int = 9090
    grafana_port: int = 3000

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """URL de connexion async pour l'utilisateur applicatif."""
        return (
            f"postgresql+asyncpg://{self.postgres_app_user}:{self.postgres_app_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_admin(self) -> str:
        """URL de connexion async pour l'utilisateur admin (migrations, DDL)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne l'instance unique des paramètres (singleton mis en cache).

    Returns:
        Instance Settings chargée depuis l'environnement.
    """
    return Settings()
