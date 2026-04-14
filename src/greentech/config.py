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
    # Architecture mono-modele cote cloud : un seul LLM instructif est utilise
    # pour les deux types de resumes (general et aspects ecologiques) ainsi
    # que pour le LLM judge de classification Green IT. Cela garantit la
    # coherence linguistique (tout en francais), la coherence qualitative et
    # simplifie l'infrastructure (un seul endpoint a monitorer).
    #
    # `Qwen/Qwen3-4B-Instruct-2507` (licence Apache-2.0) est la generation
    # Qwen3 la plus recente compatible avec les HF Inference Providers au
    # 14 avril 2026. Choix du 4B plutot que 7B : meilleur equilibre qualite /
    # empreinte (~8 Go FP16) et le 3B/1.5B ne sont pas servis par les
    # providers actifs du compte. Le 4B sert donc de reference qualitative
    # pour la chaine d'inference.
    huggingface_model_summarizer: str = "Qwen/Qwen3-4B-Instruct-2507"
    # Resume ecologique : meme modele que le general. Deux appels paralleles
    # avec des prompts distincts plutot qu'un seul appel JSON pour rester
    # robuste aux erreurs individuelles (si un prompt echoue, l'autre sort
    # quand meme un resume).
    huggingface_model_green_summarizer: str = "Qwen/Qwen3-4B-Instruct-2507"
    huggingface_model_classifier: str = "microsoft/deberta-v3-base"
    # LLM judge pour l'etage 2 de classification Green IT : verifie les
    # articles marques CANDIDATE par le pre-filtre mots-cles. Meme modele
    # Qwen3-4B que les summarizers : un seul service HF a maintenir.
    huggingface_model_classifier_llm: str = "Qwen/Qwen3-4B-Instruct-2507"
    # === Classifieur entraine en interne (baseline + fine-tuning LoRA) ===
    # `Qwen/Qwen3.5-4B` (licence Apache-2.0, 27 fevrier 2026) est la generation
    # Qwen la plus recente disponible sur le Hub au 14 avril 2026. Choisi
    # comme base du pipeline d'entrainement pour trois raisons :
    #   1. Multilingue natif (FR/EN/DE/ES/ZH) : les articles techniques scrapes
    #      depuis des sources non anglophones sont traites sans etape de
    #      traduction, directement exploites par la classification.
    #   2. Taille 4B : entrainement LoRA K-fold tenable sur RX 7900 XTX 24 Go
    #      (~14 Go VRAM avec adaptateurs r=16 + AdamW, batch size 4-6).
    #      Inference ~0.4 s/article en BF16 sur le meme GPU.
    #   3. Chat template Qwen stable entre les generations : compatibilite
    #      directe avec le reste du pipeline (summarizer, LLM judge).
    # Ce modele remplace l'ancien `meta-llama/Llama-3.2-3B` gated (besoin de
    # demande d'acces HF) comme base du challenger fine-tune sur le golden
    # dataset et promu en production par `scripts/retrain_pipeline.py`.
    huggingface_model_trainer_base: str = "Qwen/Qwen3.5-4B"
    # Meme modele utilise comme baseline : evalue zero-shot (sans fine-tuning)
    # sur l'integralite du dataset annote pour mesurer le gain apporte par
    # l'entrainement LoRA. Avoir la meme base en baseline et challenger permet
    # de comparer strictement l'impact du fine-tuning, sans bruit lie au
    # changement d'architecture.
    huggingface_model_baseline: str = "Qwen/Qwen3.5-4B"
    # Longueur maximale des sequences tokenizees lors de l'entrainement. 1024
    # tokens couvrent 95% des articles du corpus sans troncature destructive,
    # tout en maintenant un cout memoire raisonnable pour le LoRA K-fold.
    trainer_max_length: int = 1024
    # Fallback local n1 : `Qwen/Qwen2.5-3B-Instruct`. Choisi comme repli
    # principal parce qu'il est le plus proche en taille de parametres du
    # Qwen3-4B cloud (~6 Go FP16 sur GPU), tout en restant chargeable sur
    # des machines plus modestes que la RX 7900 XTX du developpeur. Quand
    # le dispatcher rencontre une erreur HF (402 quota, 400 provider
    # indisponible), il bascule sur ce modele pour la suite de la session.
    huggingface_model_local_fallback: str = "Qwen/Qwen2.5-3B-Instruct"
    # Fallback local n2 : `Qwen/Qwen2.5-1.5B-Instruct`. Auto-active par
    # `llm_local.py` lorsque le preflight memoire ou un OOM indique que la
    # machine ne peut pas heberger le 3B. Occupe ~3 Go en FP16 et reste
    # compatible CPU / GPU integre.
    huggingface_model_local_fallback_lightweight: str = "Qwen/Qwen2.5-1.5B-Instruct"

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
