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
    # `Qwen/Qwen3-4B` (licence Apache-2.0, publie le 26 juillet 2025) est le
    # LLM texte officiel 4B d'Alibaba, pleinement supporte par `transformers`
    # comme `AutoModelForCausalLM` et `AutoModelForSequenceClassification`.
    # Choisi comme base du pipeline d'entrainement pour trois raisons :
    #   1. Multilingue natif (FR/EN/DE/ES/ZH) : les articles techniques scrapes
    #      depuis des sources non anglophones sont traites sans etape de
    #      traduction, directement exploites par la classification.
    #   2. Architecture dense transformer standard : entrainement LoRA K-fold
    #      tenable sur RX 7900 XTX 24 Go (~14 Go VRAM avec adaptateurs r=16 +
    #      AdamW + gradient checkpointing, batch 2 + grad_accum 4). Inference
    #      ~0.4 s/article en BF16 sur le meme GPU.
    #   3. Chat template Qwen aligne sur `Qwen3-4B-Instruct-2507` deja utilise
    #      pour les summarizers et le LLM judge : une seule famille a maintenir.
    # Ce modele remplace l'ancien `meta-llama/Llama-3.2-3B` gated (besoin de
    # demande d'acces HF) comme base du modele fine-tune sur le golden
    # dataset et promu en production par `scripts/retrain_pipeline.py`.
    #
    # Note : la tentative precedente avec `Qwen/Qwen3.5-4B` a echoue car il
    # s'agit en realite d'un VLM (image-text-to-text, ~4.66B parametres dont
    # ~500 Mo de blocs visuels inutiles), avec une architecture a attention
    # lineaire hybride necessitant `flash-linear-attention` + `causal-conv1d`
    # non disponibles sous ROCm. Le fallback torch pur saturait la VRAM et
    # gelait le systeme au premier step d'entrainement.
    huggingface_model_trainer_base: str = "Qwen/Qwen3-4B"
    # Meme modele utilise comme baseline : evalue zero-shot (sans fine-tuning)
    # sur l'integralite du dataset annote pour mesurer le gain apporte par
    # l'entrainement LoRA. Avoir la meme base en baseline et fine-tuning permet
    # de comparer strictement l'impact du fine-tuning, sans bruit lie au
    # changement d'architecture.
    huggingface_model_baseline: str = "Qwen/Qwen3-4B"
    # Encoder concurrent du benchmark equitable B4 (avril-mai 2026) :
    # `microsoft/mdeberta-v3-base` (278M params, encoder-only, DisentangledSelfAttention,
    # licence MIT). Choisi comme alternative au `Qwen/Qwen3-4B` decoder pour mettre
    # en competition deux architectures sur le golden dataset bilingue EN/FR.
    # mDeBERTa est selectionne (et pas `deberta-v3-base` EN-only) parce que le
    # dataset final contient 25.25 % d'articles FR (dont 600 Green IT francais
    # issus principalement de GreenIT.fr) - un encoder EN-only encoderait mal ces
    # signaux et fausserait le benchmark en faveur de Qwen3. Cette base sert :
    #   1. Au benchmark BRUT zero-shot (pre-entrainement) via `benchmark_baseline.py`
    #   2. Au benchmark FINE-TUNE via la classe `MDeBERTaClassifier` du protocole
    #      unifie B3 (K-fold 5 x 3 seeds, stratification langue x label, class_weight,
    #      back-translation EN<->FR, calibration temperature+seuil, ensemble
    #      logit-average)
    # Decision documentee dans `docs/CHOIX_DEBERTA.md` (a rediger apres B4.4).
    huggingface_model_encoder_base: str = "microsoft/mdeberta-v3-base"
    # Longueur maximale des sequences tokenizees lors de l'entrainement. 512
    # tokens couvrent la majorite des articles du corpus (les plus longs
    # tronquent leur queue peu informative) et divisent par ~4 la consommation
    # memoire de l'attention par rapport a 1024, ce qui rend le LoRA K-fold
    # stable sur 24 Go de VRAM avec gradient checkpointing actif.
    trainer_max_length: int = 512
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
    # Legacy : cle NewsData.io, conservee pour historique mais la source est
    # desactivee depuis le 16 avril 2026 (contenu tronque "ONLY AVAILABLE IN
    # PAID PLANS" en free tier, dataset pourri).
    api_news_key: str = ""
    # Cle API The Guardian Open Platform (tier Developer : 5000 req/jour,
    # 12 req/s, non-commercial). Remplace NewsData.io comme source REST/JSON
    # principale. Obtention gratuite et immediate sur
    # https://open-platform.theguardian.com/access/
    guardian_api_key: str = ""
    # Dev.to / Forem API : aucune cle necessaire en lecture publique, on
    # garde une variable optionnelle pour passer en write API plus tard.
    devto_api_key: str = ""
    # Email utilise pour le Polite Pool de Crossref (api.crossref.org).
    # Fournir un contact permet a Crossref de prioriser nos requetes sur
    # le pool public (latences plus stables, meilleur rate limit). Non
    # sensible : publie dans le User-Agent de chaque requete. Ref:
    # https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/
    crossref_mailto: str = ""

    # --- Celery / Redis (file d'attente analyses) ---
    # Redis sert a la fois de broker (queue des messages) et de result backend
    # (stockage des resultats). Deux DB Redis distincts pour bien separer les
    # responsabilites et faciliter les purges ciblees (FLUSHDB) :
    #   - DB 0 : broker (queue des taches enqueueed)
    #   - DB 1 : backend (etat + resultat de chaque tache)
    # En prod, on peut pointer les deux URL vers des instances Redis distinctes
    # pour isoler la latence broker vs backend.
    redis_url: str = "redis://localhost:6379"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    # Retention des resultats Celery dans Redis (en secondes). 24h par defaut :
    # un utilisateur qui lance une analyse a largement le temps de revenir
    # consulter le statut, et au-dela on libere de la memoire Redis.
    celery_result_expires: int = 86400
    # Timeout d'execution d'une tache de classification. Pipeline complet
    # (extraction + summarize + classify + summarize_green) ~10-30s en mode
    # normal, jusqu'a 60-120s si fallback LLM local cold start. On laisse
    # une marge pour eviter les timeouts intempestifs.
    celery_task_time_limit: int = 600

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
