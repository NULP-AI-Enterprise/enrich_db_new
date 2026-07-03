from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql://advertising_user:advertising_pass@localhost:5432/advertising_db"

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    use_mock_llm: bool = False
    use_mock_search: bool = False
    batch_embed_size: int = 100     # items per embedding API call
    batch_embed_interval: int = 30  # Celery beat fallback interval (seconds)
    scrape_timeout: float = 5.0
    max_retries: int = 3
    producer_page_size: int = 500   # rows per DB page when dispatching
    task_stagger_per: int = 50      # add 1s countdown per N tasks (thundering herd)

    # ── Optional external search APIs ────────────────────────────────────────
    # Google Custom Search  (https://programmablesearchengine.google.com)
    google_api_key: str = ""
    google_cx: str = ""          # Custom Search Engine ID
    # SerpAPI  (https://serpapi.com) – simpler alternative to Google CSE
    serp_api_key: str = ""

    # ── Rate-limit courtesies ─────────────────────────────────────────────────
    llm_rate_limit_delay: float = 0.5   # seconds between LLM calls per worker
    scrape_rate_limit_delay: float = 0.2
    concurrency: int = 4                 # parallel items in run_production.py


settings = Settings()
