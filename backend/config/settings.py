"""
config/settings.py — Application settings via Pydantic BaseSettings.

Reads all configuration from environment variables (or a .env file).
Import `settings` wherever config values are needed — never hardcode thresholds.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config object.  All values are read from the environment / .env file.
    Pydantic automatically casts strings to the declared types.
    """

    # --- Model ---
    MODEL_PATH: str = "./models/LegalModelShared"

    # --- Supabase ---
    SUPABASE_URL: str = "https://placeholder.supabase.co"
    SUPABASE_KEY: str = "placeholder_key"

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # --- Active Learning thresholds ---
    CONFIDENCE_HIGH: float = 0.85   # above this → AUTO_ACCEPT
    CONFIDENCE_LOW: float = 0.55    # below this → ROUTE_TO_REVIEWER
    # between LOW and HIGH → NEEDS_EXPLANATION

    # --- Worker / inference ---
    BATCH_SIZE: int = 8

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,          # env var names are upper-case by convention
        extra="ignore",               # silently ignore unknown env vars
    )


# Single shared instance — import this everywhere
settings = Settings()
