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

    # --- Domain ---
    # Which domains/<key>.json supplies the label set. See config/domains.py.
    ACTIVE_DOMAIN: str = "legal"

    # --- Model ---
    MODEL_PATH: str = "./models/LegalModelShared"
    MAX_LENGTH: int = 256           # tokenizer truncation — must match training

    # --- Supabase ---
    SUPABASE_URL: str = "https://placeholder.supabase.co"
    SUPABASE_SERVICE_KEY: str = "placeholder_key"

    # --- Neo4j ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    # Some Aura Free instances use the instance ID as the database name
    # instead of "neo4j". If queries come back empty, check this first.
    NEO4J_DATABASE: str = "neo4j"

    # --- Confidence calibration ---
    # Divides the logits before softmax. 1.0 is off, leaving confidence
    # exactly as the model produced it.
    #
    # Measured on the held-out set (sample-data/evaluation/, 656 passages
    # across 17 documents the model has never seen; the Indian Stamp Act is
    # excluded because at 821 passages it was 56% of the data on its own and
    # its gold label is disputed):
    #
    #     accuracy                     0.845   unchanged by this setting
    #     mean confidence     0.920 -> 0.858
    #     expected calib. error
    #                         0.083 -> 0.060   at T = 1.313
    #
    # In the 0.9-1.0 confidence band the model claims 0.979 and is right
    # 0.890 of the time, so it overstates itself by about nine points where
    # it matters most — that band is above CONFIDENCE_HIGH and is
    # auto-accepted without review.
    #
    # Left at 1.0 deliberately. Dividing by 1.313 cannot change which label
    # wins, so accuracy is untouched, but every confidence drops, and some
    # predictions cross CONFIDENCE_HIGH and start routing to reviewers. That
    # is more correct and it changes the behaviour the thresholds below were
    # tuned against, so it is a deliberate switch rather than a default.
    CALIBRATION_TEMPERATURE: float = 1.0

    # --- Active Learning thresholds ---
    CONFIDENCE_HIGH: float = 0.85   # above this → AUTO_ACCEPT
    CONFIDENCE_LOW: float = 0.55    # below this → ROUTE_TO_REVIEWER
    # between LOW and HIGH → NEEDS_EXPLANATION

    # --- Human review ---
    # confidence < this → in the Review queue. Equal to CONFIDENCE_LOW: at
    # 0.5 a prediction at 0.52 was labelled ROUTE_TO_REVIEWER and never
    # reached the queue (24 Oct rehearsal, a probe misfiled at 0.516).
    REVIEW_THRESHOLD: float = 0.55

    # --- Input validation ---
    MIN_TEXT_CHARS: int = 50        # reject shorter text — model mislabels short strings

    # --- Worker / inference ---
    # Run the batch / XAI / retrain workers as threads inside the API process
    # even when a real database is configured. Without a database they always
    # run in-process (see local_workers.py). With one, the default is that
    # they run as separate `python -m workers.X` processes — which
    # run_local_demo.sh doesn't start, so it sets this to true. Safe to
    # combine with separate workers: both queues claim with SKIP LOCKED.
    RUN_WORKERS_IN_PROCESS: bool = False
    BATCH_SIZE: int = 16
    POLL_INTERVAL: int = 5          # seconds between queue polls (classify worker)

    # --- Stuck-job recovery ---
    STUCK_JOB_MINUTES: int = 5      # processing rows older than this are reset
    MAX_ATTEMPTS: int = 3           # after this many retries, row → failed

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,          # env var names are upper-case by convention
        extra="ignore",               # silently ignore unknown env vars
    )


# Single shared instance — import this everywhere
settings = Settings()
