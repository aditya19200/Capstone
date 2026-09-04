"""
repositories/_client.py — Supabase client singleton.

All repository modules import get_client() from here.
When SUPABASE_URL still contains 'placeholder', is_configured() returns False
and repositories fall back to mock_db so the dev server works without credentials.
"""

from typing import Optional

from config.settings import settings


def is_configured() -> bool:
    """Return True when real Supabase credentials are present."""
    return "placeholder" not in settings.SUPABASE_URL


def get_client():
    """
    Return a Supabase client initialised with the service-role key.

    Raises RuntimeError if called when Supabase is not configured.
    Repositories check is_configured() before calling this.
    """
    if not is_configured():
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env"
        )
    from supabase import create_client  # deferred — not installed in dev without real creds
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
