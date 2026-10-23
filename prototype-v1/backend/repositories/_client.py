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


def current_domain() -> str:
    """
    Key of the domain pack this process serves (e.g. "legal", "medical").

    Migration 007 added a `domain` column to predictions and annotations
    because one database can hold rows from several label sets, and a
    label_id only means something relative to its pack. Writes stamp it and
    list/count queries filter on it, so a medical backend never sees — or
    retrains on — legal rows sharing the same database.

    Uses active_domain() rather than settings.ACTIVE_DOMAIN so that the
    fallback it applies for a mistyped key is respected here too.
    """
    from config.domains import active_domain
    return active_domain().key


_domain_queues: Optional[bool] = None


def domain_queues_available() -> bool:
    """
    Whether migration 010 has been applied: a domain on batches, retrain_jobs
    and model_versions, and the domain-scoped claim / activate functions.

    Every repository that touches a job queue or the active-version flag asks
    this first. Before 010 the answer is False and each one behaves exactly as
    it always has, which is safe for one domain per database and is why the
    medical instance is kept in memory until then. After 010 the answer is
    True and each one scopes itself to current_domain(), so two domains can
    share Supabase without claiming each other's work.

    Probed once per process by selecting the new column; a missing column is
    a PostgREST error, not an empty result. Mock mode answers False: the
    in-memory store belongs to one process and so to one domain already.
    """
    global _domain_queues
    if _domain_queues is not None:
        return _domain_queues
    if not is_configured():
        _domain_queues = False
        return _domain_queues
    try:
        get_client().table("batches").select("domain").limit(1).execute()
        _domain_queues = True
    except Exception:
        _domain_queues = False
    return _domain_queues
