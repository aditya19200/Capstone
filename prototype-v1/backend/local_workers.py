"""
local_workers.py — runs the batch/XAI/retrain worker loops as background
threads inside the API process, for local development without a real
Postgres/Supabase instance.

Why this exists: services/mock_db.py is an in-memory Python dict, scoped to
a single process. In production, batch_worker / xai_worker / retrain_worker
run as separate `python -m workers.X` processes talking to real Postgres —
genuinely shared state, so that's fine. Locally, with no real database,
running them as separate OS processes means each one has its own empty copy
of mock_db and can never see what the API process inserted (a batch stays
at 0% forever, an explain job never starts). Running the same loops as
threads inside this process instead makes them see the same in-memory
state, so the local demo actually works end to end.

Only activated when repositories._client.is_configured() is False — real
deployments keep the intended separate-process architecture untouched.
"""

import logging
import threading

logger = logging.getLogger(__name__)


def _run_batch_loop() -> None:
    try:
        from workers import batch_worker
        batch_worker.main()
    except Exception:
        logger.exception("[local_workers] batch worker thread crashed")


def _run_xai_loop() -> None:
    try:
        from workers import xai_worker
        xai_worker.run_worker()
    except Exception:
        logger.exception("[local_workers] xai worker thread crashed")


def _run_retrain_loop() -> None:
    try:
        from workers import retrain_worker
        retrain_worker.main()
    except Exception:
        logger.exception("[local_workers] retrain worker thread crashed")


def start_local_workers() -> None:
    """Start the batch, XAI, and retrain worker polling loops as daemon threads."""
    logger.info(
        "[local_workers] No real database configured — running batch/XAI/"
        "retrain workers in-thread instead of as separate processes."
    )
    threading.Thread(target=_run_batch_loop, daemon=True, name="local-batch-worker").start()
    threading.Thread(target=_run_xai_loop, daemon=True, name="local-xai-worker").start()
    threading.Thread(target=_run_retrain_loop, daemon=True, name="local-retrain-worker").start()
