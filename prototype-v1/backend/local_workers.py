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
import time

logger = logging.getLogger(__name__)


# A worker loop that raises used to take its thread down for good: the
# exception was logged and the thread exited. On 23 Oct a DNS blip
# ("Temporary failure in name resolution") killed the batch and XAI threads
# inside a minute of each other, and nothing processed another job for the
# rest of that run -- uploads simply sat at 0% with the cause buried in the
# log. The network coming back did not help, because there was no longer
# anything polling.
#
# So a crash is now a restart, with a backoff so a persistent failure (bad
# credentials, say) does not spin. A loop that has been healthy for a while
# before falling over starts again from the shortest delay, because that is
# a blip rather than a broken configuration.
_RESTART_DELAYS = (2, 5, 15, 30, 60)
_HEALTHY_SECONDS = 60


def _supervise(name: str, run) -> None:
    attempt = 0
    while True:
        started = time.monotonic()
        try:
            run()
            logger.info("[local_workers] %s loop returned; not restarting.", name)
            return
        except Exception:
            ran_for = time.monotonic() - started
            if ran_for >= _HEALTHY_SECONDS:
                attempt = 0
            delay = _RESTART_DELAYS[min(attempt, len(_RESTART_DELAYS) - 1)]
            logger.exception(
                "[local_workers] %s worker crashed after %.0fs — restarting in %ds",
                name, ran_for, delay,
            )
            attempt += 1
            time.sleep(delay)


def _run_batch_loop() -> None:
    from workers import batch_worker
    _supervise("batch", batch_worker.main)


def _run_xai_loop() -> None:
    from workers import xai_worker
    _supervise("xai", xai_worker.run_worker)


def _run_retrain_loop() -> None:
    from workers import retrain_worker
    _supervise("retrain", retrain_worker.main)


def start_local_workers() -> None:
    """Start the batch, XAI, and retrain worker polling loops as daemon threads."""
    logger.info(
        "[local_workers] Running batch/XAI/retrain workers in-thread rather "
        "than as separate processes."
    )
    # These loops share the API's model instance, so their shutdown must not
    # unload it. See ModelService.release().
    from services.model_service import model_service
    model_service.mark_shared()

    threading.Thread(target=_run_batch_loop, daemon=True, name="local-batch-worker").start()
    threading.Thread(target=_run_xai_loop, daemon=True, name="local-xai-worker").start()
    threading.Thread(target=_run_retrain_loop, daemon=True, name="local-retrain-worker").start()
