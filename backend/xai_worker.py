"""
xai_worker.py — Entry-point for the XAI/SHAP worker process.

Run with:
    python xai_worker.py

This delegates to the xai_worker module so the worker can be started from
the project root without needing -m syntax.
"""

from workers.xai_worker import run_worker

if __name__ == "__main__":
    run_worker()
