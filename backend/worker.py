"""
worker.py — Entry-point for the prediction worker process.

Run with:
    python worker.py

This simply delegates to the prediction_worker module so the worker can be
started from the project root without needing -m syntax.
"""

from workers.prediction_worker import run_worker

if __name__ == "__main__":
    run_worker()
