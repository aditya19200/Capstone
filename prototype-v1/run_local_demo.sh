#!/usr/bin/env bash
# run_local_demo.sh — start the whole system locally for a demo.
#
# Starts: backend API (which also runs the batch + SHAP-explain workers as
# background threads internally — see backend/local_workers.py for why)
# and the frontend. Everything runs on this machine only — no internet, no
# cloud accounts. Data lives in memory and resets whenever the backend
# restarts.
#
# Usage:   ./run_local_demo.sh
# Stop:    Ctrl+C (stops everything it started)

set -e
cd "$(dirname "$0")"

# Use the newer Node.js installed via nvm (system Node 18 can't run this frontend)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use default >/dev/null

PIDS=()
cleanup() {
    echo ""
    echo "Stopping everything..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting backend API on http://localhost:8000 (batch + explain workers run inside it) ..."
(cd backend && venv/bin/python -m uvicorn main:app --port 8000) > backend.log 2>&1 &
PIDS+=($!)

echo "Waiting for the model to load..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q '"model_loaded":true'; then
        echo "Backend ready."
        break
    fi
    sleep 1
done

echo "[2/2] Starting frontend on http://localhost:5180 ..."
echo ""
echo "=========================================="
echo " Open this in your browser: http://localhost:5180"
echo " Press Ctrl+C here to stop everything."
echo "=========================================="
echo ""
(cd frontend && npm run dev -- --port 5180)
