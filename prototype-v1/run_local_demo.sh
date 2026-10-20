#!/usr/bin/env bash
# run_local_demo.sh — start the whole system locally for a demo.
#
# Starts: backend API (which also runs the batch + SHAP-explain workers as
# background threads internally — see backend/local_workers.py for why)
# and the frontend.
#
# Data: with no backend/.env (or placeholder credentials) everything runs on
# this machine only — no internet, no cloud accounts — and data lives in
# memory, resetting whenever the backend restarts. With real Supabase / Neo4j
# credentials in backend/.env, data persists there instead. Check which one
# you're on with:  cd backend && ./venv/bin/python check_data_layer.py
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
# RUN_WORKERS_IN_PROCESS: once Supabase is connected the workers would
# otherwise expect separate processes, which this script doesn't start —
# uploads would sit at 0%. See config/settings.py. No effect on the mock.
(cd backend && RUN_WORKERS_IN_PROCESS=true venv/bin/python -m uvicorn main:app --port 8000) > backend.log 2>&1 &
PIDS+=($!)

echo "Waiting for the model to load..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q '"model_loaded":true'; then
        echo "Backend ready."
        # Surface the startup checks that otherwise only appear in backend.log.
        # Whether the ontology check loaded decides whether confidently-wrong
        # predictions reach the review queue at all, so it should not be
        # something you have to go and grep for.
        grep -h "^\[startup\]" backend.log 2>/dev/null | sed 's/^/  /'
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
