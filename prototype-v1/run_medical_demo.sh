#!/usr/bin/env bash
# run_medical_demo.sh — start ONLY the medical stack.
#
# You normally do not need this: ./run_local_demo.sh now starts both domains
# and the header has a Legal / Medical switch. Kept for running medical alone.
#
# Runs alongside ./run_local_demo.sh rather than replacing it:
#
#     legal    backend :8000   frontend :5180
#     medical  backend :8001   frontend :5181
#
# Two browser tabs, no restart, no switching. That matters during a demo —
# ACTIVE_DOMAIN is read at import time (services/model_service.py binds
# ID2LABEL/NUM_LABELS at module level), so changing domains in one process
# means restarting it and reloading the weights, which is dead air you do
# not want in a twenty-minute slot.
#
# Sharing one database is safe for PREDICTIONS: migration 007 added
# predictions.domain, every insert stamps the active domain and every list
# query filters on it, so medical uploads never appear in the legal review
# queue.
#
# It is NOT safe for model_versions, which has no domain column and one
# global single-active index. So on the medical tab:
#
#     use    Annotate, Review, Ontology
#     avoid  Admin  -- its version table is the legal row, which will read
#            "Active - restart pending" because the medical weights are what
#            is actually loaded
#     never  Retrain -- it would insert into the shared table and deactivate
#            the legal baseline, and the legal backend would then try to load
#            medical weights on its next restart
#
# Run Admin and Retrain on the legal tab (:5180). The medical backend also
# logs one WARNING at startup -- it reads the active row, finds legal weights,
# fails to load them into a 12-label process and falls back to MODEL_PATH.
# That is the designed fallback, not a fault.
#
# Usage:   ./run_medical_demo.sh        (start ./run_local_demo.sh first)
# Stop:    Ctrl+C

set -e
cd "$(dirname "$0")"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use default >/dev/null

if [ ! -d backend/models/MedicalModelShared ]; then
    echo "backend/models/MedicalModelShared is missing — the medical model is not on this machine." >&2
    exit 1
fi

PIDS=()
cleanup() {
    echo ""
    echo "Stopping the medical stack..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "[1/2] Starting MEDICAL backend on http://localhost:8001 ..."
# Shared Supabase only once migration 010 is applied; in memory before it.
# See run_local_demo.sh for why.
MEDICAL_DB=(SUPABASE_URL=https://placeholder.supabase.co SUPABASE_SERVICE_KEY=placeholder_key)
if (cd backend && venv/bin/python -c "import sys; sys.path.insert(0, '.'); \
from repositories._client import domain_queues_available as ok; sys.exit(0 if ok() else 1)") \
    >/dev/null 2>&1; then
    MEDICAL_DB=()
fi
(cd backend && env "${MEDICAL_DB[@]}" ACTIVE_DOMAIN=medical MODEL_PATH=./models/MedicalModelShared \
    RUN_WORKERS_IN_PROCESS=true venv/bin/python -m uvicorn main:app --port 8001) \
    > backend-medical.log 2>&1 &
PIDS+=($!)

echo "Waiting for the model to load..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8001/health 2>/dev/null | grep -q '"model_loaded":true'; then
        echo "Backend ready."
        grep -h "^\[startup\]" backend-medical.log 2>/dev/null | sed 's/^/  /'
        break
    fi
    sleep 1
done

echo "[2/2] Starting MEDICAL frontend on http://localhost:5181 ..."
echo ""
echo "=========================================="
echo " Medical tab:  http://localhost:5181"
echo " Legal tab:    http://localhost:5180  (./run_local_demo.sh)"
echo " Press Ctrl+C here to stop the medical stack only."
echo "=========================================="
echo ""
(cd frontend && VITE_API_BASE_URL=http://localhost:8001 npm run dev -- --port 5181)
