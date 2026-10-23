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
# Usage:   ./run_local_demo.sh            start, keeping whatever data is there
#          ./run_local_demo.sh --reset    wipe the demo data first, then start,
#                                         then re-add the 55 confirmations
#                                         Retrain needs (reset_demo.py, then
#                                         seed_retrain_data.py, done for you)
# Stop:    Ctrl+C (stops everything it started)

set -e
cd "$(dirname "$0")"

RESET=0
for arg in "$@"; do
    case "$arg" in
        --reset) RESET=1 ;;
        *) echo "Unknown option: $arg  (the only one is --reset)" >&2; exit 2 ;;
    esac
done

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

DOMAIN_LINKS="legal=http://localhost:5180"
MEDICAL=0
if [ -d backend/models/medical-progression/v1_round1 ]; then
    MEDICAL=1
    DOMAIN_LINKS="legal=http://localhost:5180,medical=http://localhost:5181"
fi

# Wait until a backend reports its model loaded, then print its startup checks.
wait_for() {
    local port=$1 log=$2 name=$3
    for i in $(seq 1 60); do
        if curl -s "http://localhost:$port/health" 2>/dev/null | grep -q '"model_loaded":true'; then
            echo "$name ready."
            # Surface the startup checks that otherwise only appear in the log.
            # Whether the ontology check loaded decides whether confidently-wrong
            # predictions reach the review queue at all.
            grep -h "^\[startup\]" "$log" 2>/dev/null | sed 's/^/  /'
            return 0
        fi
        sleep 1
    done
    echo "WARNING: $name did not report a loaded model within 60s — see $log" >&2
}

if [ "$RESET" = 1 ]; then
    # Before any backend starts: the medical one re-registers its versions at
    # startup, and the legal one must boot the baseline, not a version a
    # previous rehearsal activated.
    echo "[reset] Clearing the demo data ..."
    if ! (cd backend && venv/bin/python reset_demo.py --yes); then
        echo "[reset] reset_demo.py failed — nothing was started. See the output above." >&2
        exit 1
    fi
fi

echo "[1/3] Starting the LEGAL backend on http://localhost:8000 ..."
# RUN_WORKERS_IN_PROCESS: once Supabase is connected the workers would
# otherwise expect separate processes, which this script doesn't start —
# uploads would sit at 0%. See config/settings.py. No effect on the mock.
(cd backend && RUN_WORKERS_IN_PROCESS=true venv/bin/python -m uvicorn main:app --port 8000) > backend.log 2>&1 &
PIDS+=($!)

if [ "$MEDICAL" = 1 ]; then
    # Medical shares Supabase only once migration 010 is applied. Before it,
    # the job queues (batches, xai_jobs, retrain_jobs) carry no domain and two
    # workers polling one database would claim each other's work, so medical
    # gets its own in-memory store instead. The check is the same one the
    # repositories use, so the launcher and the code cannot disagree.
    if (cd backend && venv/bin/python -c "import sys; sys.path.insert(0, '.'); \
from repositories._client import domain_queues_available as ok; sys.exit(0 if ok() else 1)") \
        >/dev/null 2>&1; then
        echo "[2/3] Starting the MEDICAL backend on http://localhost:8001 (Supabase, migration 010 present) ..."
        MEDICAL_DB=()
    else
        echo "[2/3] Starting the MEDICAL backend on http://localhost:8001 (in-memory: migration 010 not applied) ..."
        MEDICAL_DB=(SUPABASE_URL=https://placeholder.supabase.co SUPABASE_SERVICE_KEY=placeholder_key)
    fi
    (cd backend && env "${MEDICAL_DB[@]}" ACTIVE_DOMAIN=medical MODEL_PATH=./models/medical-progression/v1_round1 \
        RUN_WORKERS_IN_PROCESS=true venv/bin/python -m uvicorn main:app --port 8001) \
        > backend-medical.log 2>&1 &
    PIDS+=($!)
fi

echo "Waiting for the models to load..."
wait_for 8000 backend.log "Legal backend"

if [ "$RESET" = 1 ]; then
    echo "[reset] Adding the 55 confirmations Retrain needs (log: seed.log) ..."
    if (cd backend && venv/bin/python ../seed_retrain_data.py) > seed.log 2>&1; then
        tail -n 3 seed.log | sed 's/^/  /'
    else
        echo "[reset] WARNING: seeding did not finish — see seed.log for the command that resumes it." >&2
    fi
fi
[ "$MEDICAL" = 1 ] && wait_for 8001 backend-medical.log "Medical backend"

echo "[3/3] Starting the frontend ..."
if [ "$MEDICAL" = 1 ]; then
    # VITE_* in the environment beats frontend/.env, so the same source tree
    # serves both tabs, each pointed at its own backend.
    (cd frontend && VITE_API_BASE_URL=http://localhost:8001 VITE_DOMAIN_LINKS="$DOMAIN_LINKS" \
        npm run dev -- --port 5181 --strictPort) > frontend-medical.log 2>&1 &
    PIDS+=($!)
fi
echo ""
echo "=========================================="
echo " Open this in your browser: http://localhost:5180"
[ "$MEDICAL" = 1 ] && echo " Switch domains with the Legal / Medical toggle in the header."
echo " Press Ctrl+C here to stop everything."
echo "=========================================="
echo ""
(cd frontend && VITE_DOMAIN_LINKS="$DOMAIN_LINKS" npm run dev -- --port 5180 --strictPort)
