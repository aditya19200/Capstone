#!/usr/bin/env bash
# Put the demo back to its starting state.
#
# One command, one known state, every time:
#
#     55 validated annotations   (the confirmations that clear the 50 gate)
#      0 batches, 0 predictions  (nothing left from the last run)
#      0 retrained versions      (no v2, no v3 -- only the baseline)
#      graph untouched           (it is loaded from files, nothing to reset)
#
# Run it before a rehearsal and again before the real thing. Between those
# two points nothing changes by itself: no retrain happens unless the Retrain
# button is pressed, and the served model goes back to the baseline on every
# backend restart because startup loads settings.MODEL_PATH rather than
# whatever the database has flagged active.
#
#     ./reset_demo.sh          asks before deleting
#     ./reset_demo.sh --yes    no prompt
#
# The backend must be running, because the seed step talks to it over HTTP.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! curl -s --max-time 5 http://localhost:8000/health >/dev/null 2>&1; then
    echo "Backend is not answering on :8000 — start ./run_local_demo.sh first." >&2
    exit 1
fi

echo "==> Clearing the last run"
(cd backend && ./venv/bin/python reset_demo.py "$@")

echo
echo "==> Seeding the 55 confirmations"
python3 seed_retrain_data.py

echo
echo "==> Starting state"
curl -s http://localhost:8000/stats/dashboard \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(
    f"    documents_uploaded    {d[\"documents_uploaded\"]}\n"
    f"    annotations_validated {d[\"annotations_validated\"]}")'
curl -s http://localhost:8000/health \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(
    f"    model serving         {d[\"model_version\"]}\n"
    f"    supabase              {d[\"data_layer\"][\"supabase\"]}\n"
    f"    neo4j                 {d[\"data_layer\"][\"neo4j\"]}")'

echo
echo "Ready. annotations_validated should read 55 and the model should read"
echo "baseline. If it does, the demo will run identically to last time."
