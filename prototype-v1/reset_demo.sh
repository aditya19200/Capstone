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
# button is pressed. Note that a restart does NOT put the baseline back by
# itself -- startup honours whatever model_versions has flagged active, which
# is the point of the rollback mechanism. Deleting the retrained rows here is
# what returns the demo to the baseline.
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
# No f-strings here on purpose: this program is inside shell single quotes, so
# escaping the quotes around a dict key (f"{d[\"k\"]}") hands Python a literal
# backslash and it dies with a SyntaxError instead of printing the readout.
# /stats/dashboard is role-gated, so without the header this reads back
# {"detail": "Role 'None' is not permitted here"} and KeyErrors.
curl -s -H "X-Role: admin" -H "X-User-Id: reset" http://localhost:8000/stats/dashboard \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("    documents_uploaded    {}".format(d["documents_uploaded"]))
print("    annotations_validated {}".format(d["annotations_validated"]))'
curl -s http://localhost:8000/health \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("    model serving         {}".format(d["model_version"]))
print("    supabase              {}".format(d["data_layer"]["supabase"]))
print("    neo4j                 {}".format(d["data_layer"]["neo4j"]))'

# The Admin version dropdown is model_versions straight from the database, so
# a rehearsal leaves its retrained version sitting in it. One stale entry is
# a wrong thing to click under pressure and several look careless, so the
# count is part of the readout rather than something to remember to check.
# "serving" is the version whose weights are resident; "active" is what the
# database says should be. On a clean reset one row is both.
curl -s -H "X-Role: admin" -H "X-User-Id: reset" http://localhost:8000/admin/metrics \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
v = d.get("model_versions", [])
print("    versions listed       {}".format(len(v)))
for row in v:
    flags = [n for n, on in (("serving", row.get("is_serving")),
                             ("active", row.get("is_active"))) if on]
    print("      - {}  {}".format(row.get("version_number"),
                                  ", ".join(flags) or "inactive"))
if len(v) != 1:
    print("    ^ expected exactly 1 (the baseline). Anything else is left over.")'

echo
echo "Ready. annotations_validated should read 55, and one version should be"
echo "listed -- Baseline, serving and active. If so, the demo will run"
echo "identically to last time."
