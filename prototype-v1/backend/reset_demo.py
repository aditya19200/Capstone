#!/usr/bin/env python3
"""
reset_demo.py — clear the demo data out of the live database.

Why this exists
---------------
On the in-memory store a restart wiped everything, so every rehearsal began
from nothing. Against a live Supabase nothing wipes it, and that changes what
a rehearsal costs: the corrections you made stay validated, the batches you
uploaded stay listed, and the model version you activated stays flagged
active.

That last one is the problem worth naming. Startup always loads
settings.MODEL_PATH rather than whatever the database calls active, so after a
restart the server serves the baseline while model_versions still says v2 is
active. Nothing breaks, but the Admin page and /health disagree, which is a
bad thing to be explaining to a room.

This puts the database back to "freshly migrated": schema untouched, the
baseline model_versions row kept, everything else gone. Run seed_retrain_data.py
afterwards to get the 55 confirmations back.

    cd backend
    ./venv/bin/python reset_demo.py            # shows counts, asks before deleting
    ./venv/bin/python reset_demo.py --yes      # no prompt
    ./venv/bin/python reset_demo.py --dry-run  # show what would go, delete nothing

It never touches the schema, the migrations, or Neo4j. The graph is loaded
with MERGE statements from files in graph/, so it is reproducible on its own
and has no demo state to clear.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.settings import settings  # noqa: E402

# Deletion order is forced by the foreign keys, not by taste:
#
#   batch_items.prediction_id -> predictions(id)   NO cascade  (migration 005)
#   batch_items.batch_id      -> batches(id)       on delete cascade
#   annotations.prediction_id -> predictions(id)   on delete cascade
#   xai_jobs.prediction_id    -> predictions(id)   on delete cascade
#   retrain_jobs.model_version_id -> model_versions(id)  NO cascade
#
# So batches must go before predictions (it takes batch_items with it, which
# is what holds the un-cascaded reference), and retrain_jobs before
# model_versions. Deleting predictions first fails on batch_items.
# The third field is that table's timestamp column, and it is not uniformly
# created_at: annotations stamps annotated_at, model_versions trained_at,
# retrain_jobs triggered_at. PostgREST refuses an unfiltered delete, so the
# "delete everything" predicate has to name a column the table really has.
PLAN = [
    ("retrain_jobs", "every retrain job", "triggered_at", None),
    ("batches", "every uploaded batch (cascades batch_items)", "created_at", None),
    ("predictions", "every prediction (cascades annotations and xai_jobs)",
     "created_at", None),
    ("model_versions", "retrained versions, keeping the baseline row",
     "trained_at", ("version_number", "baseline*")),   # see the delete loop
    ("dataset_versions", "every dataset version", "created_at", None),
]

# Printed after, so the "did it work" check is in the same output.
AFTER = ["batches", "batch_items", "predictions", "annotations", "xai_jobs",
         "retrain_jobs", "model_versions", "dataset_versions"]


def count(client, table: str) -> int:
    try:
        return client.table(table).select("*", count="exact").limit(1).execute().count
    except Exception as exc:
        print(f"  ! {table}: {type(exc).__name__}")
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, delete nothing")
    args = ap.parse_args()

    from repositories._client import get_client, is_configured

    if not is_configured():
        print("SUPABASE_URL still contains 'placeholder' — nothing to reset. The "
              "in-memory store clears itself when the backend restarts.")
        return 0

    client = get_client()
    print(f"Database: {settings.SUPABASE_URL}\n")
    print("Now:")
    for table in AFTER:
        print(f"  {table:<18} {count(client, table):>6} rows")

    print("\nThis will delete:")
    for _, description, _, _ in PLAN:
        print(f"  - {description}")
    print("\nIt keeps the schema, the migrations, the baseline model_versions row,"
          "\nand the Neo4j graph. Run seed_retrain_data.py afterwards to restore"
          "\nthe 55 confirmations.")

    if args.dry_run:
        print("\n--dry-run: nothing deleted.")
        return 0

    if not args.yes:
        print(f"\nType 'reset' to delete this data from {settings.SUPABASE_URL}")
        try:
            if input("> ").strip() != "reset":
                print("Cancelled. Nothing was deleted.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled. Nothing was deleted.")
            return 1

    print()
    failures = []
    for table, description, stamp, keep in PLAN:
        try:
            query = client.table(table).delete()
            if keep:
                column, pattern = keep
                # Keep the baseline row: delete only what does NOT match it.
                #
                # Two things about this pattern, both learned the hard way:
                #
                #   ilike, not like -- the row is named "Baseline", and a
                #   case-sensitive "baseline..." would miss it. That failure
                #   is silent and backwards: it deletes the one row this is
                #   here to protect.
                #
                #   "*", not "%" -- PostgREST's wildcard is "*". A literal "%"
                #   goes into the query string as the start of a percent
                #   escape, and Supabase's edge worker 500s on it ("JSON could
                #   not be generated", Cloudflare 1101) before the request ever
                #   reaches Postgres. So the whole delete is skipped and any
                #   retrained version survives the reset.
                query = query.not_.ilike(column, pattern)
            else:
                # PostgREST refuses an unfiltered delete, so match every row
                # with a predicate true for all of them, on a column this
                # table actually has.
                query = query.gte(stamp, "1970-01-01T00:00:00Z")
            query.execute()
            print(f"  cleared  {table:<18} ({description})")
        except Exception as exc:
            failures.append(table)
            print(f"  FAILED   {table:<18} {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}")

    print("\nAfter:")
    for table in AFTER:
        print(f"  {table:<18} {count(client, table):>6} rows")

    if failures:
        # Non-zero so reset_demo.sh (set -e) stops here rather than seeding on
        # top of a half-cleared database. A FAILED line on its own scrolls past
        # under the seed output, and the state it leaves -- old predictions or a
        # leftover retrained version still in the Admin dropdown -- is exactly
        # what this script exists to prevent.
        print(f"\nNOT reset: {', '.join(failures)} still holds data. Fix that "
              f"before seeding, or the demo starts from an unknown state.")
        return 1

    print("\nNext:  python3 seed_retrain_data.py     (from the repo root)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
