# Local session scope — Aditya (backend)

Not committed. This file declares my boundary for Claude Code sessions.

## I own

`app/`, `workers/`, `tests/`, `Dockerfile`, `docker-compose.yml`,
`requirements.txt`

## Off limits in my sessions

- `db/` — Arjun owns the Postgres schema, migrations, and RPC functions.
  If an endpoint needs a column or table that doesn't exist, do NOT write
  DDL. Write the spec into `docs/SCHEMA_REQUEST.md` and tell me.
- `ml/` — Ankush. `ml/inference.py` is a read-only dependency; import
  `predict_text`, `predict_batch`, `explain_text` from it, never edit it.
- `graph/` — Arjun's Neo4j.
- `frontend/` — Bidisha.

## Working style

- Plan mode for anything touching more than two files.
- Tests alongside the code, not after. `pytest -q` must pass before you
  say a task is done.
- Don't refactor code I didn't ask about, even if it looks wrong. Tell me
  instead.
- My task list is `docs/BACKEND_TASKS.md`. I feed it to you one task at a
  time — don't read ahead or start the next one unprompted.
