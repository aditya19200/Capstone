# Handoff: connect the real data layer

Two independent jobs: **Postgres (Supabase)** and **Neo4j**. Either can be done
without the other. Neither requires touching the ML code.

Repo: `aditya19200/Capstone`, branch **`prototype-v1`**. Everything below is
relative to `prototype-v1/`.

---

## Context you need first

The prototype runs today with **no database at all**. Every repository module
checks whether real credentials exist and falls back to an in-memory Python
dict (`backend/services/mock_db.py`) when they don't:

```python
# backend/repositories/_client.py
def is_configured() -> bool:
    return "placeholder" not in settings.SUPABASE_URL
```

So **all data is lost when the backend restarts**. That is the thing being
fixed.

**The real SQL is already written.** Every repository function already has
both branches — mock and real. Example from
`backend/repositories/predictions.py`:

```python
def insert(...):
    if not is_configured():
        from services import mock_db
        return mock_db.create_prediction(...)
    result = get_client().table("predictions").insert({...}).execute()
    return result.data[0]
```

You are not writing queries. You are provisioning a database, running
migrations, and setting two environment variables.

### The one rule that matters

**The demo must keep working if the database is unreachable.** The mock
fallback is not legacy code to delete — it is what makes the prototype
runnable on a laptop with no internet. Do not remove `is_configured()`
branches. Do not delete `mock_db.py`.

---

# JOB 1 — Postgres via Supabase

### 1. Create the project
supabase.com → new project. Note the password. Wait for it to finish provisioning.

### 2. Run the migrations **in order**

SQL Editor → paste and run each file, one at a time, in this exact order:

```
db/migrations/001_init_schema.sql      8 tables
db/migrations/002_rpc_functions.sql    job-queue functions (FOR UPDATE SKIP LOCKED)
db/migrations/003_activate_model_version_and_conflict.sql
db/migrations/004_rls_policies.sql
db/migrations/005_batch_items_prediction_id.sql
db/migrations/006_created_at_ordering.sql
db/migrations/007_domain_column.sql    adds the domain column, relaxes label_id
```

Order is not optional — 005, 006 and 007 alter tables that 001 creates.

The 8 tables: `dataset_versions`, `model_versions`, `predictions`, `batches`,
`batch_items`, `xai_jobs`, `annotations`, `retrain_jobs`.

### 3. Credentials

Project Settings → API. You need the **Project URL** and the
**`service_role`** key (NOT the `anon` key — the backend bypasses RLS by
design; see `004_rls_policies.sql`).

Create `backend/.env`:

```
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=<service_role key>
```

`.env` is gitignored. **Never commit the service_role key** — it bypasses all
row-level security.

### 4. Install the client

`supabase==2.4.6` is in `requirements.txt` but may not be in the venv:

```
cd backend && ./venv/bin/python -m pip install supabase==2.4.6
```

Use `./venv/bin/python -m pip`, not `./venv/bin/pip` — the venv was moved and
the pip launcher's shebang points at a path that no longer exists.

### 5. Acceptance test

```
./run_local_demo.sh
# upload sample-data/demo/1-intake.csv through the Annotate page
# wait for classification, then Ctrl-C the backend
./run_local_demo.sh
# the batch must still be there
```

**Done = data survives a restart.** That is the whole deliverable.

### Likely failures

- Nothing persists → `SUPABASE_URL` still contains the word `placeholder`, so
  `is_configured()` is False and it is silently still using the mock. Check the
  URL first, every time.
- `relation does not exist` → a migration was skipped or run out of order.
- RPC errors on batch classify → `002` didn't run.

---

# JOB 2 — Neo4j

### What this does and does not change

It does **not** change any ML behaviour. The ontology consistency check reads
`graph/concept_markers.json` (a flat file), not Neo4j. Connecting Neo4j makes
the `/ontology` endpoints read from a real graph instead of a Python dict. Do
not expect accuracy numbers to move — if they do, something is wrong.

### 1. Create the instance

Neo4j Aura free tier (neo4j.com/cloud/aura). Download the credentials file it
gives you — the password is shown once.

Note: some Aura Free instances use the **instance ID** as the database name
rather than `neo4j`. If queries return nothing, that is usually why.

### 2. Load the graph — BOTH files

```
cd graph
pip install neo4j
export NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export NEO4J_DATABASE="neo4j"
python load_ontology.py
```

**`load_ontology.py` only loads `ontology_seed.cypher`** — the 10 domain
nodes. The file path is hardcoded at the top:

```python
SEED_FILE = Path(__file__).parent / "ontology_seed.cypher"
```

You must **also** load `concept_seed.cypher`, which holds the 752 mined
concept terms and their `INDICATES` edges. Easiest fix: add a `--file`
argument to `load_ontology.py` and run it twice. Load `ontology_seed` first —
`concept_seed` uses `MERGE` on the domain nodes, so order is forgiving, but
start with the domains anyway.

Both `.cypher` files are **generated** by
`graph/mine_concept_markers.py`. Do not hand-edit them; re-run the miner.

### 3. Add credentials to `backend/.env`

```
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

These already exist in `backend/config/settings.py` — no new settings needed.

### 4. Wire up the service

`backend/services/neo4j_service.py` has 9 functions, all delegating to
`mock_db`. The real driver block is commented out around lines 36–42:

```python
# from neo4j import GraphDatabase, Driver
# _driver: Driver = GraphDatabase.driver(
#     settings.NEO4J_URI,
#     auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
# )
```

Each function's docstring already states its Cypher. For example:

```python
def get_all_nodes():
    """
    Mock: reads from the static ONTOLOGY dict in mock_db.
    Real: MATCH (n:OntologyNode) RETURN n
    """
```

The 9 functions: `get_all_nodes`, `get_node`, `label_exists`,
`validate_label_hierarchy`, `get_children`, `get_related`, `get_parent`,
`get_full_hierarchy`, `get_all_labels`.

**Add a configuration gate, mirroring `_client.is_configured()`**, so an
unreachable Neo4j falls back to the mock rather than breaking the app:

```python
def _is_configured() -> bool:
    return "localhost" not in settings.NEO4J_URI

def get_all_nodes() -> List[Dict]:
    if not _is_configured():
        return db.get_ontology_nodes()
    with _get_session() as s:
        ...
```

Return types must not change. Each function's docstring states the exact
shape; `backend/models/response_models.py` validates it, and the frontend
reads it.

### 5. Acceptance test

```
cd graph && python verify_ontology.py --domain legal
```

This checks the graph against `domains/legal.json` — the same file
`mock_db.ONTOLOGY` is projected from. It verifies all 10 labels exist and
every `RELATED_TO` edge matches.

Then:

```
curl -s localhost:8000/ontology -H "X-Role: admin" | head
```

Must return the same 10 nodes as before the change.

**Done = `verify_ontology.py` passes and `/ontology` is unchanged.**

---

## Definition of done, both jobs

1. `./run_local_demo.sh` still works end to end
2. Data survives a backend restart (Job 1)
3. `python graph/verify_ontology.py --domain legal` passes (Job 2)
4. `cd backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest -q`
   shows no **new** failures — 3 are already failing on `prototype-v1` and are
   stale assertions, not bugs
5. No credentials committed

## Do not

- Delete `mock_db.py` or any `is_configured()` branch
- Hand-edit `graph/concept_seed.cypher` or `graph/concept_markers.json` (generated)
- Change any function's return type in `neo4j_service.py`
- Touch `backend/services/ontology_check.py`, `model_service.py`, or anything
  under `graph/mine_concept_markers.py` — that is the ML side and is being
  worked on in parallel

## If you get stuck

Say which step, paste the error, and say whether `is_configured()` is
returning True. Most problems are that it is still False and everything is
silently running on the mock.
