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

---

# UPDATE — 19 Oct: read this before starting Job 2

Three things changed after this document was written. All of them make Job 2
simpler; one of them was a bug that would have cost you an afternoon.

## 1. The loader does both files now, per domain

`load_ontology.py` previously loaded one hardcoded file. It now takes
`--domain` and loads that domain's two seeds in order:

```
cd graph
export NEO4J_URI="neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="..."
export NEO4J_DATABASE="neo4j"

python load_ontology.py --domain legal       # 32 + 1505 statements
python load_ontology.py --domain medical     # 32 + 2359 statements
```

`--dry-run` counts the statements without connecting, which is a quick way
to check your checkout is complete before you touch the database.

The earlier instruction to add a `--file` argument yourself is obsolete.

## 2. There is a second domain

A clinical specialty domain was added, with its own model, label set and
concept layer. Nothing about the data layer changes because of it — the
schema already carries a `domain` column from migration 007 — but it means
there are two sets of seeds, and the naming is now uniform:

```
graph/legal_ontology_seed.cypher      10 domain nodes, 21 RELATED_TO edges
graph/legal_concept_seed.cypher       752 concepts
graph/medical_ontology_seed.cypher    12 domain nodes
graph/medical_concept_seed.cypher     1,179 concepts (re-mined 19 Oct)
```

The old `ontology_seed.cypher`, `concept_seed.cypher` and
`concept_markers.json` are gone — renamed with their domain prefix. If you
have a branch referring to them, that is why.

Every one of these is **generated** by `graph/mine_concept_markers.py` from
`domains/<key>.json`. Do not hand-edit them; re-run the miner.

## 3. The bug: concept seeds used to attach to nothing

The concept seed attached each edge with:

```cypher
MATCH (c:Concept {...}), (d:OntologyNode {label: "..."}) MERGE (c)-[:INDICATES]->(d)
```

A `MATCH` that finds nothing is **not an error** in Cypher — it does nothing.
So loading a concept seed into a database whose domain nodes did not exist
would have reported success and created **zero edges**, silently. The legal
seed never hit this only because the hand-written ontology seed always
happened to be loaded first.

It now MERGEs the domain node, so each concept seed stands alone and order
is forgiving.

**What this means for you:** if you already loaded seeds and
`verify_ontology.py` behaved oddly, wipe and reload rather than debugging it:

```cypher
MATCH (n) DETACH DELETE n;
```

then re-run the loader. And check the edges actually landed:

```cypher
MATCH (:Concept)-[r:INDICATES]->(:OntologyNode) RETURN count(r);
// expect 752 for legal, 1179 for medical
```

## 4. Verification takes a domain too

```
python verify_ontology.py --domain legal
python verify_ontology.py --domain medical
```

It checks the graph against `domains/<key>.json` — the same file
`mock_db.ONTOLOGY` is projected from, so passing it means the mock and the
graph genuinely agree.

## Still true, and still the thing that wastes the most time

If nothing persists after you wire up Supabase, check that `SUPABASE_URL` no
longer contains the word `placeholder`. `is_configured()` returns False and
everything silently keeps using the in-memory mock, which looks exactly like
it is working.

---

# UPDATE — 20 Oct (data layer): what connecting it for real turned up

Branch `data-layer-pg-neo4j`. Each of these was found by running the app
against a real Postgres (all migrations, behind PostgREST, which is what
Supabase serves) rather than by reading code. The in-memory mock hid every
one of them — it has no column types, no table list, and one domain per
process.

**Run migration 008 too.** `008_app_schema_gaps.sql`: `annotator_id` and
`triggered_by` become `text` (the frontend sends `X-User-Id: mock-user-1`,
which a `uuid` column rejects, so every annotation failed), and `retrain_jobs`
gains `notes` / `min_annotations`. Safe to re-run. If your project already has
001–007 from earlier, run only 008.

**Use the legacy key.** Project Settings → API Keys → *Legacy API keys* →
`service_role` (starts `eyJ`). The pinned client, supabase==2.4.6, rejects the
newer `sb_secret_…` keys with "Invalid API key".

**One command tells you whether it is really connected:**

```
cd backend && ./venv/bin/python check_data_layer.py
```

It queries both databases directly (the app falls back to the mock silently,
so "the app works" proves nothing), checks every table and migration, counts
every domain's graph against its seeds, and compares the live ontology to the
mock function by function.

**The loader stamps a domain on each node.** Both domains share one graph, and
nothing recorded which node belonged to which — after loading medical,
`verify_ontology.py --domain legal` saw 22 nodes and failed. Graphs loaded by
an older loader: just re-run it for each domain.

What changed and why is in the commit message on that branch.

---

# UPDATE — 20 Oct (ML side): a third seed file, and a rebuilt corpus

Shorter than the last one. Nothing here breaks what you may have started;
one item adds a file to load, the rest is context so numbers in the repo
match what you see.

## 1. There is now a third seed file per domain

`graph/mine_graph_depth.py` mines structure *between* concepts and writes
`<key>_depth_seed.cypher`:

```
legal_depth_seed.cypher      84 NARROWS +     6 CO_OCCURS
medical_depth_seed.cypher   182 NARROWS + 1,205 CO_OCCURS
```

```
(:Concept)-[:NARROWS]->(:Concept)
    "cervical discectomy" is a kind of "discectomy"

(:Concept)-[:CO_OCCURS {jaccard, documents}]-(:Concept)
    "ankle tourniquet" and "sagittal saw" appear in the same operation note
```

**`load_ontology.py` already loads it** — third, after the ontology and
concept seeds, and only if it exists. You do not need to change the loader.
A domain without a depth seed prints a note and loads the other two, where
previously any missing file aborted the run.

Both ends of every edge are `MERGE`d, so the file is order-independent, but
load the concept seed first anyway.

**These edges are not read by anything yet.** `services/ontology_check.py`
reads the markers JSON and never opens a Neo4j session, and
`services/neo4j_service.py` queries only `(:OntologyNode)` with
`HAS_CHILD`, `RELATED_TO` and `HAS_PARENT`. So you can load them, or not,
without changing any API response or any measured number. If you want to
expose them later, that is a new endpoint and a new measurement, not a swap.

## 2. The medical corpus and markers were rebuilt

`sample-data/evaluation/medical_heldout.csv` shared 129 of its 435 rows with
`datasets/medical_v1.csv`, every one of them carrying a *different* label —
mtsamples files the same transcription under two specialties. The corpus is
rebuilt by `backend/build_medical_corpus.py`, which groups by text so copies
cannot straddle the split and asserts zero overlap before writing.

For you this means two things:

- `datasets/medical_v1.csv` is now **1,472 rows**, not 1,738, and
  `medical_heldout.csv` is **273**, not 435. If you cached row counts
  anywhere, they moved.
- `graph/medical_concept_markers.json` and `medical_concept_seed.cypher`
  were regenerated from the clean corpus: **1,179 markers**, was 1,108.
  Reload the medical graph if you had already loaded it; the old concept
  set contains terms mined from rows that no longer exist.

Legal is untouched: 752 markers, 8.0% false positives, regression-checked.

## 3. One API bug fixed, in case you hit it

`GET /batches` returned **500 on every request** since it was written —
`models/response_models.py` defined `BatchSummary` twice with different
fields and the second silently replaced the first. Fixed by renaming the
comparison-picker one to `ComparisonBatchSummary`. If you saw the Annotate
page say "Could not load recent batches", that was it, not your setup.

## 4. You do not need a GPU, a model, or the weights

Worth saying plainly because it changes how you can work:

- The **test suite stubs torch**. All 221 tests run with no model loaded,
  so `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/ -q`
  is a real pass/fail signal on any machine, including a Mac with no CUDA.
- `backend/models/` is ~4.9 GB and gitignored. You will not get it from a
  clone and you do not need it for this job.
- Applying the SQL migrations and loading the Cypher needs Postgres and
  Neo4j, nothing else. No PyTorch, no checkpoint.

Only clicking through the live UI needs a model, and that is optional for
the data layer.
