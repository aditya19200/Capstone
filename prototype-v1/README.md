# Prototype V1 — XAI Legal Annotation Framework

Classifies Indian legal text into 10 legal domains, explains *why* it chose a
label, routes uncertain cases to a human, and retrains itself on the
corrections those humans make.

This prototype demonstrates that loop running **end to end**.

---

## What it does

```
   upload text  ──►  classify  ──►  explain (SHAP)
                        │
              low confidence?
                        │
                        ▼
               human reviews and corrects
                        │
                        ▼
        retrain on those corrections  ──►  new model version
                                            (saved inactive —
                                             a human activates it)
```

**The model.** Fine-tuned InLegalBERT, 10 classes:
Contract · Criminal · Constitutional · Corporate · Property · Family ·
Labour & Employment · Intellectual Property · Taxation · Civil Procedure.

**The explanation.** SHAP token attributions — for a divorce case it surfaces
"wife", "divorce", "custody", "children" as the words that drove the call.
Not a black box.

**The loop.** Predictions below the confidence threshold go to a review queue
instead of being trusted. A reviewer confirms or corrects them, and those
corrections become training data. Once 50+ corrections exist, retraining can
run.

**The safety rail.** A retrained model is **always** saved inactive. Nothing
promotes a model automatically — a human looks at the numbers and decides. If
a new model scores more than 5 points below the active one, it's still saved
but flagged with a warning.

---

## Running it

Requires **Node 20.19+** (system Node 18 will crash the frontend) and the
model weights at `backend/models/LegalModelShared/` — those are distributed
separately, not through git.

```bash
# terminal 1 — starts backend, workers and frontend
./run_local_demo.sh
# wait for "Backend ready." then open http://localhost:5180
```

```bash
# terminal 2 — once per backend start, only needed for the retrain demo
python3 seed_retrain_data.py
```

`Ctrl+C` in terminal 1 stops everything.

**Try it:** log in as **Annotator** → Annotate → upload `demo_batch.csv` (45
rows, deliberately spread across confidence bands) → click **Explain** on a
row → switch to **Reviewer** to correct the uncertain ones → switch to
**Admin** to trigger a retrain and activate the result.

---

## Layout

| Path | What |
|------|------|
| `backend/` | FastAPI API, workers, ML services |
| `frontend/` | React app |
| `db/` | Postgres schema, migrations, RPC functions |
| `graph/` | Neo4j legal-concept ontology |
| `datasets/v1.csv` | 1701 labelled training samples |
| `demo_batch.csv` | 45-row demo batch |
| `OWNERSHIP.md` | Who owns which files |

---

## Model performance — read the caveat

Measured by `backend/evaluate_baseline.py` on all 1701 labelled samples:

| | |
|---|---|
| Overall accuracy | 70.7% |
| Mean confidence | 85.4% |
| **Overconfident by** | **14.6 points** |

Broken down by where the data came from:

| Source | Accuracy |
|---|---|
| Hand-written examples | 100% |
| AI-generated | 93% |
| Extracted PDFs | 60% |
| **Real court summaries** | **52%** |

**100% on hand-written samples is memorisation, not skill.** The deployed
checkpoint was almost certainly trained on part of this data, so 70.7% is an
upper bound rather than a held-out result. The realistic figure is closer to
**52%** — the score on genuine Supreme Court case summaries, which is the
largest and most representative slice.

Worth stating plainly: for 10 classes, random guessing is 10%. The model is
doing real work; it is also more confident than it is correct, which is
exactly why the human review step exists.

---

## Current state

Running the prototype needs no internet and no cloud accounts — the data
layer uses an in-memory store rather than a live Postgres/Neo4j connection.

**That means all data resets when the backend restarts.** Fine for a demo,
and the reason `seed_retrain_data.py` exists. Connecting the real database is
tracked as outstanding work, along with a reload step so an activated model
version actually replaces the one in memory.
