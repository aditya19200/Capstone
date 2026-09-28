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

**The second detector.** Confidence-based routing can only escalate what the
model doubts, which makes it blind to predictions that are confident *and*
wrong — those are auto-accepted and never corrected. The ontology consistency
check reads the legal-concept vocabulary in a document and flags it when that
vocabulary contradicts the label, regardless of confidence. It catches about a
third of real errors, and 88% of what it flags is a genuine mistake. Markers
are mined from the corpus's human gold labels by `graph/mine_concept_markers.py`
— never from model predictions, since a detector built on the model's own
output would inherit the blind spot it exists to catch.

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

**Try it:**

1. **Annotator** → Annotate → upload `sample-data/demo_registry_intake.csv`
2. **Reviewer** → Review — 27 items, most flagged by ontology conflict rather
   than low confidence. Each shows the concept evidence on both sides and
   arrives with the suggested label pre-selected, so correcting one is a single
   click. Correct at least 15
3. **Annotator** → upload `sample-data/demo_probe_set.csv` — 8 documents the
   system has never been corrected on. This is the "before"
4. **Admin** → Retrain → Activate
5. **Annotator** → upload `demo_probe_set.csv` again
6. **Admin** → Compare → pick the two probe batches

Seven or eight of the eight change from Family Law to Criminal Law, and none of
them were ever corrected. 15 corrections is the measured minimum for all eight;
5 and 10 both give 7/8.

---

## Layout

| Path | What |
|------|------|
| `backend/` | FastAPI API, workers, ML services |
| `frontend/` | React app |
| `db/` | Postgres schema, migrations, RPC functions |
| `graph/` | Neo4j legal-concept ontology |
| `datasets/v1.csv` | 1701 labelled training samples |
| `sample-data/` | Demo batches and the evaluation set — see below |

### `sample-data/`

| File | Rows | Confidence mix | Use for |
|------|-----:|----------------|---------|
| `demo_registry_intake.csv` | 53 | mixed, all 10 domains | **The demo.** 26 classify correctly, 5 are genuinely low confidence, 22 are confident and wrong. Four of those are real court documents failing on unrelated confusions |
| `demo_probe_set.csv` | 8 | confident and wrong | Held out for before/after. Upload once before retraining and again after, then compare. Never correct these |
| `demo_confident_errors.csv` | 25 | confident and wrong | The teaching set on its own, if you want it without the surrounding batch |
| `demo_batch_tough.csv` | 45 | 60% high / 29% med / 11% low | Older demo batch. Colour on every page, 5 items reach the review queue |
| `demo_batch_easy.csv` | 50 | 94% high / 4% / 2% | Safe fallback — but pages 2-4 are entirely green |
| `stress_test.csv` | 225 | — | Measuring accuracy. Real labelled court text with a `gold_label` answer key. 15 pages, ~75s to classify — not for live use |

The uploader reads only the `text` column, so `stress_test.csv`'s answer-key
columns never reach the model.

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
tracked as outstanding work.

Activating a version now loads its weights into the running service, so the
next prediction really does come from the model you promoted — check
`GET /health`, which reports the version actually resident rather than the one
configured. If the weights fail to load, the previously loaded model keeps
serving and `is_active` is rolled back.
