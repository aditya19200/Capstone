# Prototype V1 — XAI Legal Annotation Framework

Classifies documents into a domain's label set, explains *why* it chose a
label, routes uncertain cases to a human, and retrains itself on the
corrections those humans make.

This prototype demonstrates that loop running **end to end**, on **two
domains that differ only by a config file**:

| domain | base model | labels | held-out accuracy |
|---|---|---|---|
| legal | InLegalBERT | 10 Indian legal areas | **94.4%** document-level, 18 documents |
| medical | Bio_ClinicalBERT | 12 clinical specialties | **90.5%**, 273 documents |

Adding the second domain is what found most of the bugs worth finding. Every
number here and its caveat lives in **`docs/RESULTS.md`** — read that before
quoting anything. **`docs/PROJECT_GUIDE.md`** explains how the whole system
works and why, including the demo runbook and the terminology.

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

1. **Annotator** → Annotate → upload `sample-data/demo/legal-csv/1-intake.csv`
2. **Reviewer** → Review — 27 items, most flagged by ontology conflict rather
   than low confidence. Each shows the concept evidence on both sides and
   arrives with the suggested label pre-selected, so correcting one is a single
   click. Correct at least 15
3. **Annotator** → upload `sample-data/demo/legal-csv/2-probe-set.csv` — 8 documents the
   system has never been corrected on. This is the "before". **Look at the
   results table on the Annotator page**, which shows all 8 as Family Law with
   a green "Model certainty: HIGH" badge. It carries no ontology information
   and no gold label, because the conflict is computed at queue-read time and
   never stored on the prediction — so to show the evidence rather than assert
   it, open Review as well, where 3 of the 8 appear with the full concept
   comparison
4. **Admin** → Retrain → Activate
5. **Annotator** → upload `2-probe-set.csv` again
6. **Admin** → Compare → pick the two probe batches

> **Never press Submit on a probe row.** Looking at them is free. Submitting
> one writes a validated annotation, the retrain worker trains on every
> validated annotation, and the before/after stops meaning anything. Three
> probes sit in the review queue after step 3, flagged by ontology conflict,
> looking exactly like the items you were correcting at step 2.

Seven or eight of the eight change from Family Law to Criminal Law, and none of
them were ever corrected.

The five probes the ontology check does *not* flag are the point rather than a
gap: the loop only needs to catch enough instances of an error class to learn
the fix. It caught 18 of the 18 teaching rows inside `1-intake.csv` — all by
concept evidence, none by low confidence — and the five nothing ever flagged
were fixed anyway.

Measured offline on the five recorded retrain checkpoints
(`backend/compare_checkpoints.py`), so the result does not depend on the live
run going well:

| checkpoint | validated rows it trained on | probes correct |
|---|---|---|
| `LegalModelShared` (deployed) | — | **0 / 8** |
| v-20260904-091128 | 47 | 0 / 8 |
| v-20260910-102339 | 48 | 0 / 8 |
| v-20260928-154348 | 60 | 7 / 8 |
| v-20260928-185817 | 65 | 7 / 8 |
| v-20260929-090239 | 65 | **8 / 8** |

All eight of the deployed model's errors sit at 0.985–0.996 confidence, above
the 0.85 auto-accept threshold — in production they would be filed as Family
Law with no human ever seeing them. The eight probe texts appear in none of
`v1.csv`, `1-intake.csv`, `teaching-set-only.csv` or `seed_retrain_data.py`,
checked before quoting any of this. The flip happens somewhere between 48 and
60 validated annotations.

---

## Input formats

Three ways in, one pipeline. Past the point where an upload becomes a list of
strings, nothing downstream can tell them apart — same queue, same routing,
same ontology check, same review screen, same export, same retraining.

| Endpoint | Takes | Notes |
|---|---|---|
| `POST /batches/paste` | one document per line | |
| `POST /batches/csv` | a `text` column, or a single-column file | |
| `POST /batches/pdf` | a digital PDF | Split into passages that fit the model's 256-token window, breaking on paragraph then sentence boundaries — not on page boundaries, which fall mid-sentence |

All three normalise before storing. That is not tidiness: the retrain worker
normalises before training, so text stored raw would be classified in one
form and trained on in another. It matters most for PDFs, whose text arrives
full of hyphen-breaks and soft line wraps.

**Scanned PDFs are refused, deliberately.** A scan is a photograph of a page
with no text layer, and extraction returns empty strings rather than failing.
Ingesting one would queue blank documents, classify them confidently as
something, and quietly poison the annotation set. Reading them needs OCR,
which this does not do — so it says so instead of pretending.

## Layout

| Path | What |
|------|------|
| `backend/` | FastAPI API, workers, ML services |
| `frontend/` | React app |
| `domains/` | The label sets. One `<key>.json` per domain — names, ids, descriptions and the model that predicts them. `ACTIVE_DOMAIN` in `backend/.env` picks which one is served |
| `db/` | Postgres schema, migrations, RPC functions |
| `graph/` | Neo4j legal-concept ontology |
| `datasets/v1.csv` | 1701 labelled training samples |
| `sample-data/` | `demo/` what you upload during the demo · `evaluation/` measurement sets · `archive/` older batches |

### `sample-data/`

| File | Rows | Confidence mix | Use for |
|------|-----:|----------------|---------|
| `demo/legal-csv/1-intake.csv` | 53 | mixed, all 10 domains | **The demo.** 26 classify correctly, 5 are genuinely low confidence, 22 are confident and wrong. Four of those are real court documents failing on unrelated confusions |
| `2-probe-set.csv` | 8 | confident and wrong | Held out for before/after. Upload once before retraining and again after, then compare. Never correct these |
| `archive/teaching-set-only.csv` | 25 | confident and wrong | The teaching set on its own, if you want it without the surrounding batch |
| `archive/demo_batch_tough.csv` | 45 | 60% high / 29% med / 11% low | Older demo batch. Colour on every page, 5 items reach the review queue |
| `archive/demo_batch_easy.csv` | 50 | 94% high / 4% / 2% | Safe fallback — but pages 2-4 are entirely green |
| `evaluation/stress_test.csv` | 225 | — | **Not a test set.** All 225 rows also appear in `v1.csv`, verified by exact match after normalisation — it is a slice of the training corpus. Kept for load-testing the UI, never for accuracy |
| `evaluation/heldout/` + `heldout_manifest.csv` | 21 docs | — | **The real test set.** Documents in no CSV anywhere. 3 are scans the PDF guard refuses, so 18 reach the model |
| `evaluation/medical_heldout.csv` | 273 | — | Medical held-out, rebuilt 19 Oct so no document straddles the split |

The uploader reads only the `text` column, so the answer-key
columns never reach the model.

---

## Model performance — read the caveat

> **A held-out set exists now, and these are not it.** Until 18 Oct every
> number this project had came from `v1.csv`, which may be the deployed
> checkpoint's own training data. The figures in this section are that old
> measurement, kept because the source breakdown below is still the clearest
> evidence of what the corpus is made of. **The headline accuracy is
> 94.4% document-level on 18 held-out documents** — see `docs/RESULTS.md`
> sections 1 and 2.

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
upper bound rather than a held-out result, and the 52% on genuine Supreme
Court summaries was the honest read of it at the time.

On the held-out set built since — 21 documents appearing in no CSV, of which
18 reached the model and 3 were scans the PDF guard refused — the model gets
**17 of 18 at document level (94.4%)** and **84.5% mean per-document passage
agreement**. The single miss is the Indian Stamp Act, labelled Taxation and
predicted Property, where the prediction is arguably better than our label
and is left as a miss on purpose.

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
