# Medical demo PDFs

Two files, for the second domain. Upload through **Annotate → Upload File**
on the medical tab (`http://localhost:5181`, started by `./run_local_demo.sh`).

Every record in these files comes from `sample-data/evaluation/medical_heldout.csv`
— real clinical transcriptions the medical model was **never trained on**, each
carrying its gold specialty. That provenance is the point: when someone asks
where the documents came from, the answer is the held-out split, not the
internet.

## The domain, before any PDF

The medical tab serves **Medical Round 1** (since 24 Oct): Bio_ClinicalBERT
fine-tuned on 881 training notes, then on a first round of 296 more
(`backend/medical_progression.py`). On all 273 held-out transcriptions:

```
accuracy                       248/273 = 90.8%   (round-0 baseline: 244 = 89.4%)
auto-accepted (conf >= 0.85)   209 docs, 97.6% accurate
routed to review (conf < 0.85)  64 docs, 68.8% accurate
```

Of 25 errors, **20 (80%) fall below the auto-accept bar on their own**. The
remaining 5 are confidently wrong, 1.8% of the corpus, each a different
confusion, and the ontology check flags none of them (and no correct ones
either). The model this replaced, MedicalModelShared, made 8 confident errors,
four of them one confusion — a cardiac catheterization report filed as
Radiology — and the check caught 2.

Round 1's 90.8% is one run; the five-run result in RESULTS.md section 10 is
the evidence that feeding annotations back helps.

## The files

| Upload | File | Pages → docs | Accuracy | Avg conf | What it demonstrates |
|---|---|---:|---:|---:|---|
| 10 | `neurology-consults.pdf` | 6 → 19 | 14/19 = 73.7% | 0.632 | **The routing transfers too.** 18 of 19 go to review |
| 11 | `cardiology-catheterization.pdf` | 9 → 23 | 16/23 = 69.6% | 0.818 | **The ontology check transfers.** 1 flagged, correctly, 0 false positives |

### 10 — it knows what it does not know

On these consult passages the model is unsure, average confidence 0.632, and
**18 of 19 go to a human** instead of being auto-accepted. The one it accepts
on its own is right. Low confidence arriving where the model is weak is the
routing working, not failing.

### 11 — confidently wrong, caught

**Read the accuracy number in context: this file is deliberately weighted
toward the failure case.** It holds all four records the previous model got
confidently wrong plus four it got right. It is not a random sample and
should not be presented as one.

Twelve passages clear the auto-accept bar and three of them are wrong. The
ontology check flags one, correctly, with no false positive:

```
0.896  model said Radiology  ->  check says Cardiovascular / Pulmonary
       evidence 46.7 vs 0.0   on  pigtail, pigtail catheter, coronary angiography,
                                  femoral artery
```

`pigtail catheter`, `coronary angiography` and `femoral artery` are the
vocabulary of a cardiac catheterization. The model called it Radiology at
0.90; the vocabulary says otherwise by 46.7 to 0.

The other two confident errors get through. Say that plainly if asked: on
medical the check is conservative. On the 273 held-out notes it flags nothing
for this model, and caught 2 of 8 for the previous one at 0.5% false
positives, against legal's 8.0%. The operating point was never tuned for
medical, and tuning it on eight examples would be overfitting, not
engineering.

## Why passage accuracy is below 90.8%

A PDF is split into passages packed to the model's 256-token window, and each
passage is classified on its own. A clinical transcription is short and
front-loads its diagnosis, so a middle passage can be a list of measurements
with the context stripped away. Legal judgments do not have this problem —
they are long and every passage is still about the same case, which is why
`judgment-labour.pdf` scores 12/12.

This is a property of the documents, not a defect in the extractor. Say so.

## Re-measuring

These numbers come from `backend/measure_demo_pdfs.py`, which runs each file
the way an upload does: `pdf_extract.extract()` packing by the tokeniser's
count, `model_service.predict_batch()` in the worker's group size, then
`ontology_check.check()` on everything above the bar. Run it whenever the
served medical model changes:

```
cd backend
ACTIVE_DOMAIN=medical MODEL_PATH=./models/medical-progression/v1_round1 ./venv/bin/python measure_demo_pdfs.py
```

The first version of this page was measured by hand with passages split by
character count, which no upload does; its cardiology figure (four flagged)
described passages the app never produces. Corrected 24 Oct. For reference,
the previous model on the real path: neurology 13/19 with 18 routed,
cardiology 10/23 with 7 confident errors and 1 caught. The urology file,
now in `sample-data/archive/medical-pdfs/`, scores 10/14 on Round 1.

## How the medical tab is wired

The Admin tab lists two versions: **Medical Baseline** (round 0, 89.4%) and
**Medical Round 1** (90.8%, active and serving), registered at startup from
`backend/metrics/medical_baseline_metrics.json` and
`backend/metrics/medical_versions.json`. Both are scored on the same 273
held-out notes, and the line under Round 1 says so.

`./run_local_demo.sh` starts both domains; switch with the **Legal / Medical**
toggle in the header. Each domain is its own backend process, because the label
set (10 vs 12) is fixed when a process starts.

**Until migration 010 is applied**, the medical backend runs on an
**in-memory store**, not the shared Supabase. The
job queues (`claim_batch_items`, `claim_xai_job`, retrain jobs) record no
domain, so two workers polling one database would claim each other's work — a
legal upload classified by the medical model. Isolating the store makes that
impossible without a schema change. Its consequences:

- the medical tab's status strip reads **in-memory** for the database — true;
- medical uploads disappear when the stack restarts;
- Retrain on medical reports not eligible (no validated medical annotations),
  and cannot touch the legal tables in any case.

Neo4j stays shared: the graph is only read here, and every query filters by
domain.

**After migration 010** (`db/migrations/010_domain_on_job_queues.sql`, pasted
into the Supabase SQL editor), the launcher detects it and puts medical on
Supabase too: each upload, explanation and retrain job carries its domain, each
worker claims only its own, and each domain keeps its own active version.
`check_data_layer.py` reports which state you are in.
