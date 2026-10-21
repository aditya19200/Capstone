# Medical demo PDFs

Three files, for the second domain. Upload through **Annotate → Upload File**
on the medical tab (`http://localhost:5181`, started by `./run_medical_demo.sh`).

Every record in these files comes from `sample-data/evaluation/medical_heldout.csv`
— real clinical transcriptions the medical model was **never trained on**, each
carrying its gold specialty. That provenance is the point: when someone asks
where the documents came from, the answer is the held-out split, not the
internet.

## The domain, before any PDF

`MedicalModelShared` on all 273 held-out transcriptions:

```
accuracy                       247/273 = 90.5%
auto-accepted (conf >= 0.85)   214 docs, 96.3% accurate
routed to review (conf < 0.85)  59 docs, 69.5% accurate
```

The routing is doing its job: of 26 errors, **18 (69%) fall below the
auto-accept bar on their own**. The remaining 8 are confidently wrong — 2.9%
of the corpus — and are invisible to confidence-based active learning. Four
of those eight are the same confusion: a cardiac catheterization report filed
as **Radiology** at 0.93–0.96.

That is the legal Pradeep failure, in a different domain, found the same way.

## The files

| Upload | File | Pages → docs | Accuracy | Avg conf | What it demonstrates |
|---|---|---:|---:|---:|---|
| 1 | `urology-operative-notes.pdf` | 6 → 14 | 91.7% | 0.814 | **The pipeline transfers.** Same code, same upload, different domain |
| 2 | `neurology-consults.pdf` | 6 → 19 | 75.0% | 0.666 | **The routing transfers too.** 17 of 19 go to review |
| 3 | `cardiology-catheterization.pdf` | 9 → 23 | 47.6% | 0.830 | **The ontology check transfers.** 4 flagged, 3 correct, 0 false positives |

### 1 — it just works

Urology is the model's strongest specialty (100% on held-out rows). Nothing
clever happens here, which is the point: a second domain goes through the same
uploader, the same worker, the same review screen.

### 2 — it knows what it does not know

Neurology is one of the model's weakest specialties (77.8% on held-out rows,
average confidence 0.691). Watch what the system does with that: **17 of 19
documents route to a human** instead of being auto-accepted. Low accuracy and
low confidence arriving together is the routing working, not failing.

### 3 — confidently wrong, caught

**Read the accuracy number in context: this file is deliberately weighted
toward the failure case.** It holds all four confidently-wrong cardiology
records plus four the model gets right. It is not a random sample and should
not be presented as one.

Of the 16 documents that clear the auto-accept bar, the ontology check flags
4. Three are genuinely wrong, none of the flags is a false positive:

```
0.933  model said Radiology  ->  check says Cardiovascular / Pulmonary
       evidence 38.9 vs 0.0   on  left main, main, hemodynamics, obtuse marginal
0.892  model said Radiology  ->  check says Cardiovascular / Pulmonary
       evidence 45.9 vs 0.0   on  left main, right coronary, anterior descending,
                                  ventriculography
0.858  model said Radiology  ->  check says Cardiovascular / Pulmonary
       evidence 36.2 vs 2.5   on  pigtail catheter, femoral artery, left main
```

`left main`, `right coronary`, `anterior descending` and `ventriculography` are
coronary anatomy. The model read a catheterization report and called it
Radiology at 0.93; the vocabulary says otherwise by a margin of 39 to 0.

The same mechanism, the same thresholds file, a different corpus — and on
medical it is far more conservative than on legal: **0.5% false positives
against legal's 8.0%**, catching 2 of 8 rather than 3 of 5. Say that plainly
if asked. The operating point was never tuned for medical, and tuning it on
eight examples would be overfitting, not engineering.

## Why chunk accuracy is below 90.5%

A PDF is split into ~1000-character chunks that fit the model's window, and
each chunk is classified on its own. A clinical transcription is short and
front-loads its diagnosis, so a middle chunk can be a list of measurements
with the context stripped away. Legal judgments do not have this problem —
they are long and every chunk is still about the same case, which is why
`judgment-taxation.pdf` scores 32/33.

This is a property of the documents, not a defect in the extractor. Say so.

## Re-measuring

These numbers come from the real path — `pdf_extract.extract()` then
`model_service.predict()` then `ontology_check.check()`. The ontology catches
depend on where chunk boundaries fall, so **if you regenerate or edit these
PDFs, re-measure before quoting anything.**

## What not to touch on the medical tab

`predictions` carries a domain column (migration 007) and every query filters
on it, so medical uploads stay out of the legal review queue. `model_versions`
does **not** — one table, no domain column, one global single-active index.

So on `:5181`: use **Annotate, Review, Ontology**. Do not open **Admin**, whose
version table shows the legal row, and never press **Retrain**, which would
deactivate the legal baseline in the shared table and leave the legal backend
trying to load medical weights on its next restart.

Admin and Retrain belong on the legal tab. That split matches the demo anyway:
legal does the CSV → review → retrain → compare story, medical does PDFs.
