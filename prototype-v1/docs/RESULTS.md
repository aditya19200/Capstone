# Every measured number, and what it does not say

One place for all of it, so nothing in the report or the viva rests on
memory. Each figure is followed by the caveat that belongs to it. If a number
is quoted without its caveat it will be wrong in a way that is hard to
recover from in the room.

Measured 18 Oct 2026 against the deployed checkpoint
(`backend/models/LegalModelShared`, InLegalBERT, 10 classes, 256 tokens).

---

## 1. The evaluation problem, found and fixed

Until 18 Oct every number this project had came from `datasets/v1.csv`.

- `sample-data/evaluation/stress_test.csv` was quoted as a separate test. It
  is **225 rows, every one of which is also in v1.csv** — verified by exact
  match after normalisation. A subset of the training corpus, not an
  independent measurement.
- `evaluate_baseline.py` already warned that the deployed checkpoint's
  provenance was never recorded, so v1.csv **may be its training data**.

**If asked "what was your test set", the honest answer before 18 Oct was
"we didn't have one."** Now there is one.

### The held-out set

`sample-data/evaluation/` — 21 documents the model has never seen. Four
Supreme Court judgments, seventeen contracts and statutes. Labels come from
the documents themselves: SCR headnotes, and document type for the contracts.

`heldout_SKIPPED.md` lists nine further documents deliberately left
unlabelled because their domain is genuinely arguable — the IT Act spans
cybercrime and e-contracts, non-compete clauses are both Contract and Labour.

**What it is not:** 21 documents, 5 of 10 domains, badly imbalanced — 10
Contract Law, 7 Labour, 2 Taxation, 1 Property, 1 Criminal. It says nothing
about Constitutional, Corporate, IP, Family or Civil Procedure.

---

## 2. Accuracy

| measure | value | |
|---|---|---|
| **Document-level, held-out** | **17/18 = 94.4%** | the headline |
| Mean per-document passage agreement | 84.5% | each document weighted equally |
| Pooled passage accuracy | 55.2% | **misleading — see below** |
| Majority-class baseline | 50.0% | always guessing "Contract Law" |
| Baseline on `v1.csv` | 70.7% | upper bound, possibly training data |

1,477 passages from 597 pages.

**Why the pooled figure is misleading:** the Indian Stamp Act is 356 pages
and splits into 821 passages — **56% of the whole set by itself**. One
document decides the pooled number by weight of size. Quote 94.4%
(document-level) or 84.5% (per-document passage), never 55.2% without
explaining it.

**Document-level is the easier measure.** Aggregating ~80 passages smooths
out individual errors. Say which one you are quoting.

### The single miss

`Indian Stamp Act 1899` — labelled **Taxation**, predicted **Property**. The
Act governs duty on conveyances, leases and mortgages, so the prediction is
defensible and our gold label is the weaker of the two. Left as a miss:
changing a label after seeing the answer is how a test set stops being one.

### Three documents never reached the model

`NDA template.pdf` and both `wrongful termination` files are scans with no
text layer. The scanned-PDF guard refused them during the evaluation run —
not planted, not expected.

---

## 3. Calibration — is the confidence honest?

`backend/calibrate.py`. Fitted on half the data, measured on the other half.

| set | n | accuracy | mean conf | ECE before | ECE after | T |
|---|---|---|---|---|---|---|
| **held-out, ex-Stamp Act** | 656 | 0.845 | 0.920 → 0.858 | **0.083** | **0.060** | 1.313 |
| held-out, all | 1477 | 0.563 | 0.866 | 0.315 | 0.066 | 2.182 |
| `v1.csv` | 1701 | 0.696 | 0.847 | 0.155 | 0.071 | 1.592 |

**Use row 1.** Row 2 is 56% one disputed document and reports an error nearly
four times higher on that basis alone. Row 3 may be training data.

### The number that matters

```
confidence band 0.9–1.0:   254 passages   claimed 0.979   actually right 0.890
```

> **When the model says 98%, it is right 89% of the time.**

Auto-accept fires above 0.85, so **roughly 1 in 10 auto-accepted predictions
is wrong** — and confidence can never surface them, because being confident
is what put them there. That is the thesis of this project, with a rate
attached instead of an anecdote.

**Temperature scaling cannot change accuracy.** Dividing every logit by a
constant does not change which is largest. It makes the model honest, not
better. Shipped as `CALIBRATION_TEMPERATURE`, default **1.0 (off)**, because
enabling it moves predictions across the routing thresholds those thresholds
were tuned against.

An aside worth a line: accuracy on `v1.csv` (69.6%) is **lower** than on
genuinely unseen documents (84.5%). If the checkpoint had memorised v1.csv
you would expect the opposite. Weak evidence against heavy contamination.

---

## 4. The ontology consistency check

> **Re-measured 19 Oct, after the corpus rebuild of section 9.** The medical
> markers had been mined from the contaminated corpus (`sample_count: 1738`).
> Re-mined from the clean 1,472-row corpus: **1,179 markers across 12
> classes**, and the false-positive rate on the clean 273-row held-out set is
> **1.1%** (3 rows), against 1.6% before. Still an honest no-op: medical's
> thresholds are set where the check almost never fires, because it cannot
> catch this domain's errors (0 of 5, at thirty threshold combinations) and a
> check that only produces false positives is worth less than nothing.
>
> **Legal is unchanged and was regression-checked: 136/1,701 = 8.0% false
> positives, exactly as recorded below.** Re-mining medical touched only
> medical's files.


Catches confidently-wrong predictions, which uncertainty sampling is
structurally blind to. Independent of the model: markers are mined from human
gold labels, never from model output.

| | markers | false pos. | on Family Law | recall | Pradeep |
|---|---|---|---|---|---|
| before 18 Oct | 393 | 7.5% | 9.6% | 74.5% | **0/5** |
| **now** | **752** | **8.0%** | **11.8%** | **76.0%** | **3/5** |

Recall is measured on the 200 Criminal Law samples re-labelled Family Law —
the confusion it was built for. False positives are measured by feeding each
corpus sample its own gold label, so any flag is wrong by construction.

**What changed:** `TOP_N_PER_LABEL` was 40, an arbitrary cap discarding
two-thirds of the terms that passed every quality filter. Criminal Law had
188 eligible; 148 were binned, including `testimony` and `oath`.
`DOMINANCE_RATIO` rose 1.2 → 1.5 to pay for the extra evidence.

**Half a point more reviewer load for a class of error previously detected at
0%.** Across six real judgments the four correctly-classified ones raise **0
flags out of 87 documents** — the extra coverage did not arrive as noise.

### A negative result worth as much as the change

Giving each concept weighted edges to **several** domains — the obvious way
to say "child" belongs to Family and Criminal both — was tested across 64
combinations of edge-precision floor, support floor and both thresholds. Best
result: **67.5% recall at 8.2% false positives**, worse on both axes, and
0/5 on Pradeep. The concept layer stays one edge per term **because that
measured better**, not because a graph could not express more.

If asked why the knowledge graph is not multi-domain: *we tried it, here are
the numbers, it was worse.*

---

## 5. PRADEEP v. STATE OF HARYANA — the worked example

A Supreme Court murder appeal (IPC 302/324/449) turning on child-witness
testimony. Five passages classify as **Family Law at 0.84–0.98**: three at
0.96–0.98, above the auto-accept threshold, and two at 0.844, just under it.

The ontology check flags all three confident ones back to Criminal Law on
`testimony`, `oath`, `the evidence`. The two at 0.844 are below the bar, so
they are not filed silently: they get an explanation and are marked for one,
though at 0.844 they sit above the review queue's 0.55 cut-off and do not
appear in it. The check would not have flagged them — `child` genuinely
dominates their vocabulary. **Say that**; it is the honest limit of a
vocabulary-based check. Without the check, three would be accepted unseen.

Re-measured 24 Oct through the path an upload takes (passages packed by token
count, batched prediction). An earlier revision said four cleared the
threshold and two were missed.

### What the XAI–ontology comparison adds

| passage | verdict | |
|---|---|---|
| @0.977 | `no_overlap` | drivers were `tutored`, `age`, `tutoring` — not legal vocabulary at all |
| @0.968 | `overruled` | `child` pushed toward Family, `criminal` pushed **against**; model went with Family anyway |
| @0.963 | `mixed` | used `testimony`, a *Criminal* marker, to support Family Law |
| Taxation @0.990 | `aligned` | `tax act`, `tax` |
| Labour @0.994 | `aligned` | `labour` |

The verdict varies with the evidence rather than tracking confidence.

The three Pradeep rows were re-run 25 Oct at SHAP max_evals 500, the current
setting, with wordpieces merged into words: same verdicts. At the earlier
setting of 200, `susceptible` also ranked among the @0.977 drivers.

---

## 6. Retraining — the active-learning loop

Re-measured 19 Oct 2026 with `backend/compare_checkpoints.py`, after an
earlier revision of this section was retracted for quoting figures that could
not be sourced.

### What is retracted and stays retracted

70.7% to 75.3% on the 1,701-row set, ILC 52.4 to 59.3, MILPaC 41.9 to 66.7,
Family Law F1 0.690 to 0.807, "15 corrections is the measured minimum".
`75.3` appears in no metrics file, no script output and no model record. Gone.

### What is measured, and now has a source

The probe set — `sample-data/demo/legal-csv/2-probe-set.csv`, eight documents whose gold
label is Criminal Law and whose vocabulary reads as Family Law: Section 498A
cruelty, 467/471 will forgery, 494 bigamy, 361 kidnapping from guardianship,
criminal trespass, 379 theft, unlawful assembly, abetment of suicide. Marked
`PROBE - never correct these`.

Run through the deployed checkpoint and each of the five recorded retrain
checkpoints:

| checkpoint | validated rows it trained on | probes correct |
|---|---|---|
| `LegalModelShared` (deployed) | — | **0 / 8** |
| v-20260904-091128 | 47 | 0 / 8 |
| v-20260910-102339 | 48 | 0 / 8 |
| v-20260928-154348 | 60 | 7 / 8 |
| v-20260928-185817 | 65 | 7 / 8 |
| v-20260929-090239 | 65 | **8 / 8** |

**All eight of the deployed model's errors are confident errors**, at 0.985 to
0.996. Every one clears the 0.85 auto-accept threshold, so in production all
eight would have been labelled Family Law and filed with no human ever seeing
them. After retraining, all eight are correct, at 0.502 to 0.985.

**That was one run, and the count varies between runs.** On 24 Oct the demo's
82 validated annotations were retrained on ten times, offline, exactly as the
worker trains: five times as annotated and five with four mislabelled
corrections set to their right labels. The probes came out 8, 7, 7, 5, 4 of 8
and 6, 7, 7, 4, 4 of 8. Fixing the labels did not help; the spread is the
training's own randomness (dropout; the worker seeds the shuffle, not torch)
on very little data. Some misses are cautious (0.52-0.58) and some are not:
two runs filed two probes as Property at 0.91-0.96. The worker is left
unseeded on purpose, so a retrain shown twice is two real runs.

Example, verbatim:

```
gold  Criminal Law
v1    Family Law  @ 0.996   <- would have been auto-accepted
v2    Criminal Law @ 0.984
text  The accused was charged with abetting the suicide of his daughter-in-law
      following sustained harassment at the matrimonial home.
```

**Contamination checked before quoting any of this.** The eight probe texts
appear in none of `datasets/v1.csv` (1,701 rows), `sample-data/demo/legal-csv/1-intake.csv`,
`sample-data/archive/teaching-set-only.csv` or the inline corpus in
`seed_retrain_data.py` — zero overlap after normalisation. They are genuinely
held out from every path into the retrain corpus.

So the "0/8 to 7-8/8" claim retracted earlier **was true**. What was missing
was its provenance, which is a different failure and was the right thing to
retract for. Row 1 of the table is the measurement; `backend/compare_checkpoints.py`
regenerates it, and `backend/metrics/legal_probe_v1_vs_v2.csv` holds the rows.

### The incidental gain curve

The flip happens between 48 and 60 validated annotations: two checkpoints
below it score 0/8, three at or above it score 7/8 or better. That is weaker
than a designed sweep — the checkpoints differ in which annotations they got,
not only how many — but it is measured, and it bounds the useful answer to
"how many corrections does this take" at somewhere under 60 for this probe
set.

### The same checkpoint on the real held-out set

The probes show the loop fixing what it was aimed at. The obvious next
question is what it did to everything else, so the same two checkpoints were
run over the 21-document held-out set — 1,477 passages from 18 documents,
three refused as scans.

| | v1 deployed | v2 retrained | |
|---|---|---|---|
| **document-level** | **17/18 = 94.4%** | **17/18 = 94.4%** | unchanged |
| pooled passage | 55.2% | 50.9% | **−4.3pp** |
| passages fixed | — | 31 | 12 of them confidently wrong |
| passages broken | — | 95 | 2 of them confidently wrong |

**Document-level accuracy did not move, and pooled passage accuracy got
worse.** Both models miss the same single document, the Indian Stamp Act.

Where the breakage is concentrated:

| gold label | passages | fixed | broken |
|---|---|---|---|
| Taxation Law | 854 | 13 | **73** |
| Contract Law | 476 | 13 | 16 |
| Labour & Employment | 99 | 1 | 5 |
| **Criminal Law** | 28 | **4** | **0** |
| Property / Real Estate | 20 | 0 | 1 |

**Criminal Law — the class the corrections were about — improves and breaks
nothing.** 4 fixed, 0 broken, which is the probe result showing up on real
documents.

**77% of the damage is Taxation**, and Taxation here is 854 passages of which
821 are the Indian Stamp Act: one 356-page document whose gold label this
document already calls the weaker of two defensible readings (section 2).
Both checkpoints were already poor on it — v1 got 558 of its 854 Taxation
passages wrong before any retraining.

So the honest reading: **the correction loop fixed the failure it was aimed
at and did not improve the model overall.** That is the same shape as the
medical result in section 10 and the medical corrections-only recipe —
a small correction set moves the class it concerns and trades elsewhere.

**And this one is weaker evidence than the medical experiment**, which
should be said before it is quoted. `v-20260929-090239` is a demonstration
checkpoint from a September session with 65 ad-hoc annotations, not a
controlled run: there is no matched baseline, no seed sweep, and no record
of which 65 items were validated. Section 10 is the controlled measurement.

### What this does not claim

Eight probes, one class pair, chosen to be confidently wrong. It shows the
loop fixes the failure it was aimed at. It does **not** show overall accuracy
improving on legal — measured above, it does not.

---

## 7. PDF ingestion

- Text-layer extraction, split paragraph → sentence → word, packed to 256 tokens
- **99.9–100% of extracted characters retained** across six real judgments
- Nine-page judgment → 20 documents; Pradeep's twelve pages → 28
- Splitting is **not** by page: page breaks fall mid-sentence. One test
  judgment runs "bench of Orissa / High Court" across one
- **Scanned PDFs are refused.** No text layer means extraction returns
  nothing, and ingesting it would queue blank documents and classify them
  confidently. OCR is not implemented and is not claimed

### Hindi — graceful degradation

| | avg confidence |
|---|---|
| Indian Contract Act, English | 0.93 |
| Indian Contract Act, Hindi | **0.37** |

**181 of 206 Hindi passages route to review; only 2 clear auto-accept.**

Be precise: the system is **not detecting Hindi**. It has no language
identification. It is uncertain, and uncertainty is routed. The behaviour is
correct; the mechanism is simpler than it looks.

---

## 8. Engineering state

- **202 tests passing, 0 failing**
- Frontend builds clean; one pre-existing lint error in `AuthContext.jsx`
- Label set is one file per domain in `domains/`; adding a domain is a file,
  not a code change
- `<domain>_concept_markers.json` and the `.cypher` seeds are regenerated
  byte-identically by `graph/mine_concept_markers.py` — reproducible across
  hash seeds

### Known gaps — name them before anyone else does

- **Data lives in memory.** Restart the backend and annotations vanish. The
  real Supabase queries are written and gated behind `is_configured()`
- **Neo4j is not connected.** The ontology is a Python dict; the driver code
  is commented out in `services/neo4j_service.py`
- **No frontend tests**
- **No failure recovery for long jobs.** A crashed worker leaves `batch_items`
  in `processing` forever. `STUCK_JOB_MINUTES` and `MAX_ATTEMPTS` exist in
  the schema; no reaper uses them. This was found when a GPU driver mismatch
  killed an evaluation run mid-way and nothing noticed
- **One domain demonstrated.** The framework is built for more; that is
  architecture, not evidence, until a second domain runs

---

## 9. The medical corpus was contaminated, and fixing it moved everything

Found 19 Oct 2026, after sections 1-8 were written. **Every medical number in
an earlier revision of this document was measured against a broken held-out
set and has been replaced.**

### What was wrong

`sample-data/evaluation/medical_heldout.csv` shared 129 of its 435 rows with
`datasets/medical_v1.csv`. Overlap normally inflates a score. This one
deflated it, and the mechanism is the interesting part: **all 129 carried a
different gold label from their training copy. Not one agreed.**

mtsamples files the same transcription under more than one specialty — a
cervical discectomy as both Neurosurgery and Orthopedic, a foetal ultrasound
as both Obstetrics and Radiology. The old split separated the copies, so the
model was trained to answer A and then marked wrong for not answering B. No
level of competence scores on those rows.

| subset | rows | accuracy |
|---|---|---|
| full old held-out | 435 | 68.3% |
| the contaminated rows | 129 | **24.8%** |
| the clean remainder | 306 | **86.6%** |

379 of 1,745 unique transcriptions conflict this way, concentrated in exactly
the classes that looked worst: Radiology appears in 248 conflicts, Neurology
131, Orthopedic 110, Neurosurgery 74. **Radiology's 0.29 F1 was never the
model failing.** It was being graded against labels its own corpus
contradicts.

### The fix

`backend/build_medical_corpus.py`, which did not previously exist — the
corpus was built ad hoc, which is why this survived unnoticed. It groups rows
by normalised text so two copies cannot straddle the split, keeps conflicting
texts out of the held-out set, keeps them in training, and **asserts zero
overlap before writing**. Exclusions are documented in
`sample-data/evaluation/medical_heldout_SKIPPED.md`.

Conflicting texts stay in training because excluding them everywhere was
tried first and is not survivable: it leaves Radiology with 17 training rows
and Neurosurgery with 16. They are assigned to whichever competing specialty
has the fewest unambiguous examples — deterministic and written down rather
than correct, because the project has no clinician and will not invent a
clinical judgement.

Clean corpus: **1,472 train / 273 held-out, zero overlap, asserted.**

### What the model actually scores

Bio_ClinicalBERT, 12 classes, 3 epochs on the clean corpus:

```
held-out accuracy     90.48%   (273 rows)
before training        6.96%   (random = 8.33%)
top-2 accuracy         97.8%
```

| F1 | | F1 | |
|---|---|---|---|
| ENT - Otolaryngology | 1.00 | Nephrology | 0.90 |
| Ophthalmology | 1.00 | Neurosurgery | 0.89 *(n=4)* |
| Orthopedic | 0.97 | Neurology | 0.85 |
| Gastroenterology | 0.96 | Cardiovascular / Pulmonary | 0.84 |
| Urology | 0.95 | Hematology - Oncology | 0.73 |
| Obstetrics / Gynecology | 0.92 | **Radiology** | **0.50** *(n=4)* |

**This is not a model improvement and must never be presented as one.** The
architecture did not change. A test set that was asking unanswerable
questions was replaced with one that is not. The claim is "we fixed our
measurement", and the evidence for it is a builder script with an assertion
in it.

### A cost of the fix, measured rather than assumed

The conflict-assignment rule gave Radiology 248 extra training rows, which
inflated its prior. Radiology now has **recall 1.00 and precision 0.33** — it
has become a magnet class, and 4 of the model's 8 confident errors are
Cardiovascular predicted as Radiology.

Radiology and Neurosurgery have **4 unambiguous held-out rows each**. Neither
is reliably measurable in this corpus, and no split engineering changes that:
248 of Radiology's 269 documents are filed under a second specialty as well.
Reported as-is, with support alongside every F1, rather than dropped to make
the table tidier.

---

## 10. The correction loop on medical, measured honestly

`backend/retrain_experiment.py`. Re-run 19 Oct against the clean corpus.

Setup: 1,472 rows split 1,177 train / 295 "incoming queue", stratified,
seed 42. Baseline trained on the 1,177 for 3 epochs — a model somebody would
plausibly deploy, **not** an undertrained straw man. It predicts the queue:
**230 confirmed, 65 corrected**. The reviewer is the dataset's own specialty
field, so no label is invented. Both checkpoints measured on the same 273
held-out rows, which no stage touches.

| | baseline | 1 retrain epoch | 3 retrain epochs |
|---|---|---|---|
| held-out accuracy | 0.8938 | 0.8828 **(−1.1pp)** | 0.9011 **(+0.7pp)** |
| mean confidence | 0.875 | 0.866 | 0.910 |
| confidently-wrong rate | 3.8% | 5.5% | **6.4%** |
| ECE | 0.039 | 0.043 | 0.030 |
| wrong above auto-accept | 8 | 11 | **15** |

One epoch made it *worse*, so the worker's three epochs stay. An earlier
revision reported +1.6pp with improved calibration; that was measured on the
contaminated set, where 30% of the rows were unanswerable and created
headroom that does not exist.

### Three seeds, because one run cannot tell a gain from noise

+0.7pp sits exactly on the ±0.7pp training variance of section 8, so the
whole pipeline was re-run at three seeds. Each seed re-splits the corpus,
re-trains the baseline, re-works the queue and re-trains, so the queue
contents vary too and not only the floating-point order.

| seed | baseline | retrained | delta | ECE before | ECE after | wrong above 0.85 |
|---|---|---|---|---|---|---|
| 42 | 0.8938 | 0.9011 | +0.7pp | 0.0390 | 0.0302 | 8 → 15 |
| 1337 | 0.8974 | 0.9304 | +3.3pp | 0.0474 | 0.0452 | 6 → 7 |
| 7 | 0.8864 | 0.9121 | +2.6pp | 0.0452 | 0.0321 | 10 → 9 |
| 2024 | 0.8718 | 0.9267 | **+5.5pp** | 0.0654 | 0.0485 | 11 → 9 |
| 99 | 0.8901 | 0.9377 | +4.8pp | 0.0537 | 0.0361 | 10 → 6 |

```
baseline mean   0.8879        retrained mean  0.9216
accuracy delta  +3.37pp mean, sd 1.87pp, min +0.7pp, positive in 5 of 5
ECE             improved in 5 of 5, mean -1.17pp
wrong above 0.85  mean 9.0 -> 9.2, flat
```

Seeds were chosen before any of them was run — 42 because it was already the
project's seed, the rest arbitrary — and all five are reported. None was
discarded.

**The gain is real and bigger than one run suggested.** Seed 42 — the run
quoted everywhere above, and the only one measured at first — was the worst
of the three. The mean is +2.20pp against a ±0.7pp noise floor, every seed is
positive, and calibration error improved in every seed with no calibration
step applied.

**Quote +3.4pp across five seeds, positive in all five.** Not +0.7pp, which
was seed 42 alone, and not +5.5pp, which was the luckiest.

The confidently-wrong count is flat across five seeds, 9.0 → 9.2 on average.
Seed 42's 8 → 15 is the only run where it rose materially; the others gave
6 → 7, 10 → 9, 11 → 9 and 10 → 6. The earlier claim that retraining doubles
confident errors was an artefact of that single run.

**Where the loop does pay, and this is above the noise:**

| class | baseline | 3 retrain epochs | |
|---|---|---|---|
| Hematology - Oncology | 0.667 | 0.842 | **+17.5pp** |
| Neurosurgery | 0.727 | 0.889 | **+16.2pp** |
| Neurology | 0.733 | 0.788 | +5.5pp |
| Nephrology | 0.818 | 0.857 | +3.9pp |
| Urology | 0.943 | 0.963 | +2.0pp |
| Radiology | 0.500 | 0.375 | **−12.5pp** |

**The loop improves the tail, not the average.** The large classes were
already above 0.94 and had nothing to gain, and they dominate the aggregate,
so a real improvement in the weak classes shows up as +0.7pp overall. That is
the correct thing for active learning to do and the wrong thing to measure
with one number.

The honest summary is three sentences, not one:

> "Retraining on human corrections raised medical held-out accuracy by 3.4
> points on average across five seeds — positive in all five, against a
> training variance of 0.7 — and improved calibration error in all five. On
> legal the same loop fixed most of the eight probes it was aimed at — four
> to eight of them, depending on the run — and did not improve the model
> overall, and we report both."

### Do the corrections matter, or is it just more data?

The five-seed result says feeding 295 validated rows back improves the
model. It does not say *why*, and the obvious alternative is dull: v2 got
25% more training data, and any model improves with 25% more data. If that
is the whole story, the human in the loop is decorative.

Two arms, same seed, same baseline, **the same number of new documents**,
differing only in which documents:

| arm | gets | n |
|---|---|---|
| **C** | the rows the model got **wrong** | 38–46 |
| **D** | the same number of queue rows, drawn **at random** | 38–46 |

| seed | baseline | C corrections | D random | C − D |
|---|---|---|---|---|
| 42 | 0.8938 | 0.8498 | 0.8974 | −4.8pp |
| 1337 | 0.8974 | 0.9158 | 0.8901 | +2.6pp |
| 7 | 0.8864 | 0.9121 | 0.8864 | +2.6pp |

```
corrections vs baseline    +0.00pp mean, positive 2/3
random      vs baseline    -0.12pp mean, positive 1/3
CORRECTIONS MINUS RANDOM   +0.13pp mean, range -4.8 to +2.6
```

**Correcting the model's mistakes is not measurably better than adding the
same number of unselected documents.** The difference is +0.13pp against a
spread of seven points. That is nothing.

Said plainly, because it is the claim the project's own title leans on:
**this experiment does not demonstrate that human correction specifically is
what drives the improvement.** It demonstrates that feeding validated
annotations back improves the model, which is weaker.

Two honest qualifications, neither of which rescues the stronger claim:

- The test is **underpowered**. At 38–46 rows *neither* arm moves the
  held-out set at all (+0.00pp and −0.12pp), so of course they cannot be
  told apart. The +3.4pp needs the full 295-row feedback. A sharper test
  needs a bigger correction budget than this corpus has — there are only 65
  corrections to be had.
- One seed is not noise but a mechanism, below.

### Why corrections alone go wrong, which is the useful part

Seed 42's corrections arm fell 4.4pp *below* its own baseline. Per class:

| | baseline | corrections only | |
|---|---|---|---|
| Neurosurgery | 0.727 | 1.000 | **+27.3pp** |
| Nephrology | 0.818 | 0.900 | +8.2pp |
| Radiology | 0.500 | 0.353 | **−14.7pp** |
| ENT - Otolaryngology | 0.971 | 0.878 | −9.3pp |
| Cardiovascular / Pulmonary | 0.855 | 0.768 | −8.7pp |
| Gastroenterology | 0.919 | 0.849 | −7.0pp |

The weak classes improve enormously and the strong ones fall apart. The top
confusion afterwards is Cardiovascular predicted as Radiology, eight times.

The cause is the class prior. **Corrections are by definition concentrated
in the classes the model is bad at**, so a model trained only on them learns
that those classes are far more common than they are, and starts reaching
for them everywhere. It is the same effect as the corrections-only recipe
measured earlier on the contaminated corpus, and it reproduces here on clean
data.

**So the confirmations are not padding.** The 230 rows the model already got
right are what hold the prior in place while the 65 corrections do their
work.

State the strength of that carefully, because the bald version is wrong.
Across the three seeds the corrections-only arm averaged **+0.00pp** against
baseline and **two of the three were positive**, so "feeding corrections back
alone is harmful" is *not* what was measured. What was measured is that
corrections alone are **unstable** — and that when a seed goes wrong, it goes
wrong in this one specific, explainable way. The mechanism is the finding; the
average is not.

`workers/retrain_worker.py` trains on every validated annotation, not only
the changed ones. That was not a considered choice when it was written — it
is simply what `build_training_data()` does — but it is the right one, and
this is the measurement that says so.

> "Corrections alone are unstable: in one seed of three the weakest class
> gained 27 points of F1 while four strong classes lost between 7 and 15,
> because corrections are concentrated where the model is weak and that skews
> what it thinks is common. Fed back alongside the confirmations, the same loop
> gains 3.4 points across five seeds. The confirmations are what keep the
> corrections honest."

`backend/summarize_medical.py` prints this section, the five-seed table and
the control arm straight from the recorded metrics, so a figure on a slide can
always be traced back to one named file.

### Temperature scaling does not transfer to medical either

`backend/calibrate.py` was legal-only — it hardcoded the legal PDF manifest.
It now takes any labelled CSV (`--set csv`) and any checkpoint (`--model`),
so a retrained model can be calibrated separately from the one the domain
pack serves. Fitted on half the medical held-out set, measured on the other:

| checkpoint | accuracy | ECE before | ECE after | fitted T |
|---|---|---|---|---|
| medical baseline | 0.8978 | 0.0632 | 0.0633 | 0.854 |
| medical retrained | 0.8978 | 0.0475 | **0.0502** | 0.947 |

**It makes things very slightly worse, and both fitted temperatures are below
1.0** — meaning the fit wants the model to be *more* confident. Legal's is
1.313, which lowers confidence.

The reason is in the bins. Medical is badly **under**confident at the bottom
(the 0.3–0.4 bin is right 100% of the time, 0.5–0.6 is right 80%) and mildly
**over**confident at the top (0.9–1.0 claims 0.960, delivers 0.934).
Temperature is one scalar dividing every logit, so it moves every bin the
same direction. It cannot pull the top down and push the bottom up at once,
and the two errors largely cancel in the fit.

> "Temperature scaling corrects a model that is uniformly overconfident. Ours
> is overconfident at the top and underconfident at the bottom, so a single
> scalar has nothing to fix. Legal is uniformly overconfident and gets a real
> 0.083 to 0.060 improvement out of the same code."

That is the third technique measured to work on legal and not on medical,
after the ontology thresholds and the ontology check itself, and the third
time the cause is a property of the domain rather than a bug. **Medical needs
no calibration step:** retraining improved its ECE in all three seeds on its
own.

**Still not done:** nothing refits the temperature automatically after a
retrain. That matters for legal, where the temperature is real and a new
checkpoint inherits the old one's. It is one setting per checkpoint and the
script now prints it.

### Confident errors on the clean set

`backend/confident_error_analysis.py`, shipped checkpoint, 273 rows:

```
accuracy                          247/273   90.5%
top-2 accuracy                    267/273   97.8%
above CONFIDENCE_HIGH (0.85)      214/273   78.4%
  of those, wrong                     8      3.7%
  of those eight, gold was 2nd        7     87.5%
below the threshold, wrong           18
  of those, gold was 2nd             13     72.2%
```

**Top-2 accuracy is 97.8%.** When the model is wrong its second choice is
almost always right — 7 of the 8 confident errors and 13 of the 18 quiet
ones. Showing a reviewer two candidates instead of one is the cheapest
accuracy the system can buy, and it needs no model change.

### The margin guard could not fire

`services/active_learning.py` overrode AUTO_ACCEPT when confidence cleared
0.85 but the top-2 margin was under 0.10. **The branch was unreachable.** A
softmax sums to one, so

```
margin = p1 − p2  ≥  p1 − (1 − p1)  =  2·p1 − 1
```

At p1 > 0.85 the margin is necessarily above 0.70 — under 0.10 requires p1
under 0.55, where ROUTE_TO_REVIEWER has already fired. The guard's docstring
gave its motivating example as "0.86 vs 0.84", which sums to 1.70 and is not
a distribution. The unit test covering it passed confidence 0.90 with margin
0.05 as unrelated scalars, the only place such a prediction has ever existed.

Measured twice, on both the old and the clean held-out set: smallest margin
above the threshold **0.7428**, times the guard fired **zero**, out of 214.

Removed 19 Oct. It changes no behaviour, because unreachable code has none.
It changes the claim: **there is no margin-based protection on auto-accept
and there never was.** Two tests assert the bound so it cannot return, and
the API-level test has been inverted and kept as the regression.

**What does protect auto-accept:** the threshold; the ontology check, in
domains whose classes contradict each other; and refitting the model so
fewer wrong predictions reach the threshold — which on this evidence is the
one that still needs work.

### A second round of feedback, 23 Oct

`backend/medical_progression.py`, design written before it ran. The same
seed-42 split as above, but the 40% held back is fed in as two rounds of
about 295 notes, the way the retrain worker would receive them. Every
version is scored on the same 273 held-out notes.

| version | trained on | held-out | wrong above 0.85 |
|---|---|---|---|
| v0 baseline | 881 notes | 0.8938 | 3 |
| v1 | v0 + round 1 (296) | **0.9084** | 5 |
| v2 | v1 + rounds 1 and 2 (591), as the worker retrains | 0.8864 | 16 |
| v2b | v0 + rounds 1 and 2 (591), each note learned once | 0.9084 | 13 |

Round 1 gained 4 notes of 273. Retraining from v1, which is what the worker
does, made round 2 cost 6 notes and triple the confident errors: v1 had
already learned round 1, so those notes got six epochs in all, and mean
confidence rose 0.84 → 0.88 → 0.92. One fix was tested, committed before it
ran: start every retrain from the baseline. It recovered the
loss, tied round 1 exactly, and still made 13 confident errors.

The worker still retrains from the active version. Switching it to the
baseline has one run behind it and has not been made.

### What the medical tab serves, from 24 Oct

**Medical Round 1** (v1 above), with the round-0 model registered as Medical
Baseline, so the version table shows both. MedicalModelShared is retired: the
numbers in section 9 and earlier in this section describe it, not the model
now serving. Measured on round 1, the path the app takes (batched prediction;
`backend/metrics/medical_confident_errors_round1.json`):

| | MedicalModelShared (retired) | Medical Round 1 (serving) |
|---|---|---|
| held-out accuracy | 247/273, 90.5% | 248/273, 90.8% |
| top-2 accuracy | 97.8% | 97.1% |
| auto-accepted, and wrong among them | 214, 8 (3.7%) | 209, 5 (2.4%) |
| ontology check on those confident errors | caught 2 of 8, 1 false positive | caught 0 of 5, 0 false positives |

Round 1 makes fewer confident errors, and the check catches none of them:
they are five different confusions, where the old model's were mostly one
(cardiac catheterization filed as Radiology) that the markers know. The
demo PDFs were re-measured the same way (`backend/measure_demo_pdfs.py`);
their first figures had been measured with character-split passages, which
no upload produces.

---

## 11. Sentences that are safe to say

> "On 18 held-out documents the system had never seen — real Supreme Court
> judgments and commercial contracts — it assigned the correct domain to 17.
> The one miss is a statute where our own gold label is arguable."

> "When our model says 98% confident, it is right 89% of the time. About one
> in ten auto-accepted predictions is wrong, and confidence can never surface
> them, because being confident is what put them there."

> "Uncertainty sampling escalates what the model is unsure about. It is
> structurally blind to the opposite failure. In a real murder judgment,
> three passages were filed as Family Law at 96 to 98% confidence, high
> enough to be accepted with nobody looking. Our ontology check reads the
> vocabulary independently and flags all three."

> "We tested weighted multi-domain concept edges and they measured worse on
> both recall and false positives, so we kept the simpler representation."

> "We don't sell the model. We produce the annotated dataset and the system
> that makes it, and the retraining loop is how we show the output is
> training-grade."

> "Our medical held-out set shared 129 rows with training, and every one of
> them carried a conflicting label, because the source corpus files the same
> transcription under two specialties. We rebuilt the split so no document
> can straddle it, and the same architecture measures 90.5% instead of
> 68.3%. We did not change the model — we fixed a test set that was asking
> unanswerable questions."

> "Retraining raised medical held-out accuracy 3.4 points on average across
> five seeds, positive in all five against a training variance of 0.7, and
> improved calibration error in all five."

> "In one run, feeding back a round of validated annotations took the score
> on 273 unseen test notes from 89.4% to 90.8%. Across five runs, the average
> gain was 3.4 points, and every run improved."

> "Active learning improves the tail, not the average. Our large classes
> were already above 0.94 and dominate the aggregate, so the right way to
> measure the loop is per class."

> "2.4% of confident medical predictions are wrong, against about one in ten
> for legal. The two domains fail differently, so they need different
> safeguards, and we measured which one each needs."

> "Top-1 is 90.8% and top-2 is 97.1%, so a reviewer shown two candidates has
> the right label in front of them almost every time."

> "Two of our twelve specialties are not separable in this corpus, and we
> can show you the 248 documents that prove it."

> "We removed a safeguard after proving it could never fire. A softmax sums
> to one, so a high-confidence prediction cannot be a near-tie — the test
> that covered it was asserting on inputs no model can produce."

> "Eight documents our model called Family Law at 99% confidence are actually
> Criminal Law — IPC charges in a domestic setting. All eight would have been
> auto-accepted with no human seeing them. After retraining on corrected
> annotations, most of them come out right — between four and all eight,
> depending on the run. We checked first that none of the eight appears
> anywhere in the training corpus."

## Sentences that are not

- ~~"94.4% accurate"~~ without saying document-level, on 18 documents, 5 domains
- ~~"our stress test shows…"~~ — it is a subset of the training corpus
- ~~"it detects Hindi"~~ — it does not; it is merely uncertain
- ~~"fine-tuned to suit the domain perfectly"~~
- ~~"it annotates diagnoses"~~ — this is single-label document classification,
  not entity extraction
- ~~"retraining took legal from 70.7% to 75.3%"~~ — retracted 19 Oct, no
  source in the repo for any figure in that table
- ~~"15 corrections is the measured minimum"~~ — same, unsourced. The
  measured bound is "under 60 validated annotations" (section 6)
- ~~"our retrained model scored 100%"~~ — on an 11-row test split
- ~~"the margin guard catches confident near-ties"~~ — arithmetically
  impossible, fired 0/181, removed 19 Oct
- ~~"retraining improved medical by 0.7pp"~~ — that was seed 42 alone, the
  worst of five. The five-seed mean is +3.4pp, positive in 5/5
- ~~"retraining improves the model"~~ as a general claim — true on medical
  across five seeds, false on legal, where document-level accuracy was
  unchanged and pooled passage accuracy fell 4.3pp (section 6)
- ~~"human corrections are what improve the model"~~ — measured against a
  random sample of the same size, corrections won by +0.13pp across three
  seeds, which is nothing. Say "feeding validated annotations back", which
  is what was measured
- ~~"we retrain on the corrections"~~ — retraining on corrections *alone*
  made it worse, by skewing the class prior toward the weak classes
- ~~"the ontology check catches medical's confident errors"~~ — it caught 2
  of 8 for MedicalModelShared; for Medical Round 1, serving since 24 Oct, 0 of
  5 on the held-out set and 1 on the cardiology demo PDF
- ~~"every round of feedback improves it"~~ — on medical, round 2 lost 2.2pp
  retrained from round 1 and tied retrained from the baseline (section 10)
- ~~"retraining doubles our confident errors"~~ — 8→15 was seed 42 alone; the
  other two seeds gave 6→7 and 10→9
- ~~"we calibrate both domains"~~ — temperature scaling helps legal
  (0.083→0.060) and does nothing for medical (0.0475→0.0502)
- ~~"we improved medical from 68.3% to 90.5%"~~ — the model did not change;
  the test set was repaired. Say that, or there is nothing to point at
- ~~"all twelve specialties work"~~ — Radiology and Neurosurgery have four
  held-out rows each and Radiology's precision is 0.33
- ~~"the loop is calibrated after retraining"~~ — nothing refits the
  temperature; it ships at 1.0 and is not per-domain
