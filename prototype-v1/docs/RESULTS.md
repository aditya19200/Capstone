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

> **Stale, flagged 19 Oct:** the medical figures in this section (21.1% FP
> with legal's thresholds, 1.6% after tuning, 0/5 at thirty combinations)
> were measured against the contaminated corpus described in section 9. The
> conclusion — that the check needs classes which contradict rather than
> share vocabulary — is unaffected, because it rests on which vocabulary the
> classes use and not on the split. The percentages have not been re-run.


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
testimony. Five passages classify as **Family Law at 0.84–0.98**; four are
above the auto-accept threshold.

The ontology check flags three of them back to Criminal Law on `testimony`,
`oath`, `the evidence`. Two are still missed — passages where `child`
genuinely dominates the vocabulary. **Say that**; it is the honest limit of a
vocabulary-based check.

### What the XAI–ontology comparison adds

| passage | verdict | |
|---|---|---|
| @0.977 | `no_overlap` | drivers were `tutored`, `age`, `susceptible` — not legal vocabulary at all |
| @0.968 | `overruled` | `child` pushed toward Family, `criminal` pushed **against**; model went with Family anyway |
| @0.963 | `mixed` | used `testimony`, a *Criminal* marker, to support Family Law |
| Taxation @0.990 | `aligned` | `tax act`, `tax` |
| Labour @0.994 | `aligned` | `labour` |

The verdict varies with the evidence rather than tracking confidence.

---

## 6. Retraining — the active-learning loop

Measured earlier in the project.

> **Caveat added 19 Oct:** the three corpus rows below (1,701 set, ILC,
> MILPaC) were measured before the held-out set of section 1 existed, so
> they are movements on data that may be the model's own training data. The
> held-out probe row is the only contamination-free line in the table, and
> it is eight probes. The medical equivalent, measured properly in section
> 10, came out at +0.7pp aggregate — so treat +4.6pp here as an upper bound
> and re-run it against `sample-data/evaluation/` before quoting it.

| | before | after |
|---|---|---|
| Held-out probes (never corrected) | 0/8 | 7–8/8 |
| Full 1,701 set | 70.7% | 75.3% |
| ILC subset | 52.4% | 59.3% |
| MILPaC subset | 41.9% | 66.7% |
| Criminal Law F1 | 0.646 | 0.726 |
| **Family Law F1** | 0.690 | **0.807** |
| stress_test | 68.9% | 68.0% |

**Family F1 went up**, which is the interesting part: the model stopped
over-applying Family Law rather than trading one class for another.

**15 corrections** is the measured minimum for 8/8 probes; 5 and 10 both give
7/8.

**stress_test went slightly down and that is reported as-is.** It is also not
an independent test set (section 1).

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

**The aggregate gain is inside the noise floor and must not be claimed.**
+0.7pp sits exactly on the ±0.7pp training variance from section 8. One
epoch made it *worse*. An earlier revision of this document reported +1.6pp
with improved calibration; that was measured on the contaminated set, where
30% of the rows were unanswerable and created headroom that does not exist.

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

> "Sixty-five corrections moved our two weakest measurable classes by 16 and
> 17 points of F1. Aggregate accuracy moved 0.7 points, which is inside our
> training variance, so we do not claim it. The cost was seven more wrong
> predictions reaching auto-accept, because we do not refit the calibration
> temperature after retraining, and that is the next thing to fix."

**Not done, and it is the same gap as before:** nothing refits
`settings.CALIBRATION_TEMPERATURE` after a retrain. It is one global constant
at 1.0, not per-domain. Both recipes raised the confidently-wrong rate, which
is exactly the condition temperature scaling exists to correct.

**Also not done:** three seeds. Every delta in this section is one run
against one split. The per-class movements are large enough to survive
±0.7pp; the aggregate is not, and is reported as not.

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

---

## 11. Sentences that are safe to say

> "On 18 held-out documents the system had never seen — real Supreme Court
> judgments and commercial contracts — it assigned the correct domain to 17.
> The one miss is a statute where our own gold label is arguable."

> "When our model says 98% confident, it is right 89% of the time. About one
> in ten auto-accepted predictions is wrong, and confidence can never surface
> them, because being confident is what put them there."

> "Uncertainty sampling escalates what the model is unsure about. It is
> structurally blind to the opposite failure. Our ontology check reads the
> vocabulary independently and catches three of five confident errors in a
> real murder judgment that confidence-based review would never have shown
> anyone."

> "We tested weighted multi-domain concept edges and they measured worse on
> both recall and false positives, so we kept the simpler representation."

> "We don't sell the model. We produce the annotated dataset and the system
> that makes it, and we prove the output is training-grade by retraining on
> it ourselves and measuring the gain."

> "Our medical held-out set shared 129 rows with training, and every one of
> them carried a conflicting label, because the source corpus files the same
> transcription under two specialties. We rebuilt the split so no document
> can straddle it, and the same architecture measures 90.5% instead of
> 68.3%. We did not change the model — we fixed a test set that was asking
> unanswerable questions."

> "Sixty-five corrections moved our two weakest measurable classes by 16 and
> 17 points of F1. Aggregate accuracy moved 0.7 points, which is inside our
> training variance, so we do not claim it."

> "Active learning improves the tail, not the average. Our large classes
> were already above 0.94 and dominate the aggregate, so the right way to
> measure the loop is per class."

> "3.7% of confident medical predictions are wrong, against about one in ten
> for legal. The two domains fail differently, so they need different
> safeguards, and we measured which one each needs."

> "Top-1 is 90.5% and top-2 is 97.8%, so a reviewer shown two candidates has
> the right label in front of them almost every time."

> "Two of our twelve specialties are not separable in this corpus, and we
> can show you the 248 documents that prove it."

> "We removed a safeguard after proving it could never fire. A softmax sums
> to one, so a high-confidence prediction cannot be a near-tie — the test
> that covered it was asserting on inputs no model can produce."

## Sentences that are not

- ~~"94.4% accurate"~~ without saying document-level, on 18 documents, 5 domains
- ~~"our stress test shows…"~~ — it is a subset of the training corpus
- ~~"it detects Hindi"~~ — it does not; it is merely uncertain
- ~~"fine-tuned to suit the domain perfectly"~~
- ~~"it annotates diagnoses"~~ — this is single-label document classification,
  not entity extraction
- ~~"the margin guard catches confident near-ties"~~ — arithmetically
  impossible, fired 0/181, removed 19 Oct
- ~~"retraining improved the medical model"~~ — aggregate gain was +0.7pp,
  inside the ±0.7pp training variance. Claim the per-class tail gains
  (+17.5pp Hematology, +16.2pp Neurosurgery) and say the aggregate is noise
- ~~"we improved medical from 68.3% to 90.5%"~~ — the model did not change;
  the test set was repaired. Say that, or there is nothing to point at
- ~~"all twelve specialties work"~~ — Radiology and Neurosurgery have four
  held-out rows each and Radiology's precision is 0.33
- ~~"the loop is calibrated after retraining"~~ — nothing refits the
  temperature; it ships at 1.0 and is not per-domain
