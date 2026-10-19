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

## 9. The correction loop on a second domain

`backend/retrain_experiment.py`. Measured 19 Oct 2026.

**Setup, which must be stated whenever these numbers are quoted.** The
1,738-row medical corpus is split 1,391 train / 347 "incoming queue",
stratified by label, seed 42. A baseline is trained on the train part only,
so it is a *weaker* model than the shipped checkpoint (68.3%), which saw all
1,738 rows. The baseline then predicts the queue: 238 rows agree with the
gold label (confirmations), 109 disagree (corrections). No label is invented
anywhere — "the reviewer" is the dataset's own specialty field, so this
measures the loop's mechanics and not our medical opinion. Both checkpoints
are measured on the same 435-row held-out file, which no stage touches.

Three retrain recipes, one baseline, one held-out set:

| | accuracy | conf. right | conf. wrong | confidently wrong | ECE | wrong >0.85 |
|---|---|---|---|---|---|---|
| v1 baseline (1,391 rows) | 0.6782 | 0.830 | 0.610 | 7.8% | 0.083 | 17 |
| **A** all 347, 3 epochs | 0.6920 | 0.857 | 0.653 | **9.2%** | **0.096** | **22** |
| **B** corrections only (109), 3 ep. | **0.7172** | 0.735 | 0.594 | 7.3% | 0.070 | 8 |
| **C** all 347, 1 epoch | 0.6943 | 0.797 | 0.578 | **5.8%** | **0.059** | **11** |

**Recipe A is what the shipped worker would actually run** — 3 epochs on
every validated annotation (`workers/retrain_worker.py`) — and it is the
worst of the three. It buys +1.4pp of accuracy and pays with *more*
confidence on the predictions it got wrong (+4.3pp) than on the ones it got
right (+2.8pp), a higher confidently-wrong rate and worse calibration. That
is the signature of a model that got louder rather than better, and it is
why the confidence columns are in this table at all. An accuracy-only
comparison would have reported recipe A as a success.

**Recipe B has the best headline and should not be used.** Fine-tuning on
only the corrected rows shifts the class prior toward whatever was corrected
— 39 of the 109 corrections were Radiology — so Radiology F1 goes 0.302 →
0.594 (+29.2pp) and Orthopedic 0.693 → 0.811, while Neurosurgery loses
14.8pp, Neurology 13.0pp and Nephrology 12.6pp. It trades classes rather
than improving the model. The aggregate hides that; per-class F1 does not.

**Recipe C is the result to quote.**

| | v1 | v2 | |
|---|---|---|---|
| held-out accuracy | 0.6782 | 0.6943 | +1.6pp |
| mean confidence | 0.759 | 0.730 | −2.9pp |
| confidently-wrong rate | 7.8% | 5.8% | **−2.0pp** |
| ECE | 0.083 | 0.059 | **−2.4pp** |
| wrong above auto-accept | 17 | 11 | **−6** |

Accuracy up, confidence *down*, calibration better, six fewer wrong
predictions reaching auto-accept. Per-class: Neurosurgery +15.6pp, Urology
+4.4, Neurology +4.0, ENT +3.5, Orthopedic +2.7, Radiology +1.4; only
Ophthalmology (−3.7) and Obstetrics (−1.2) lost ground.

**The finding is about the system, not the model:** the retrain worker's 3
epochs is too many when fine-tuning an already-converged checkpoint on a few
hundred rows. One epoch is both more accurate and better calibrated. The
worker's hyperparameters had never been measured against an alternative,
because until there was a second domain there was nothing to measure them
on.

**Caveat on size.** +1.6pp sits against a training-run variance of ±0.7pp
(section 8) — about twice the noise, not comfortably clear of it. The
calibration movement is the more interesting number, but run-to-run variance
of ECE has *not* been measured, so quote it as observed on this pair of
checkpoints rather than as a guaranteed effect. Repeating recipe C with
three seeds is the honest way to firm either number up, and has not been
done.

**Not done:** nothing in the pipeline refits the calibration temperature
after a retrain. `settings.CALIBRATION_TEMPERATURE` is one global constant
shipping at 1.0 (off), not per-domain and not re-measured, so a retrained
model inherits whatever calibration the previous one was given. Recipe A
shows that is exactly when calibration moves.

---

## 10. Confident errors in medical, and a guard that could never fire

`backend/confident_error_analysis.py` → `metrics/medical_confident_errors.json`.
Shipped medical checkpoint, 435 held-out rows, 19 Oct 2026.

```
accuracy                          297/435   68.3%
top-2 accuracy                    401/435   92.2%
above CONFIDENCE_HIGH (0.85)      181/435   41.6%
  of those, wrong                     5      2.8%
  of those five, gold was runner-up   3     60.0%
below the threshold, wrong          133
  of those, gold was runner-up      101     75.9%
```

**Medical is far safer at auto-accept than legal.** 2.8% of confident
medical predictions are wrong, against roughly one in ten for legal
(section 3). This reframes section 4: the ontology check catching 0 of 5
planted medical errors is a smaller loss than it first appears, because
there are only five confident errors in the entire held-out set to catch.
Medical's errors are overwhelmingly *low-confidence* errors — 133 of 138 —
and those the routing already escalates to a reviewer. The failure mode the
ontology check exists for is one this domain barely exhibits.

The five confident errors are exactly the shared-vocabulary pairs predicted
in section 4: Neurosurgery→Orthopedic, Nephrology→Urology,
Radiology→Cardiovascular, and the Gastroenterology/Cardiovascular pair in
both directions.

**Top-2 accuracy is 92.2% against a top-1 of 68.3%.** When the model is
wrong the right answer is usually its second choice. That is a product
finding rather than a model one: showing a reviewer two candidates instead
of one puts the correct label in front of them nine times out of ten.

### The margin guard could not fire

`services/active_learning.py` overrode AUTO_ACCEPT when confidence cleared
0.85 but the top-2 margin was under 0.10. **The branch was unreachable.** A
softmax sums to one, so the runner-up is at most 1 − p1, so

```
margin = p1 − p2  ≥  p1 − (1 − p1)  =  2·p1 − 1
```

At p1 > 0.85 the margin is necessarily above 0.70. It cannot be under 0.10
unless p1 is under 0.55, and below 0.55 the ROUTE_TO_REVIEWER rule has
already fired. The guard's own docstring gave its motivating example as
"0.86 vs 0.84", which sums to 1.70 and is not a probability distribution,
and the unit test covering it passed confidence 0.90 with margin 0.05 as two
unrelated scalars — which is the only place such a prediction has ever
existed.

Measured rather than argued: of the 181 predictions above the threshold the
smallest margin observed was **0.7727**, and the guard fired **zero** times.

Removed 19 Oct. It changes no behaviour, because unreachable code has none.
What it changes is the claim: **there is no margin-based protection on
auto-accept and there never was.** Two tests now assert the arithmetic bound
so it cannot be reintroduced, and the API-level test that asserted the dead
branch has been inverted and kept as the regression test.

**If asked what does protect auto-accept:** the threshold itself; the
ontology check, in domains whose classes contradict each other; and — on the
evidence of section 9 recipe C — retraining in a way that stops wrong
predictions reaching the threshold at all, which is the only one of the
three that measurably improved.

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

> "On the medical domain the loop raised held-out accuracy 1.6 points while
> *lowering* mean confidence and cutting confidently-wrong predictions by
> two points. The model got better, not louder — and we report the recipe
> that failed that test alongside the one that passed."

> "Our retrain worker runs three epochs. We measured one epoch against it on
> a second domain and one epoch won on both accuracy and calibration. We
> only found that because we added a domain."

> "2.8% of confident medical predictions are wrong, against about one in ten
> for legal. The two domains fail differently, so they need different
> safeguards, and we measured which one each needs."

> "Top-1 is 68.3% and top-2 is 92.2%, so a reviewer shown two candidates
> finds the right label nine times out of ten."

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
- ~~"retraining improved the medical model"~~ without naming the recipe —
  the shipped worker's recipe made calibration *worse*
- ~~"Radiology F1 went from 0.30 to 0.59"~~ — true only for recipe B, which
  bought it by losing 12–15pp on three other classes
- ~~"the loop is calibrated after retraining"~~ — nothing refits the
  temperature; it ships at 1.0 and is not per-domain
