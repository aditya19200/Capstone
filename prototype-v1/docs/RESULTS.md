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
- `concept_markers.json` and `concept_seed.cypher` are regenerated
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

## 9. Sentences that are safe to say

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

## Sentences that are not

- ~~"94.4% accurate"~~ without saying document-level, on 18 documents, 5 domains
- ~~"our stress test shows…"~~ — it is a subset of the training corpus
- ~~"it detects Hindi"~~ — it does not; it is merely uncertain
- ~~"fine-tuned to suit the domain perfectly"~~
- ~~"it annotates diagnoses"~~ — this is single-label document classification,
  not entity extraction
