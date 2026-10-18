# Demo PDFs

Six files, six different things to show. Upload through **Annotate → Upload File**.
Measured numbers below are from the current model and thresholds; re-run
`backend/` against this folder if either changes.

| Upload | File | Pages → docs | What it demonstrates |
|---|---|---:|---|
| 1 | `judgment-labour.pdf` | 6 → 12 | **It just works.** 12/12 Labour & Employment, avg 0.94, 11 auto-accepted. Fast opener |
| 2 | `judgment-taxation.pdf` | 16 → 33 | **Holds up on a long document.** 32/33 Taxation, avg 0.96 |
| 3 | `contract-vendor-agreement.pdf` | 6 → 19 | **Not a judgment.** A commercial contract template → 19/19 Contract Law at 0.99 |
| 4 | `contract-employment.pdf` | 6 → 18 | Employment contract → Labour & Employment. Same pipeline, different document type |
| 5 | `judgment-criminal-pradeep.pdf` | 12 → 28 | **The showpiece.** See below |
| 6 | `hindi-contract-act.pdf` | 41 → 206 | **Graceful degradation.** See below |

## 5 — the one that makes the argument

*PRADEEP v. STATE OF HARYANA*, a Supreme Court murder appeal (IPC 302/324/449)
that turns on the testimony of a child witness.

Five passages come out as **Family Law at 0.84–0.98 confidence** — the model
sees "child" and "minor" and files a murder case under Family Law. Four are
above the 0.85 auto-accept threshold, so confidence-based active learning
never shows them to anyone.

The ontology check reads the vocabulary independently and flags three of them
back to Criminal Law, on `testimony`, `oath`, `the evidence`.

> Uncertainty sampling escalates what the model is unsure about. It is
> structurally blind to the opposite failure — being confidently wrong. This
> is that failure, on a real judgment, caught by a detector that never looks
> at the model's confidence.

Two of the five are still missed: passages where the vocabulary genuinely is
Family Law, `child` outweighing everything else. Say so — it is the honest
limit of a vocabulary-based check.

## 6 — what happens outside the model's competence

The Indian Contract Act in Hindi. InLegalBERT is English-only, so this is
input the model cannot read.

It does not fail loudly, and it does not confidently invent an answer:

```
English edition:  avg confidence 0.93
Hindi edition:    avg confidence 0.37  —  181 of 206 documents route to review
```

Only 2 of 206 clear the auto-accept bar. The routing layer holds: unusable
input becomes human work rather than silent wrong labels.

Be precise about the claim. The system is **not** detecting Hindi — it has no
language identification. It is uncertain, and uncertainty is routed. The
behaviour is correct; the mechanism is simpler than it looks.

## Scanned PDFs

Not included, deliberately — they are refused. A scan has no text layer, so
extraction returns nothing, and ingesting it would queue blank documents and
classify them confidently. Reading them needs OCR, which this does not do.
To show the refusal, upload any scanned PDF and read the error.
