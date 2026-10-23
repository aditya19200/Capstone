# Legal demo PDFs — after retraining

Two files, uploaded through **Annotate → Upload File** on the legal tab
**after Retrain → Activate**, so the retrained version classifies them.
The other four legal PDFs are in `sample-data/archive/legal-pdfs/`.

Numbers below are from the 24 Oct rehearsal: the demo's 77 training rows
retrained offline exactly as the worker does, then each file run the way an
upload processes it (passages packed to 256 tokens, batched prediction, the
ontology check on everything at or above 0.85). Retraining is not perfectly
repeatable, so live numbers can differ by a passage or two.

| Upload | File | Pages → passages | After retraining |
|---|---|---:|---|
| 7 | `judgment-labour.pdf` | 6 → 12 | 12/12 Labour & Employment, avg 0.95, 11 auto-accepted |
| 8 | `judgment-criminal-pradeep.pdf` | 12 → 28 | **28/28 Criminal Law**, avg 0.96, 26 auto-accepted |

## 7 — it just works

A labour judgment, start to finish, through the same uploader and worker as
the CSVs. Nothing clever happens, which is the point: a real document, split
into passages, every one classified correctly.

## 8 — the one that makes the argument

*PRADEEP v. STATE OF HARYANA*, a Supreme Court murder appeal (IPC 302/324/449)
that turns on the testimony of a child witness. It is not in any training
data, and the retrain never saw it.

**Before retraining** (the deployed model), five passages came out as
**Family Law** — the model sees "child" and "minor" and files a murder case
under Family Law:

```
0.977  0.968  0.963   above the 0.85 bar: would be auto-accepted unseen.
                      The ontology check flags all three back to Criminal Law,
                      on testimony, oath, the evidence.
0.844  0.844          below the bar: go to review on low confidence.
```

**After retraining** on the reviewers' corrections — which were about
domestic-setting criminal cases, not this judgment — all 28 passages are
Criminal Law. The same mistake, fixed on a document the loop never saw.

Say it in that order: the system caught the confident errors before, and the
corrections it collected fixed them after.

## Scanned PDFs

Refused, deliberately. A scan has no text layer, so extraction returns
nothing, and ingesting it would queue blank documents. Reading them needs OCR,
which this does not do. To show the refusal, upload any scanned PDF and read
the error.
