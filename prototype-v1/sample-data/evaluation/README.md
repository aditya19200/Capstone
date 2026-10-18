# Evaluation data

## `heldout_manifest.csv` — the only genuinely held-out set

21 documents the model has never seen: 4 Supreme Court judgments and 17
contracts and statutes. Labels come from the documents themselves — SCR
headnotes for the judgments, document type for the contracts — not from
anyone's reading of the text.

`heldout_SKIPPED.md` lists nine further documents that were available and
deliberately left unlabelled because their domain is genuinely arguable.

### Result (baseline model, Oct 2026)

```
document-level accuracy    17/18 = 94.4%
mean passage agreement     84.5%
majority-class baseline    50.0%
1,477 passages from 597 pages
```

Three of the 21 were refused by the scanned-PDF guard — `NDA template.pdf`
and both `wrongful termination` files have no text layer. That was not
planned; they are genuinely scans, and the guard caught them.

### Read these before quoting the number

- **Document-level is the easy measure.** Aggregating ~80 passages hides
  individual errors. 84.5% passage agreement is the harder, fairer figure.
- **18 documents, 5 of 10 domains, badly imbalanced** — half are Contract
  Law. This says nothing about Constitutional, Corporate, IP, Family or
  Civil Procedure.
- **The one miss is arguable.** The Indian Stamp Act was labelled Taxation
  here and predicted Property. The Act governs duty on conveyances, leases
  and mortgages, so the prediction is defensible and the gold label is the
  weaker of the two.
- Growing this set means more SCR judgments. Their headnotes state the
  subject matter, so the label comes from the document rather than from us.

## `stress_test.csv` — NOT a test set

225 rows, every one of which is also in `datasets/v1.csv` — verified by exact
match after normalisation. It is a curated subset of the training corpus,
useful for probing difficulty, and **not** an independent measurement.

Numbers from it were previously quoted as if they were held-out. They are
not. Use `heldout_manifest.csv` for any claim about generalisation.
