# Demo files — the run, in order

```
legal-csv/      the retraining demo        1-intake.csv, 2-probe-set.csv
legal-pdfs/     after retraining           judgment-labour.pdf, judgment-criminal-pradeep.pdf
medical-pdfs/   the second domain          neurology-consults.pdf, cardiology-catheterization.pdf
```

Everything else that used to live here is in `sample-data/archive/`.

## Before you start

```
./run_local_demo.sh              local: everything on this laptop, no internet,
                                 uploads in seconds. Starts empty every time and
                                 adds the 55 confirmations Retrain needs.

./run_local_demo.sh --live       the hosted Supabase and Neo4j. Same app, but every
                                 uploaded row is a few round trips (~2 s a row).
./run_local_demo.sh --live --reset   the same, after wiping the last rehearsal
```

Wait for the terminal to print "Open this in your browser" before clicking.
Both modes use the same ports: Ctrl+C one before starting the other.

## 1 — Legal: retraining (legal tab, :5180)

| # | File | Rows | What happens |
|---|------|-----:|--------------|
| **1** | `legal-csv/1-intake.csv` | 53 | 26 classify correctly · 5 are genuinely low confidence · 22 are confident and wrong. Correct the flagged ones in the Review screen |
| **2** | `legal-csv/2-probe-set.csv` | 8 | Upload **before** retraining, then **again after**. These are the before/after. **Never correct these** — they are the held-out test |

```
1  Annotator  upload legal-csv/1-intake.csv
2  Reviewer   correct the flagged items           (one click each; the 23 Oct rehearsal corrected 22)
3  Annotator  upload legal-csv/2-probe-set.csv    <- "before"
4  Admin      Retrain -> Activate
5  Annotator  upload legal-csv/2-probe-set.csv    <- "after"
6  Admin      Compare -> Compare
```

The eight probes change from Family Law to Criminal Law, and none of them was
ever corrected. The Admin table shows the new version's accuracy on `datasets/v1.csv` minus
any rows it trained on (1,697 in rehearsal), with the old model's score on
the same rows under it.

Every CSV carries a `gold_label` column as an answer key. The uploader reads
only the `text` column, so those answers never reach the model.

## 2 — Legal: PDFs, after Activate (legal tab)

See `legal-pdfs/README.md`.

```
7  Annotator  upload legal-pdfs/judgment-labour.pdf            it just works: 11-12 of 12
8  Annotator  upload legal-pdfs/judgment-criminal-pradeep.pdf   26-28 of 28 Criminal Law
```

Do not correct anything from these two in Review — they are not part of the
retraining, and corrections would become training data for a next retrain.

## 3 — Medical (switch to Medical in the header, :5181)

See `medical-pdfs/README.md`. No retraining on medical.

```
9   Admin      the version table: Medical Baseline 89.4%, Medical Round 1 90.8% (serving)
10  Annotator  upload medical-pdfs/neurology-consults.pdf          1 of 19 auto-accepted, 6 in Review
11  Annotator  upload medical-pdfs/cardiology-catheterization.pdf  1 confident error flagged
12  Reviewer   the queue: low-confidence items and the flagged one, with its evidence
```
