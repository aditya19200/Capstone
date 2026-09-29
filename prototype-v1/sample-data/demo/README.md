# Demo files — upload in this order

The numbers are the order you upload them, not a version.

| # | File | Rows | What happens |
|---|------|-----:|--------------|
| **1** | `1-intake.csv` | 53 | 26 classify correctly · 5 are genuinely low confidence · 22 are confident and wrong. Correct at least 15 of the flagged ones in the Review screen |
| **2** | `2-probe-set.csv` | 8 | Upload **before** retraining, then **again after**. These are the before/after. **Never correct these** — they are the held-out test |
| — | `teaching-set-only.csv` | 25 | The teaching rows on their own, without the surrounding batch. Not needed for the normal run |

## The run

```
1  Annotator  upload 1-intake.csv
2  Reviewer   correct ~18 flagged items          (one click each)
3  Annotator  upload 2-probe-set.csv             <- "before"
4  Admin      Retrain -> Activate
5  Annotator  upload 2-probe-set.csv again       <- "after"
6  Admin      Compare -> Compare
```

Seven or eight of the eight probes change from Family Law to Criminal Law, and
none of them were ever corrected.

15 corrections is the measured minimum for all eight; 5 and 10 both give 7/8.

Every file carries a `gold_label` column as an answer key. The uploader reads
only the `text` column, so those answers never reach the model.
