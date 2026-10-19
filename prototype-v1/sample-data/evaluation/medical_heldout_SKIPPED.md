# Medical transcriptions deliberately left out of both splits

379 transcriptions appear in mtsamples under more than one
medical specialty. The same text, filed twice, labelled differently.

They are excluded from the HELD-OUT set: a test set needs answerable
questions, and a row whose own corpus gives it two labels is not one. Same
reasoning as heldout_SKIPPED.md on the legal side.

They remain in TRAINING, assigned to whichever competing specialty has the
fewest unambiguous examples. Excluding them everywhere leaves Radiology with
17 training rows and Neurosurgery with 16, which is not a trade worth making.
The project has no clinician and does not invent clinical judgements, so the
rule is deterministic and written down rather than correct.

They were previously split across train and held-out, which meant 129 held-out
rows were scored against a label that contradicted the one the model had been
trained on. Those rows scored 24.8% against 86.6% on the clean remainder, and
dragged the reported accuracy from 86.6% down to 68.3%.

## The competing label pairs

| count | filed as |
|---|---|
| 65 | Cardiovascular / Pulmonary / Radiology |
| 64 | Neurology / Radiology |
| 47 | Neurosurgery / Orthopedic |
| 29 | Neurology / Orthopedic / Radiology |
| 26 | Orthopedic / Radiology |
| 25 | Neurology / Neurosurgery |
| 20 | Obstetrics / Gynecology / Radiology |
| 13 | Gastroenterology / Nephrology / Radiology |
| 10 | Hematology - Oncology / Obstetrics / Gynecology |
| 9 | Nephrology / Urology |
| 8 | Hematology - Oncology / Radiology |
| 7 | Gastroenterology / Radiology |
| 6 | Nephrology / Radiology |
| 5 | Neurology / Orthopedic |
| 4 | Gastroenterology / Hematology - Oncology |
| 4 | Cardiovascular / Pulmonary / ENT - Otolaryngology |
| 3 | Radiology / Urology |
| 3 | Gastroenterology / Urology |
| 3 | Hematology - Oncology / Neurology |
| 3 | Cardiovascular / Pulmonary / Hematology - Oncology |
| 3 | Cardiovascular / Pulmonary / Gastroenterology |
| 2 | ENT - Otolaryngology / Radiology |
| 2 | Hematology - Oncology / Ophthalmology |
| 2 | Cardiovascular / Pulmonary / Nephrology |
| 2 | ENT - Otolaryngology / Hematology - Oncology |
| 1 | Hematology - Oncology / Radiology / Urology |
| 1 | Obstetrics / Gynecology / Urology |
| 1 | Nephrology / Radiology / Urology |
| 1 | Gastroenterology / Hematology - Oncology / Urology |
| 1 | Ophthalmology / Radiology |
| 1 | Nephrology / Orthopedic / Radiology |
| 1 | Gastroenterology / Obstetrics / Gynecology / Radiology |
| 1 | Neurology / Neurosurgery / Orthopedic |
| 1 | Hematology - Oncology / Orthopedic |
| 1 | Neurology / Ophthalmology |
| 1 | Cardiovascular / Pulmonary / Obstetrics / Gynecology |
| 1 | Cardiovascular / Pulmonary / Neurology / Neurosurgery |
| 1 | Gastroenterology / Neurology |
| 1 | Cardiovascular / Pulmonary / Neurology |

## Per specialty

How many excluded transcriptions each specialty appears in. The totals exceed
379 because every conflict involves at least two specialties.

| specialty | appears in |
|---|---|
| Cardiovascular / Pulmonary | 80 |
| Orthopedic | 110 |
| Radiology | 248 |
| Gastroenterology | 33 |
| Neurology | 131 |
| Urology | 19 |
| Obstetrics / Gynecology | 33 |
| ENT - Otolaryngology | 8 |
| Neurosurgery | 74 |
| Hematology - Oncology | 35 |
| Ophthalmology | 4 |
| Nephrology | 32 |
