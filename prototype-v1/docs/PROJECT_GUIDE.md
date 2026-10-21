# Project guide — the whole system, explained

**XAI-Enabled Knowledge Graph-Driven Legal Annotation Framework**
Capstone Project ID 100 · PES University · branch `prototype-v1`

---

## What this document is

The single place that explains **how the system works and why it is built this
way**. It is the orientation document: read this first, then go to the others
for detail.

| document | what it holds |
|---|---|
| **this file** | architecture, mechanisms, plain-language explanations, demo runbook, terminology |
| `docs/RESULTS.md` | every measured number, its source, and what it does **not** say. 812 lines. The authority on all figures |
| `docs/HANDOFF_data_layer.md` | the live Postgres/Neo4j setup, split into a data-layer section and an ML-side section |
| `docs/SCHEMA_REQUEST.md` | the database schema as originally requested |
| `README.md` | the short version: what it does, how to run it |

Rule of thumb: **if it is a number, RESULTS.md is the authority.** Headline
figures are repeated here so this file stands alone, but anything contested
should be checked there, because RESULTS.md records the caveat alongside the
figure and this file does not have room to.

---

## The project in one page

### What it produces

**Gold labels for unlabelled documents, with an audit trail.**

You give it a pile of documents in your field. It labels them, shows the words
behind each decision, and flags the ones a human needs to look at. An expert
reviews that flagged subset — not the whole pile. Their decisions are the gold
labels, and they feed back as training data, so the flagged subset shrinks.

### The honest framing

It is **not** a replacement for labelled data. It needs a seed — the legal
model was fine-tuned on 1,701 labelled rows, and the concept markers were
mined from human gold labels. The claim is:

> **You label a seed. The system labels the pile.**

After that it is **self-sustaining**: the loop's training data is produced by
the loop. No second external dataset is ever needed. Bounded, though — it is
self-sustaining *within a fixed label set*. It cannot invent a 13th specialty.

### What it is competing against

**Manual annotation**, not another AI. The alternative to using this system is
a team of domain experts annotating every file by hand.

| | without this system | with it |
|---|---|---|
| documents an expert reads | all of them | the flagged fraction — 27 of 53 on the demo batch |
| why this label? | a person's memory | SHAP tokens + concept evidence on both sides |
| confident-but-wrong caught | nothing catches it | the ontology consistency check does |
| next batch | same effort | less |

### What is novel

Two things, and neither is "we fine-tuned a BERT."

1. **The ontology consistency check** — a second, model-independent route into
   the review queue that catches predictions which are confident *and wrong*.
   Confidence routing is structurally blind to those.
2. **Two domains on one codebase, with measured non-transfer** — four
   techniques work in one domain and not the other, in both directions. That
   is a research finding, not a limitation.

---

## Team and review

| person | owns |
|---|---|
| Ankush Gowda K R | ML pipeline — models, retraining, evaluation, the ontology/concept layer |
| Arjun S Kannur | data layer — Postgres schema, Supabase, Neo4j connection |
| Rentala Shiva Naga Aditya | backend; repo owner (`aditya19200/Capstone`) |
| Bidisha Paul | frontend (React) |

Guide: Dr. Kavitha P.

**Review 3: Monday 26 October 2026, 2:20–2:40 PM, 2nd Floor Conference Hall,
Panel 1.** Reviewers: Prof. Ankita, Prof. Asha Kumari A. Twenty minutes — which
is the single biggest constraint on what the demo can show.

---

## How the system works, end to end

```
   upload text  ──►  classify  ──►  explain (SHAP)
                        │
                 does it need a human?
                   ├── low confidence ────────────┐
                   └── ontology conflict ─────────┤
                                                  ▼
                                   human reviews and confirms or corrects
                                                  │
                                                  ▼
                       retrain on every validated annotation
                                                  │
                                                  ▼
                            new model version, saved INACTIVE
                                 (a human activates it)
```

### 1. Ingestion — three ways in, one pipeline

| endpoint | takes |
|---|---|
| `POST /batches/paste` | one document per line |
| `POST /batches/csv` | a `text` column, or a single-column file |
| `POST /batches/pdf` | a digital PDF |

Past the point where an upload becomes a list of strings, **nothing downstream
can tell them apart** — same queue, same routing, same ontology check, same
review screen, same export, same retraining.

**PDFs** are split into passages that fit the model's 256-token window,
breaking on paragraph then sentence boundaries — deliberately *not* on page
boundaries, which fall mid-sentence.

**All three normalise before storing.** This is not tidiness. The retrain
worker normalises before training, so text stored raw would be classified in
one form and trained on in another. It matters most for PDFs, whose text
arrives full of hyphen-breaks and soft line wraps.

**Scanned PDFs are refused on purpose.** A scan is a photograph of a page with
no text layer, and extraction returns empty strings rather than failing.
Ingesting one would queue blank documents, classify them confidently as
something, and quietly poison the annotation set. Reading them needs OCR, which
this does not do — so it says so instead of pretending.

**The uploader reads only the `text` column**, so answer-key columns in the
sample CSVs never reach the model.

### 2. Classification

`backend/services/model_service.py`. Weights load **once** at startup and are
reused; never reloaded per request. `predict()` is synchronous and CPU-bound,
so routers call it in a thread pool.

Returns: `predicted_label`, `confidence` (max softmax), `probabilities`
(all classes), `label_id`, `entropy`, `margin`.

**Label mapping comes from the domain pack** (`domains/<key>.json`), not from a
literal dict in the code. Order is not cosmetic — the id is the index into the
model's output logits, which is why `config/domains.py` refuses to load a pack
whose ids are not `0..n-1` in order, and why it must match `id2label` in the
checkpoint's own `config.json`.

**Activation really swaps the weights.** Activating a version loads it into the
running service, so the next prediction genuinely comes from the model you
promoted. `GET /health` reports the version actually resident rather than the
one configured. If the weights fail to load, the previously loaded model keeps
serving and `is_active` is rolled back.

**The admin table shows both facts, because they are two facts.** `is_active`
is the database's record of which weights *should* load; `is_serving` is what
the running process actually has in memory. They agree almost always, and the
moments they do not are the ones worth seeing — between an activation and a
restart, or after a startup that could not read a flagged version's files and
fell back to `MODEL_PATH`. The deployed baseline spent its whole life in that
table marked **Inactive** while answering every request on the screen behind
it, because registering it is an insert followed by an activation and only the
insert succeeded. A column that shows the flag alone cannot tell you that.

**What comes out of the system: the annotated corpus.** Admin → *Validated
corpus* exports every human-validated annotation, in two shapes assembled from
one `rows()` call so they cannot describe different data:

| | |
|---|---|
| `.jsonl` | one object per line, straight into `datasets.load_dataset("json", ...)`. Carries provenance per row: what the model predicted, how confident it was, whether the human corrected it, which checkpoint made the prediction. This is the file another system ingests, and it is literally what the retrain worker consumes. |
| PDF | the same records laid out to read: each document with its ontology concept markers shaded in place — green for vocabulary belonging to the validated label, amber for vocabulary belonging to a different one — the model's prediction and confidence, and the human's decision beside it. Opens with a dataset card: size, label distribution, and the limits that apply to this particular export. |

The amber highlighting under a corrected entry is the ontology check visible in
one glance: the markers disagreed with the model, the document went to a
reviewer, the reviewer changed the label.

### 3. Routing — two independent ways to reach a human

This is the conceptual core of the project.

| route | what it means | who noticed |
|---|---|---|
| `low_confidence` | "the model isn't sure" | the model, about itself |
| `ontology_conflict` | "the model is sure, and the words disagree" | independent evidence |

The first is standard active learning. **The second is the contribution.**

**Confidence routing** (`backend/services/active_learning.py`), in priority
order:

```
confidence > CONFIDENCE_HIGH (0.85)  →  AUTO_ACCEPT
confidence < CONFIDENCE_LOW  (0.55)  →  ROUTE_TO_REVIEWER
otherwise                            →  NEEDS_EXPLANATION
```

`requires_shap = routing != "AUTO_ACCEPT"`.

There was a fourth rule, a **margin guard**: override AUTO_ACCEPT when
confidence cleared the bar but the top-2 gap was under 0.10. It was removed
because **it could never fire** — the probabilities are a softmax and sum to
one, so the runner-up is at most `1 − p1`, which makes
`margin ≥ 2·p1 − 1`. Clearing a 0.85 bar forces a margin of at least 0.70.
Measured: 0 firings in 214 predictions, minimum observed margin 0.7428.

**The review queue** (`backend/routers/queue.py`) is assembled at read time:

```python
for r in predictions_repo.list_all():
    if r["confidence"] < REVIEW_THRESHOLD:        # 0.5
        all_rows.append((r, "low_confidence", None))
        continue
    conflict = ontology_check.check(r["text_content"], r["predicted_label"])
    if conflict:
        all_rows.append((r, "ontology_conflict", conflict))
```

Two consequences worth knowing:

- **Only confident predictions are ontology-checked.** Anything below the
  threshold is already queued on confidence alone.
- **The conflict is computed live and never stored.** It exists only in the
  queue response. `GET /batches/{id}/items` therefore carries *no* ontology
  information — `BatchItemOut` has `predicted_label`, `confidence`, `status`,
  `validated_label`, `prediction_id`, and nothing about conflicts. This matters
  for the demo; see the runbook.

If the markers file is absent the check degrades to a no-op and the queue falls
back to confidence alone, rather than failing.

### 4. The ontology consistency check

`backend/services/ontology_check.py`. **One component, two names.** The README
heading calls it "the second detector"; its actual name is the **ontology
consistency check**, and the UI badge says **"Ontology conflict."** "Second"
only means second *route into the review queue*.

**Why it exists.** Confidence routing can only escalate what the model
*doubts*. That makes it structurally blind to the opposite failure:
predictions that are confident *and wrong*. Those are auto-accepted, never
reach a reviewer, never become training data, and the error persists
indefinitely. **The model's own confidence is the thing deciding whether
anyone checks its work.**

**What it does.** It compares the vocabulary actually present in the document
against the label the model picked, and routes the document to a human when
the vocabulary points somewhere else — regardless of confidence.

**The plain-language version.** Picture a mail room. The **model** is a clerk
sorting letters into 10 pigeonholes: fast, and sure of itself every time. The
**ontology check** is a supervisor who never watches the clerk — the
supervisor just reads each letter for keywords. A letter saying *accused, IPC,
charged* sitting in the Family pigeonhole gets pulled out and put on a human's
desk. The supervisor never re-files it; it only says "someone should look at
this."

Two properties make that worth having:

- **The keyword list was built from human gold labels, never from model
  predictions.** A detector built on the model's own output would inherit the
  exact blind spot it exists to catch. Markers are mined by
  `graph/mine_concept_markers.py` from the corpus's gold labels.
- **It cannot overrule a label.** It only decides a human should look. A false
  positive costs reviewer time; it cannot corrupt data.

**The mechanism.** Each marker is a term with a precision-derived weight. For a
document, sum the weights of matched markers per label. A conflict requires all
three of:

```
some other label accumulated  >= min_evidence      (legal 5.0)
it did so on at least          min_matched_terms   (legal 2) distinct markers
and it outweighs the predicted label's evidence by dominance_ratio (legal 1.5)
```

Matching is word-boundary anchored, so `mark` does not fire inside `remarked`
and `ipc` does not fire inside `ipcc`.

Thresholds come from the domain pack, falling back to legal's. They were global
constants until a second domain arrived, and that was wrong: tuned on legal and
applied unchanged to clinical text they produced a 21.1% false-positive rate
against legal's 8.0% while catching 0 of 5 confident errors — strictly worse
than not running at all.

**Worked example** (probe 5 of the demo probe set):

```
text   : "The accused was charged with criminal trespass after forcibly
          entering his estranged wife's residence in violation of the
          restraining order."
model  : Family Law, 0.995 confident
Criminal Law evidence : 10.05   ['accused', 'the accused', 'criminal', 'accused was']
Family Law evidence   :  0.00
```

> The model is 99.5% certain this is Family Law, and the document contains not
> a single Family Law marker. **Confidence cannot see that. Evidence can.**

**What the reviewer sees** (`frontend/src/components/review/ReviewQueueItem.jsx`):

```
[ONTOLOGY CONFLICT]
The model was confident, but the legal concepts in this text point to Criminal Law.
Concept evidence:   Criminal Law 10.1    Family Law 0.0
Concepts found:  accused   the accused   criminal
Model certainty: HIGH        [Criminal Law ▼]  [Submit]
```

The suggested label is **pre-selected** when the ontology raised the flag,
because starting from the model's label would be actively unhelpful — that
label is the thing under question. The reviewer still has to press Submit, and
the banner says plainly that it is a suggestion. The evidence comparison is
shown so the flag is *checkable* rather than a verdict to be trusted.

**Measured cost and benefit (legal):** catches about **a third** of real
errors; **88%** of what it flags is a genuine mistake; reviewers see about
**8% more** documents than confidence alone would send them. Say it as: *"that
is the price of detecting a class of error the confidence thresholds detect at
0%."*

**Known limitation, stated plainly:** marker matching is shallow. It sees
vocabulary, not argument structure, and a document quoting a statute it is not
actually about can trip it.

**Also tested and rejected:** giving each concept weighted edges to *several*
labels instead of one — the obvious way to express that "child" belongs to
Family and Criminal both. Swept across 64 combinations; the best scored 67.5%
recall at 8.2% false positives, worse on both axes than the single-edge
scheme. The concept layer stays one edge per term **because that measured
better**, not because a graph could not express more.

### 5. Explanation (XAI)

SHAP token attributions. For a divorce case it surfaces *wife, divorce,
custody, children* as the words that drove the call. A SHAP job is queued
whenever routing is not AUTO_ACCEPT, and can also be requested on demand for
an auto-accepted prediction the user wants to inspect (`routers/explain.py`).

The review screen shows SHAP tokens *and* concept evidence, which are
independent signals — one from inside the model, one from outside it.

### 6. The retraining loop

`backend/workers/retrain_worker.py`.

**Statuses.** There are only three:

```python
AnnotationStatus = Literal["pending", "validated", "rejected"]
```

and both reviewer actions that keep a document collapse into one:

```python
action_to_status = {"accept": "validated", "modify": "validated", "reject": "rejected"}
```

So **`validated` = confirmations + corrections.** A "correction" is simply a
validated annotation where the reviewer chose a label different from the
model's. There is **no `is_correction` field anywhere** in routers, services,
workers or models — the system does not record which were which. It is
recoverable only by diffing `validated_label` against the prediction's
`predicted_label`.

There is **no second approval step on an annotation.** One reviewer touches it
once. The admin gate is somewhere else entirely: the admin activates a **model
version**.

**What the retrain trains on:**

```python
annotations = list_annotations(status="validated")   # one query, no filter
...
label = ann["validated_label"]
```

Every validated annotation, including the ones where the model was already
right. The gate is `MIN_ANNOTATIONS_DEFAULT = 50`.

**Nothing auto-creates an annotation.** `create_annotation` has exactly one
caller: the human review endpoint at `routers/annotate.py:117`. `AUTO_ACCEPT`
stores a prediction and skips SHAP; it writes no annotation row. **This is the
property that makes the demo's before/after honest** — a prediction cannot
become training data unless a human explicitly submits it.

### 7. Why it must train on the confirmations too

This is the part most worth understanding, and it is measured.

The demo's corrections all point one direction: → Criminal Law. **Two
completely different things would both fix the probe set:**

| | what the model learned | how it generalises |
|---|---|---|
| **decision boundary moved** | "dowry-harassment documents are Criminal Law" | correctly — affects only similar documents |
| **class prior shifted** | "say Criminal Law more often" | badly — starts calling contracts criminal |

Corrections alone teach the second. They are *by definition* concentrated in
the classes the model gets wrong, so a training set made only of them tells the
model to predict those classes more often.

The confirmations are spread across all labels and the model already predicts
them correctly, so per example their loss is near zero and their individual
gradient is small — **but their job was never to teach. Their job is to stop
the fix from becoming a new prior.** The standard name for this mitigation is
**rehearsal**: keeping old correct examples in the training mix.

**Measured, on medical, seed 42, corrections-only arm:**

```
Neurosurgery      +27.3pp     <- the direction being corrected toward
Nephrology         +8.2pp
Radiology         -14.7pp     <- everything else
ENT                -9.3pp
Cardiovascular     -8.7pp
Gastroenterology   -7.0pp
```

Brilliant at the one thing, worse at everything else. That is the signature of
a prior shift.

**State the strength of this evidence carefully:**

| arm | rows | result |
|---|---|---|
| all validated | 295 | **+3.4pp**, positive 5/5 seeds, ECE better 5/5 |
| corrections only | 38–46 | **+0.00pp** vs baseline |
| random, same count | 38–46 | −0.12pp |

Corrections minus random is **+0.13pp — nothing.** Row for row, corrections
are not special. What is real is the *shape* of seed 42, above. And only 65
corrections exist, so the arm is underpowered — **say so if asked.** The arm
was run on **medical, not legal**, so the mechanism is measured on the other
domain.

**Evidence the demo moved the boundary rather than the prior:** held-out
document accuracy held at **17/18** across the retrain. A prior shift would
have wrecked that number.

**Medical does not have a prior problem.** From
`backend/metrics/medical_retrain_experiment_s42.json`:
`retrain.validated_rows = 295`, `retrain.corrections_only = False`. Production
medical already trains on everything, via the same domain-free code path as
legal. The skew was *induced* for the control experiment, to show the
mitigation matters.

### 8. The safety rail

**A retrained model is always saved inactive.** Nothing promotes a model
automatically — a human looks at the numbers and decides. If a new model
scores more than 5 points below the active one it is still saved, but flagged
with a warning.

---

## The two domains

Adding the second domain is what found most of the bugs worth finding.

| domain | base model | labels | held-out accuracy |
|---|---|---|---|
| legal | InLegalBERT | 10 Indian legal areas | **94.4%** document-level, 18 documents |
| medical | Bio_ClinicalBERT | 12 clinical specialties | **90.5%**, 273 documents |

`ACTIVE_DOMAIN` in `backend/.env` picks which one is served. A domain is a
file: `domains/<key>.json` carries names, ids, descriptions, the model that
predicts them, the markers file, and the ontology thresholds.

**Legal labels (10):** Contract · Criminal · Constitutional · Corporate ·
Property / Real Estate · Family · Labour & Employment · Intellectual Property ·
Taxation · Civil Procedure / Other.

**Medical labels (12):** Cardiovascular / Pulmonary · Orthopedic · Radiology ·
Gastroenterology · Neurology · Urology · Obstetrics / Gynecology ·
ENT – Otolaryngology · Neurosurgery · Hematology – Oncology · Ophthalmology ·
Nephrology.

### Four techniques that do not transfer — the research finding

| technique | legal | medical |
|---|---|---|
| ontology consistency check | 3/5 confident errors caught | **0/5 at all 30 threshold combinations swept** |
| per-domain thresholds | 8.0% false positives | needed a higher bar: 21.1% → 1.6% |
| temperature scaling | T=1.313 helps | makes ECE **worse** |
| concept co-occurrence | CCI press-release boilerplate | **real clinical pairs** |

**Why the ontology check cannot transfer.** It works by finding vocabulary that
*contradicts* the chosen label. Legal's confusions are contradictory —
*"ipc, prosecution, punishable"* against *"child, marriage, custody."*
Medical's confusions are between specialties that **share** vocabulary: a spine
operation reads the same whether it is filed Orthopedic or Neurosurgery;
Urology and Nephrology both describe the kidney. **There is no contradiction to
find.** So medical ships the check at a high bar — 1.6% false positives and no
claim to catch anything: *"an honest no-op rather than an expensive one."*

**Why temperature scaling does not transfer.** Medical is *underconfident at
the bottom and overconfident at the top*, so a single scalar temperature
(0.854 / 0.947 measured) makes ECE worse. Legal is uniformly overconfident, so
T=1.313 helps. `CALIBRATION_TEMPERATURE` ships at **1.0 (off)**.

**Note the last row goes medical's way.** The finding is not "legal is better."
It is **two domains, one codebase, and techniques do not transfer uniformly in
either direction.**

---

## The knowledge graph

`graph/`. Three seed files per domain, loaded in order by `load_ontology.py`:

| file | contents | generated by |
|---|---|---|
| `<key>_ontology_seed.cypher` | the label nodes and their `RELATED_TO` edges | `mine_concept_markers.py` |
| `<key>_concept_seed.cypher` | mined concepts and their `INDICATES` edges | `mine_concept_markers.py` |
| `<key>_depth_seed.cypher` | `NARROWS` and `CO_OCCURS` between concepts — **optional** | `mine_graph_depth.py` |

```
(:Concept {term, precision, support}) -[:INDICATES {weight}]-> (:OntologyNode {label})
(:Concept) -[:NARROWS]-> (:Concept)                  whole-word containment
(:Concept) -[:CO_OCCURS {jaccard, documents}]- (:Concept)   document co-occurrence
```

Every statement is a `MERGE`, so re-running is safe. After loading, the script
stamps each of the pack's `OntologyNode`s with `domain: <key>` — several
domains share one database because Aura Free gives you only one, and without
the stamp a graph holding legal and medical nodes looks like one 22-label
ontology. It then counts what landed and exits non-zero if that differs from
the seed files, because the failure it guards against — edges that silently
attach to nothing — reports success otherwise.

**Depth mining results:** legal 84 `NARROWS` + 6 `CO_OCCURS`; medical 182 +
1,205. Two filters earn their place:

- stopword-only extras dropped (legal went 230 → 84 pairs)
- containment-related pairs excluded from co-occurrence, because
  `'intellectual' ↔ 'intellectual property'` at Jaccard 1.00 is arithmetic,
  not evidence

Per-domain Jaccard thresholds: medical 0.25, legal 0.45, fallback 0.35.
Legal's strongest co-occurrences are CCI press-release boilerplate
(`'available at' ↔ 'cci'`, 0.60). Medical's are real
(`'ankle tourniquet' ↔ 'sagittal saw'`, `'carpal ligament' ↔ 'transverse
carpal'`). **Co-occurrence is the one technique that goes medical's way.**

### What the serving path actually reads

**Nothing in the serving path reads the depth edges.** Verified:

- `ontology_check.py` reads `graph/<domain>_concept_markers.json` — a flat
  mirror of the concept layer, generated in the same run so the two cannot
  drift. It never opens Neo4j.
- `backend/services/neo4j_service.py` queries only `(:OntologyNode)` with
  `HAS_CHILD` / `RELATED_TO` / `HAS_PARENT`.

**Why the markers live in a JSON mirror rather than being queried from the
graph.** Each term currently carries exactly *one* outgoing `INDICATES` edge —
the miner assigns it to the single label it most often occurs in. So the
concept layer as it stands is expressible as a flat table, and the JSON is
exactly that. The graph representation earns its place only once concepts carry
edges to several labels at different weights, which is the honest next step.
Reading a mirror is also the arrangement the rest of the graph layer already
uses, so swapping to a live Neo4j is a query-for-query replacement with no
behaviour change.

**Known graph-level issue (Arjun flagged it):** `Concept.term` is unique
graph-wide, so a cross-domain term such as "sentence" or "spouse" is one node
and keeps the precision/support of whichever domain was loaded last. Nothing at
runtime reads those properties from the graph, so it is currently harmless —
`load_ontology.py` prints a note when it happens.

---

## The data layer, from zero

### The problem it solves

Without a database the app **forgets everything when the backend restarts.**
Upload 50 documents, correct them, stop the server — gone. That is the
in-memory path, and it is the reason `seed_retrain_data.py` exists.

A database is just somewhere to put things so they are still there tomorrow.

### Why there are two, and they are not redundant

Two *shapes* of information.

**Shape 1 — records.** Flat facts, one per row. This is **Postgres**.

| table | holds |
|---|---|
| `batches` | "someone uploaded 53 documents at 2:14pm" |
| `batch_items` | "document 7 of that batch" |
| `predictions` | "the model said Criminal Law, 0.92 confident" |
| `annotations` | "a human changed it to Family Law" |
| `model_versions` | "v-20260929, trained on 65 rows, 81% accurate" |
| `retrain_jobs` | "a retrain was requested and finished" |
| `xai_jobs` | "an explanation was computed for prediction 412" |
| `dataset_versions` | "the corpus as it stood when v-20260929 was trained" |

**Shape 2 — connections.** Things pointing at things. This is **Neo4j**.

```
"dowry"               --indicates-->  Family Law
"cervical discectomy" --narrows--->   "discectomy"
"ankle tourniquet"    --occurs with-- "sagittal saw"
Criminal Law          --related to--  Family Law
```

You *could* force the graph into Postgres. But the questions you ask it —
*"starting at this term, what does it point to, and what do those point to?"* —
become horrible nested JOINs in SQL and one short line in a graph database.

> **Postgres answers "what happened." Neo4j answers "what connects to what."**

The project is called *Knowledge **Graph**-Driven*, so the second is the title,
not decoration.

### What "cloud" means here

Nothing is installed locally. Someone else runs the databases:

| | what it is | what you get |
|---|---|---|
| **Supabase** | hosted Postgres plus an HTTP layer | a URL and a secret key |
| **Neo4j Aura** | hosted Neo4j | a URL, a username, a password |

That is all: **a database on someone else's computer, reachable over the
internet, with a password.**

### `.env` is a light switch

Six lines — the two URLs and the keys. The app treats it as a switch:

```
.env present   →  real Supabase + real Neo4j
.env absent    →  in-memory, forgets everything, demo-safe
```

Nothing else changes. Same code, same UI. The entire mechanism is:

```python
def is_configured():
    return "placeholder" not in settings.SUPABASE_URL
```

*"Does the URL look real? Then use it."*

### The two one-time setup steps

**Migrations** (`db/`, `001`–`008`) — SQL files that say "create a table called
`predictions` with these columns." Run once against an empty database. Arjun
has already done this; migration `008_app_schema_gaps.sql` was the last.

**Graph seeds** — `.cypher` scripts of "create this node, connect it to that
one." `graph/load_ontology.py` sends them to Neo4j.

### What breaks if each dies

| down | what happens |
|---|---|
| **Neo4j** | falls back to the in-memory ontology, **keeps working** |
| **Supabase** | uploads and annotations **error out** — no fallback |

Supabase is the fragile one. Which is why the switch matters: if the network
dies mid-demo,

```bash
mv backend/.env backend/.env.off     # in-memory, cannot fail
mv backend/.env.off backend/.env     # real databases again
```

**Practise that move before the review.** It is the parachute.

### Credential handling

**Never paste `.env` contents, keys or passwords into a chat or a commit.** Do
not ask tooling to open, `cat`, print or edit `.env`, and deny any request that
would. `.claude/settings.local.json` blocks it:

```json
{"permissions": {"deny": ["Read(./prototype-v1/backend/.env)", "Read(**/.env)"]}}
```

Shell commands that would print the file or the environment (`cat .env`, `env`,
`printenv`, `echo $SUPABASE_SERVICE_KEY`) are denied too. Running the app, the
tests and `check_data_layer.py` is fine — they read the values without
displaying them.

### Setup gotchas, learned the hard way

- Use the **legacy `service_role` key** (starts `eyJ`). `supabase==2.4.6`
  rejects the newer `sb_secret_…` format. `pip install supabase==2.4.6` may be
  needed.
- `./venv/bin/python check_data_layer.py` must end with
  **"All checks passed. Both databases are live."**
- `ServiceUnavailable` on port 7687 means **college wifi is blocking Neo4j** —
  use a phone hotspot.
- `model_versions`, `retrain_jobs` and `batches` have **no `domain` column**.
  Open decision.

---

## Demo runbook — verified screen by screen

**Requires Node 20.19+** (system Node 18 will crash the frontend) and the
weights at `backend/models/LegalModelShared/` (418 MB, distributed separately,
gitignored).

```bash
# terminal 1 — backend, workers and frontend
./run_local_demo.sh
# wait for "Backend ready." then open http://localhost:5180
```

Backend on `:8000`, frontend on `:5180`. `Ctrl+C` in terminal 1 stops
everything.

### Setup, once, well before the slot

```bash
python3 seed_retrain_data.py
```

55 real API calls against the running backend, each submitting a validated
annotation that **agrees with the model's own label** — the same
`POST /annotate` the Review screen makes, done in bulk. Nothing is faked. Takes
a minute or two.

Two things to know:

- `TEXTS` has **55 entries** and the retrain gate is **50**, so the seed clears
  the gate on its own. The corrections you make during the demo are the
  *signal*, not the quota.
- Those 55 are **confirmations**, spread across all 10 labels. They are the
  rehearsal set that keeps the class prior in place.
- The script is **not idempotent**. On the in-memory path a restart gives a
  clean slate, which for a repeatable demo is a feature. On a persisted
  Supabase it would double-seed.

### Step 1 — upload `sample-data/demo/1-intake.csv`

53 rows. Measured: **27 reach the review queue, 25 are predicted wrong.**

### Step 2 — Review. Demo the ontology check here.

Do it now, while the probe set is **not yet in the database**, so nothing is at
risk.

`teaching-set-only.csv` is 25 rows, **all gold Criminal Law**, and **v1
predicts Family Law on 25 of 25**. 18 of those 25 sit inside `1-intake.csv`,
and of those 18:

```
flagged by ontology       18 / 18
flagged by low confidence  0 / 18
```

Every single one caught by evidence; none by confidence. **That is the
demonstration.** Correct them — each card arrives with the suggested label
pre-selected, so it is one click.

The name "teaching set" is not an ML term. It is just this team's name for
*"the rows you correct during the demo."* The ML name would be the correction
set. `teaching-set-only.csv` exists so you can upload those 25 without the
surrounding 53-row batch if you want a shorter run.

### Step 3 — upload `sample-data/demo/2-probe-set.csv`

8 documents, all gold Criminal Law, **never corrected, in no training file.**
This is the "before".

```
#  gold          predicted      conf   routing       ontology  in queue?
1  Criminal Law  Family Law    0.996   AUTO_ACCEPT   -         no
2  Criminal Law  Family Law    0.985   AUTO_ACCEPT   FLAG      yes
3  Criminal Law  Family Law    0.996   AUTO_ACCEPT   -         no
4  Criminal Law  Family Law    0.996   AUTO_ACCEPT   -         no
5  Criminal Law  Family Law    0.995   AUTO_ACCEPT   FLAG      yes
6  Criminal Law  Family Law    0.994   AUTO_ACCEPT   -         no
7  Criminal Law  Family Law    0.995   AUTO_ACCEPT   -         no
8  Criminal Law  Family Law    0.996   AUTO_ACCEPT   FLAG      yes
```

**Where to look.** The Annotator page fetches `getBatchItems()` straight after
upload and renders `BatchItemsTable`, which shows `predicted_label` and a
`ConfidenceBadge`. The badge is green at `>= 0.7`, so **all 8 rows read
"Family Law" + "Model certainty: HIGH".** That table carries **no ontology
information and no gold label**, so on its own it is an assertion, not proof.

So also open **Review**, where 3 of the 8 appear with the full evidence card —
green HIGH badge and `Family Law 0.0` in the same card. That is proof on screen.

> ### The one rule: never press **Submit** on a probe row.
> Looking at them is free. Submitting one writes a validated annotation, which
> enters the retrain and destroys the before/after. Three probes are sitting in
> that queue looking exactly like the items you were just correcting.

**On the 5 that nothing flagged** — this is the strongest point in the demo, so
do not apologise for it:

> The detector does not need to catch every instance. It needs to catch enough
> instances of an error **class** for the loop to learn the fix. It caught 18
> of 18 here. The five probes nothing ever flagged got fixed anyway — **which
> is the point.** You do not have to detect every error; you have to detect the
> error.

Honest refinement if pushed: an error class the detector misses *entirely*
would never surface, so recall still matters — per class, not per document.

### Steps 4–6

**Admin → Retrain → Activate**, then **upload `2-probe-set.csv` again**, then
**Admin → Compare → pick the two probe batches.**

Seven or eight of the eight flip from Family Law to Criminal Law, and none of
them were ever corrected.

**Proof, strongest first:** the Compare tab (same 8 documents, v1 vs v2, side
by side) · the Review evidence cards · the `gold_label` column in the CSV.

### Offline backup, immune to anything going wrong live

`backend/compare_checkpoints.py` replays the five recorded checkpoints, so the
result does not depend on the live run:

| checkpoint | validated rows it trained on | probes correct |
|---|---|---|
| `LegalModelShared` (deployed) | — | **0 / 8** |
| v-20260904-091128 | 47 | 0 / 8 |
| v-20260910-102339 | 48 | 0 / 8 |
| v-20260928-154348 | 60 | 7 / 8 |
| v-20260928-185817 | 65 | 7 / 8 |
| v-20260929-090239 | 65 | **8 / 8** |

All eight of the deployed model's errors sit at **0.985–0.996** confidence,
above the 0.85 auto-accept threshold — in production they would be filed as
Family Law with no human ever seeing them. The flip happens somewhere between
48 and 60 validated annotations.

**Contamination checked before any of this was quoted:** the eight probe texts
appear in none of `datasets/v1.csv`, `1-intake.csv`, `teaching-set-only.csv`
or `seed_retrain_data.py`.

### The demo batch is one confusion, on purpose

Say it before anyone else does:

> *"The batch is deliberately one confusion. In twenty minutes you need an
> error the audience can see and a fix they can verify. The general claim is
> not carried by this batch — it is carried by the held-out measurement."*

And the confusion is **real, not a toy**: 498A, dowry demands, cruelty,
abetment of suicide genuinely sit on the Criminal/Family boundary in Indian
law, and misfiling a 498A matter as a family dispute has actual consequences.

### Demo-safety notes

- Run the demo on the **in-memory path** and show the real database separately.
  `.env` present/absent is the switch.
- Neo4j falls back to the mock if it is down. **Supabase does not.**
- Medical is a **table, not a live demo.** Switching domains means restarting
  the backend with different weights, and there are 20 minutes. All 29 metrics
  files are precomputed in `backend/metrics/`.

---

## Headline numbers

`docs/RESULTS.md` is the authority. These are the ones worth memorising.

### Accuracy

- **Legal: 94.4% document-level, 17 of 18 held-out documents**, 84.5% mean
  per-document passage agreement. The single miss is the Indian Stamp Act,
  labelled Taxation and predicted Property, where the prediction is arguably
  better than the label; left as a miss on purpose.
- **Medical: 90.5% on 273 held-out documents.**
- For 10 classes, random guessing is 10%.

### Calibration

Legal, on all 1,701 corpus samples: **70.7% accuracy at 85.4% mean
confidence — overconfident by 14.6 points.** That is not an unfixed bug; it is
the entire justification for the human review step, and the reason the
ontology check cannot rely on confidence.

By source: hand-written 100% · AI-generated 93% · extracted PDFs 60% · real
court summaries 52%. **100% on hand-written samples is memorisation, not
skill** — the deployed checkpoint was almost certainly trained on part of that
data, so 70.7% is an upper bound, not a held-out result.

### The ontology check

~⅓ of real errors caught, **88% precision**, 8.0% false positives, ~8% extra
reviewer volume. Operating point chosen by grid search, not taste. Medical:
0/5 at every one of 30 combinations swept, ships at 1.6% FP as an honest no-op.

### Retraining

- **Medical, five seeds (42/1337/7/2024/99):** +0.7, +3.3, +2.6, +5.5, +4.8pp
  → **mean +3.37pp, sd 1.87pp, positive 5/5. ECE improved 5/5.** Confident
  errors flat, 9.0 → 9.2. Quote it as **+3.4pp across five seeds**. Seed 42,
  the one originally reported, was the *worst* of the five.
- **Legal, on the real held-out set** (1,477 passages / 18 documents,
  LegalModelShared vs v-20260929-090239): **document-level 17/18 → 17/18
  unchanged; pooled passage 55.2% → 50.9% (−4.3pp).** 31 passages fixed (12
  of them confidently wrong before), 95 broken (2 confidently wrong). Criminal
  Law 4 fixed / 0 broken. Taxation 13 fixed / 73 broken — 821 of the 854
  Taxation passages are the Indian Stamp Act.

  **The loop fixed what it was aimed at and did not improve the model
  overall.** Say the first. Do not volunteer the second, and do not deny it.
- **Control (corrections vs random):** +0.13pp. Nothing. See "Why it must train
  on the confirmations too" above for how to state this.

### Probes

`LegalModelShared` 0/8 at 0.985–0.996 confidence → v-20260929-090239 **8/8**.

---

## Known limits — own them, do not volunteer them

### Out-of-domain text

**There is no abstain class.** Any document gets one of the 10 labels. Measured
on 8 off-domain samples through the deployed model plus the ontology check:

```
text           predicted label                conf  ont?  auto-accepted
recipe         Intellectual Property Law     0.985  -     YES -- no human
football       Property / Real Estate Law    0.865  -     YES -- no human
python_docs    Contract Law                  0.995  -     YES -- no human
weather        Property / Real Estate Law    0.507  -     no (reviewed)
lorem ipsum    Labour & Employment Law       0.993  -     YES -- no human
clinical_note  Family Law                    0.641  -     no (reviewed)
"ok"           Property / Real Estate Law    0.501  -     no (reviewed)
gibberish      Contract Law                  0.525  -     no (reviewed)
control_crim   Criminal Law                  0.996  FLAGS 9/10 other labels
```

**0 of 8 flagged. 4 of 8 silently auto-accepted.** Confidence range
0.501–0.995, mean 0.751.

The pattern: **gibberish gets ~0.50, fluent prose gets 0.99.** The model
responds to *fluency*, not to domain membership.

The ontology check is blind here **by construction** — `check()` opens with
`ev = self.evidence(text); if not ev: return None`. Zero markers matched means
zero flag, and junk text has no domain vocabulary to contradict.

**The obvious fix was built and measured and rejected** — reproduce it with
`backend/out_of_domain_probe.py`, which writes
`backend/metrics/legal_out_of_domain.json`. A total-marker-evidence floor,
routing anything below it to review:

```
 floor   legal wrongly flagged    OOD caught
   1.0               233 (13.7%)        7/8
   2.0               371 (21.8%)        8/8
   5.0               681 (40.0%)        8/8
```

Real legal text in `v1.csv` has median evidence 7.1 but a **5th percentile of
0.0** — 13.7% of genuine legal passages match no marker at all. Catching 7 of
8 junk samples costs 233 real documents in the queue, worse than the ontology
check's entire false-positive budget, for a problem no user has.

**The answer if asked:** *"Measured it. Out-of-domain text gets no ontology
flag — the detector finds vocabulary that contradicts the label, and
off-domain text has none to contradict. Lorem ipsum comes back Labour &
Employment at 0.993. We designed the obvious fix, an evidence floor, and
measured it: catching it costs 13.7% of real legal documents going to review.
So it needs an abstain class trained on negatives, not a threshold. Not in
this build."*

**Do not claim out-of-domain robustness.** Staying quiet is fine; asserting it
is not.

### Other gaps

- **The legal retrain does not lift the model globally** — see above.
- **Marker matching is shallow** — vocabulary, not argument structure.
- **`evaluation/stress_test.csv` is not a test set.** All 225 rows also appear
  in `v1.csv`, verified by exact match after normalisation. It is a slice of
  the training corpus, kept for load-testing the UI only.
- **The ontology check's recall is ~⅓**, so two thirds of confident errors
  still ship as gold. The claim is **provenance and measured agreement**, not
  correctness.

---

## Terminology, defined so it can be said out loud

**Decision boundary** — the model turns a document into a point in
high-dimensional space and carves that space into one region per label. The
boundary is the surface where it flips from one label to another. *"The line
the model draws between two labels; documents near it are ambiguous, and moving
it changes which side they land on."* Dowry-harassment documents sit on the
Criminal/Family boundary because they genuinely contain both vocabularies.

**Prior (class prior)** — *"the model's baseline expectation of how common each
label is, independent of what is in the text."* A flu-clinic doctor's prior is
flu; it shifts what they guess when symptoms are unclear.

**Class-prior skew** — training data over-represents a class, so the model
over-predicts it. Seed 42's corrections-only arm is the measured example.

**Memorisation vs generalisation** — being right on documents you trained on,
versus being right on documents you have never seen. *"If it had memorised, the
probes would not move — they are not in the training data."*

**Catastrophic forgetting** — fine-tuning on new examples costs the model
things it already knew. **Rehearsal** is the standard mitigation: keeping old
correct examples in the training mix. That is what the 55 confirmations are.

**Calibration / ECE** — a model is calibrated when its confidence matches its
accuracy; **Expected Calibration Error** is the average gap. Legal is
overconfident by 14.6 points. Medical retraining improved ECE in 5 of 5 seeds:
*"not just more accurate, more honest about when it is unsure."*

**Softmax / confidence** — the model's raw outputs are turned into
probabilities that sum to 1; confidence is the largest of them. Because they
sum to 1, high confidence mathematically forces a large margin — which is why
the margin guard could never fire.

**Jaccard index** — overlap between two sets, used for concept co-occurrence:
documents containing both terms, divided by documents containing either.

**Precision / recall, for the ontology check** — precision is "88% of what it
flags is a real mistake"; recall is "it catches about a third of real
mistakes." The two trade off against each other.

---

## Repo map

| path | what |
|---|---|
| `backend/` | FastAPI API, workers, ML services |
| `backend/services/model_service.py` | model load + inference |
| `backend/services/active_learning.py` | confidence routing |
| `backend/services/ontology_check.py` | the ontology consistency check |
| `backend/services/neo4j_service.py` | graph queries (OntologyNode only) |
| `backend/services/supabase_service.py`, `_supabase_real.py` | Postgres, real and mock |
| `backend/services/mock_db.py` | the in-memory store |
| `backend/workers/retrain_worker.py` | the retrain loop |
| `backend/routers/` | `batches`, `predict`, `queue`, `annotate`, `explain`, `stats`, admin |
| `backend/metrics/` | **29 precomputed experiment result files** |
| `backend/models/` | `LegalModelShared`, `MedicalModelShared`, five recorded checkpoints |
| `backend/compare_checkpoints.py` | offline probe replay; `--pdf-manifest`/`--pdf-dir`/`--exclude` |
| `backend/retrain_experiment.py` | the seed/arm experiment driver; `--seed`, `--retrain-epochs`, `--random-sample`, `--tag` |
| `backend/evaluate_baseline.py` | corpus-wide baseline measurement |
| `backend/check_data_layer.py` | verifies both live databases |
| `frontend/` | React app (Node 20.19+) |
| `domains/legal.json`, `medical.json` | the label sets and per-domain thresholds |
| `db/` | Postgres schema, migrations `001`–`008`, RPC functions |
| `graph/` | seeds, `mine_concept_markers.py`, `mine_graph_depth.py`, `load_ontology.py`, `verify_ontology.py` |
| `datasets/v1.csv` | 1,701 labelled legal training samples |
| `datasets/medical_v1.csv` | medical training corpus |
| `sample-data/demo/` | what you upload during the demo |
| `sample-data/evaluation/` | measurement sets, including `heldout/` and `medical_heldout.csv` |
| `sample-data/archive/` | older demo batches |

### Sample data

| file | rows | use for |
|---|---|---|
| `demo/1-intake.csv` | 53 | **the demo.** 27 reach the queue, 25 predicted wrong |
| `demo/2-probe-set.csv` | 8 | before/after. **Never correct these** |
| `demo/teaching-set-only.csv` | 25 | the correction set alone; all gold Criminal Law, v1 gets 0/25 |
| `evaluation/heldout/` + `heldout_manifest.csv` | 21 docs | **the real legal test set.** In no CSV anywhere. 3 are scans the PDF guard refuses, so 18 reach the model |
| `evaluation/medical_heldout.csv` | 273 | medical held-out, rebuilt 19 Oct 2026 so no document straddles the split |
| `evaluation/stress_test.csv` | 225 | **not a test set** — a slice of `v1.csv`. UI load-testing only |
| `archive/demo_batch_tough.csv` | 45 | older batch, 5 items reach review |
| `archive/demo_batch_easy.csv` | 50 | safe fallback, but pages 2–4 are entirely green |

---

## Operations

```bash
# virtualenv
prototype-v1/backend/venv

# tests — 295 pass (221 backend + 74 data layer)
cd backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/python -m pytest tests/ -q

# run everything
./run_local_demo.sh                      # :8000 backend, :5180 frontend

# seed the retrain gate, once per backend start on the in-memory path
python3 seed_retrain_data.py

# graph. Two things this needs that are easy to miss: the neo4j driver, which
# is in the backend venv and not in system python3 (and "python" is not a
# command on Ubuntu at all), and the Neo4j credentials, which the graph
# scripts read from the environment rather than from backend/.env the way the
# API does. "set -a" exports the file without printing it.
set -a; . backend/.env; set +a
./backend/venv/bin/python graph/load_ontology.py --domain legal
./backend/venv/bin/python graph/load_ontology.py --domain medical
./backend/venv/bin/python graph/verify_ontology.py --domain legal
./backend/venv/bin/python graph/mine_graph_depth.py --domain legal

# verify the live databases
cd backend && ./venv/bin/python check_data_layer.py
# must end: "All checks passed. Both databases are live."
```

**Key settings** (`backend/config/settings.py`, overridable in `.env`):

```
ACTIVE_DOMAIN            = legal
MODEL_PATH               = ./models/LegalModelShared
CONFIDENCE_HIGH          = 0.85     # above this: AUTO_ACCEPT
CONFIDENCE_LOW           = 0.55     # below this: ROUTE_TO_REVIEWER
REVIEW_THRESHOLD         = 0.5      # the queue's low-confidence cut
CALIBRATION_TEMPERATURE  = 1.0      # off; see RESULTS.md section 3
```

**The models are not in git.** `LegalModelShared` is a real 418 MB directory
(it used to be a symlink), SHA-256 verified, gitignored.

**Note for the data-layer owner:** the backend tests stub torch, so running
them needs no GPU, no model and no weights.

---

## History — what was wrong and got fixed

Worth knowing, because each one is a measurement lesson rather than a bug fix.

**1. The medical test set was contaminated.** The old held-out set shared
129 of 435 rows with training — **all of them with conflicting labels**,
because mtsamples files the same transcription under two specialties. The model
was being marked wrong for answering what it had been taught. Rebuilt by
`build_medical_corpus.py` into **1,472 train / 273 held-out with zero overlap
asserted**, and 379 conflicts documented in `medical_heldout_SKIPPED.md`.
Medical accuracy went **68.3% → 90.5%**. **The model never changed. The exam
was unfair.** Markers were re-mined from the clean corpus: 1,179 markers, 1.1%
false positives, legal regression-checked at 136/1,701 = 8.0%.

**2. `GET /batches` had returned 500 since the day it was written** — the
"Could not load recent batches" error. Two classes named `BatchSummary` in
`response_models.py`; Python kept the second silently. Renamed to
`ComparisonBatchSummary`, 5 regression tests added.

**3. The margin guard could never fire.** Arithmetic, not a tuning problem.
Measured 0 firings in 214 predictions, minimum margin 0.7428. Removed.

**4. A number in the docs had no source.** "70.7% → 75.3%" appeared nowhere in
the repo. Section 6 of RESULTS.md was retracted, then the probe claim was
re-measured so it has a source. The README's "15 corrections is the measured
minimum" was removed for the same reason. **Both retractions stay retracted.**

**5. The legal probes were contamination-checked** against `v1.csv`,
`1-intake.csv`, `teaching-set-only.csv` and `seed_retrain_data.py` before being
quoted. Zero overlap.

---

## Open threads

- **Reload the graph** to pick up the depth edges. The loaded counts (legal
  10/21/752, medical 12/19/1179) predate the depth seed; a reload adds 90 legal
  and 1,387 medical edges. Expect legal 32 + 1,505 + 90, medical 32 + 2,359 +
  1,387.
- **Demo rehearsal** end to end, and a visual check that the frontend handles
  12 labels as well as 10. The API side is already verified: `/domains` serves
  10 and 12, ids contiguous, and no frontend code assumes 10.
- **No `domain` column** on `model_versions`, `retrain_jobs`, `batches`.
- **`Concept.term` is unique graph-wide**, so cross-domain terms collide.
- **Out-of-domain rejection** — needs an abstain class trained on negatives.
  Designed, measured, deliberately not shipped.
- **Concepts with edges to several labels at different weights** — the honest
  next step for the graph, and the thing that would make the graph
  representation earn its place over the JSON mirror. Measured worse with the
  current marker set.
