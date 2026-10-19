#!/usr/bin/env python3
"""
graph/mine_concept_markers.py — derive domain concept markers from the corpus.

The ontology consistency check needs to know which terms are evidence for
which legal domain. Hand-writing that list would encode whoever wrote it;
this derives it from datasets/v1.csv, where 1,701 samples already carry
human gold labels.

Crucially it learns from the GOLD LABELS, never from model predictions. That
independence is the whole point: a detector built from the model's own
output would inherit the model's blind spots, which is exactly what we are
trying to catch.

Method, deliberately simple so it can be explained and audited:

  1. Tokenise into unigrams and bigrams (statute references like "section
     498a" only mean anything as a pair).
  2. Keep terms appearing in at least MIN_DOCS distinct documents, so a
     single quirky sample cannot mint a marker.
  3. Keep terms where P(label | term) >= MIN_PRECISION for some label —
     i.e. when this term shows up, that domain is what it usually is.
  4. Require the term to appear in at least MIN_SOURCES distinct corpora.
     Without this the miner learns dataset provenance instead of law: terms
     like "www cci", "cpio sebi" and OCR breakage like "con dentiality"
     (from "confidentiality" in extracted PDFs) score perfectly because they
     occur in exactly one source, whose samples share a label. A concept
     that is real legal vocabulary shows up in more than one corpus.
  5. Score each surviving term by precision x log(support), so a term that
     is both reliable and common outranks one that is merely reliable.

Outputs both representations of the same model:

  graph/concept_seed.cypher   — the concepts AS GRAPH:
                                (:Concept)-[:INDICATES {weight}]->(:OntologyNode)
                                Loads with graph/load_ontology.py alongside
                                ontology_seed.cypher.
  graph/concept_markers.json  — the flat export the runtime reads while no
                                Neo4j instance is stood up.

The two are generated from the same mining run and cannot drift, which is
the same arrangement ontology_seed.cypher already documents for the domain
nodes: model it in the graph, mirror it for the mock, swap without any
behaviour change. Note the current limit: each term is assigned to its
single most frequent domain, so every Concept has exactly one INDICATES
edge and the layer is still flat in substance. Emitting several weighted
edges per concept is the change that would make the graph representation
do work a table cannot.

Re-runnable; nothing is hand-edited in place.

    python3 graph/mine_concept_markers.py [--min-docs 8] [--min-precision 0.65]
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "domains"

# Which corpus to mine and where to write the markers both come from the
# domain pack, so mining a second domain is --domain <key> rather than an edit
# here. Markers are not transferable between domains: they are the vocabulary
# of one corpus, scored against that corpus's own gold labels.
#
# Set by resolve_domain() in main(); module-level so the helpers below read
# them the way they always did.
DATASET = None
OUTPUT = None
CYPHER_OUTPUT = None
DOMAIN_KEY = None
DOMAIN_LABELS = []        # label names in pack (id) order — the output order
PACK_LABELS = []          # the full label objects, for the ontology layer
ONTOLOGY_OUTPUT = None


def resolve_domain(key: str):
    """Point DATASET/OUTPUT/CYPHER_OUTPUT at one domain pack's files."""
    global DATASET, OUTPUT, CYPHER_OUTPUT, DOMAIN_KEY, DOMAIN_LABELS
    global PACK_LABELS, ONTOLOGY_OUTPUT
    path = PACK_DIR / f"{key}.json"
    if not path.exists():
        available = sorted(p.stem for p in PACK_DIR.glob("*.json"))
        sys.exit(f"No domain pack at {path}. Available: {', '.join(available) or 'none'}")

    pack = json.loads(path.read_text(encoding="utf-8"))
    dataset = pack.get("dataset")
    markers = pack.get("markers_file")
    if not dataset or not markers:
        sys.exit(
            f"domains/{key}.json must set both 'dataset' and 'markers_file' "
            f"before markers can be mined for it."
        )

    DOMAIN_KEY = key
    DOMAIN_LABELS = [l["name"] for l in pack["labels"]]
    PACK_LABELS = pack["labels"]
    DATASET = ROOT / dataset
    OUTPUT = ROOT / markers
    # The Cypher mirror sits beside the JSON, named after it, so the two are
    # obviously one output of one run.
    CYPHER_OUTPUT = OUTPUT.with_name(OUTPUT.stem.replace("_markers", "_seed") + ".cypher")
    ONTOLOGY_OUTPUT = ROOT / "graph" / f"{key}_ontology_seed.cypher"

# These defaults are the operating point the committed concept_markers.json
# was actually mined at, and that services/ontology_check.py's MIN_EVIDENCE
# was grid-searched against. They used to read 6 and 0.60, which meant a
# plain re-run quietly produced a smaller, different marker set than the one
# in the repo — so the file could not be regenerated from its own script.
MIN_DOCS_DEFAULT = 4
MIN_PRECISION_DEFAULT = 0.55
MIN_SOURCES_DEFAULT = 2
# How many markers to keep per domain. This was 40, which was arbitrary and
# turned out to be the binding constraint on the detector: Criminal Law alone
# had 188 terms passing every quality filter above, and 148 of them were being
# discarded by this cap rather than by any measure of quality.
#
# Raising it to 100 (752 markers, up from 393) is what let the detector catch
# confidently-wrong predictions in PRADEEP v. STATE OF HARYANA, a real murder
# appeal whose child-witness passages the model files under Family Law at
# 0.84-0.98 confidence. At 40 it caught 0 of 5; at 100, 3 of 5.
#
# Measured cost, against all 1,701 corpus samples: false positives 7.5% ->
# 8.0%, and on genuine Family Law documents 9.6% -> 11.8%. Recall on the
# criminal-as-Family confusion rose 74.5% -> 76.0%. Going further to 220
# bought another 1.0pp of recall for another 0.5pp of false positives and
# caught no additional Pradeep passages, so 100 is where it stops.
TOP_N_PER_LABEL = 100

# Words that carry no domain signal. Kept short on purpose — an aggressive
# stoplist would quietly drop real legal vocabulary.
STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "with", "from", "as", "that", "this", "these", "those", "it", "its", "is",
    "was", "were", "be", "been", "being", "has", "have", "had", "not", "no",
    "but", "if", "then", "than", "so", "such", "which", "who", "whom", "whose",
    "after", "before", "during", "under", "over", "against", "between", "into",
    "their", "his", "her", "he", "she", "they", "them", "we", "our", "us",
    "court", "case", "filed", "sought", "alleged", "claims", "claimed",
    "petitioner", "respondent", "plaintiff", "defendant", "appellant",
}

# Web and response-boilerplate fragments that survive the multi-source filter
# because several corpora are scraped from portals. They are provenance, not
# legal vocabulary, so they are dropped by name rather than by statistics.
JUNK = {
    "http", "https", "www", "com", "gov", "gov in", "https www", "at https",
    "portal", "query", "queries", "website", "online", "email", "pdf",
    "following information", "the response", "also provided", "guided",
    "con", "ling", "cation", "dentiality", "ces", "rst", "cer",
}
_JUNKY = re.compile(r"https?|www|\.in\b|@")

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    """Lowercase unigrams plus adjacent bigrams, stopwords dropped."""
    words = _TOKEN.findall(text.lower())
    unigrams = [w for w in words if w not in STOP and len(w) > 2]
    # Bigrams are built from the *unfiltered* stream so "section 498a" and
    # "breach of contract" survive — dropping stopwords first would break them.
    bigrams = [
        f"{a} {b}" for a, b in zip(words, words[1:])
        if not (a in STOP and b in STOP) and len(a) > 1 and len(b) > 1
    ]
    return set(unigrams) | set(bigrams)


def load_rows():
    if not DATASET.exists():
        sys.exit(f"Dataset not found: {DATASET}")
    with DATASET.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in ("text", "label_name") if c not in (rows[0] if rows else {})]
    if missing:
        sys.exit(f"{DATASET.name} is missing column(s): {', '.join(missing)}")
    return rows


def mine(rows, min_docs: int, min_precision: float, min_sources: int):
    term_total = Counter()                       # docs containing term
    term_by_label = defaultdict(Counter)         # term -> label -> docs
    term_sources = defaultdict(set)              # term -> {source, ...}
    label_total = Counter()

    for row in rows:
        label = row["label_name"].strip()
        if not label:
            continue
        source = (row.get("source") or "unknown").strip()
        label_total[label] += 1
        for term in tokenize(row["text"]):
            term_total[term] += 1
            term_by_label[term][label] += 1
            term_sources[term].add(source)

    markers = defaultdict(list)
    for term, total in term_total.items():
        if total < min_docs:
            continue
        if term in JUNK or _JUNKY.search(term):
            continue
        if any(part in JUNK for part in term.split()):
            continue
        if len(term_sources[term]) < min_sources:
            continue                             # single-corpus artifact
        label, count = term_by_label[term].most_common(1)[0]
        precision = count / total
        if precision < min_precision:
            continue
        markers[label].append({
            "term": term,
            "precision": round(precision, 4),
            "support": count,
            "sources": len(term_sources[term]),
            "score": round(precision * math.log(count + 1), 4),
        })

    # Both orderings below exist to make the generated files reproducible.
    # tokenize() returns a set, so the order terms are discovered in varies
    # with PYTHONHASHSEED; that leaked into the output twice over — into
    # which of two equal-scoring markers came first (a stable sort inherits
    # the input order) and into the label key order (markers is populated
    # while iterating the term counts). Neither changed *which* markers were
    # mined, but both made the files churn on every run, so a real change was
    # invisible in the diff.
    ordered = {}
    for label in DOMAIN_LABELS:                  # pack (id) order, not discovery order
        if label not in markers:
            continue
        markers[label].sort(key=lambda m: (-m["score"], m["term"]))
        ordered[label] = markers[label][:TOP_N_PER_LABEL]

    # A gold label present in the corpus but absent from the pack: surfaced
    # rather than silently dropped, since it means the two disagree.
    for label in markers:
        if label not in ordered:
            print(f"WARNING: corpus label {label!r} is not in domains/{DOMAIN_KEY}.json")

    return ordered, label_total


def write_cypher(markers) -> None:
    """
    Emit the concept layer as Cypher.

    Node label   : Concept { term, precision, support }
    Relationship : (:Concept)-[:INDICATES { weight }]->(:OntologyNode { label })

    The domain node is MERGEd, not MATCHed. That distinction is the whole
    reason this file can be loaded on its own: a MATCH that finds nothing is
    not an error in Cypher, it simply does nothing, so a concept seed loaded
    against a database with no domain nodes would report success and create
    zero edges. This file previously did exactly that, which only became
    visible when a second domain was added and had no hand-written ontology
    seed to load first.
    """
    lines = [
        "// concept_seed.cypher",
        "//",
        "// GENERATED by graph/mine_concept_markers.py — do not hand-edit.",
        "// Re-run the miner to regenerate both this file and",
        "// concept_markers.json, which is its flat mirror for the mock path.",
        "//",
        "// The concept layer under the 10 domain nodes seeded by",
        "// ontology_seed.cypher. Each Concept is legal vocabulary mined from",
        "// the human gold labels in datasets/v1.csv; the INDICATES weight is",
        "// how strongly that term points at the domain. A term may indicate",
        "// several domains at different weights.",
        "//",
        "// Consumed by services/ontology_check.py to decide whether a",
        "// prediction's label is contradicted by the vocabulary in its text.",
        "",
        "CREATE CONSTRAINT concept_term IF NOT EXISTS",
        "FOR (c:Concept) REQUIRE c.term IS UNIQUE;",
        "",
    ]
    for label in markers:            # already in pack order
        lines.append("// ---- " + label + " ----")
        for m in markers[label]:
            term = m["term"].replace(chr(34), "")
            lines.append(
                'MERGE (c:Concept {term: "' + term + '"})'
                ' SET c.precision = ' + str(m["precision"])
                + ', c.support = ' + str(m["support"]) + ';'
            )
            lines.append(
                'MATCH (c:Concept {term: "' + term + '"}) '
                'MERGE (d:OntologyNode {label: "' + label + '"}) '
                'MERGE (c)-[r:INDICATES]->(d) SET r.weight = '
                + str(m["score"]) + ';'
            )
        lines.append("")
    CYPHER_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def write_ontology_cypher(pack_labels) -> None:
    """
    Emit the domain layer — the OntologyNodes the concepts attach to.

    graph/ontology_seed.cypher was written by hand for the ten legal domains
    and has no equivalent for any other domain, so a new domain pack had
    nowhere for its concepts to attach. This generates that file from the
    pack, which means adding a domain stays a matter of writing one JSON
    file.

    Descriptions and RELATED_TO edges come from the pack, so this and
    mock_db.ONTOLOGY — which projects the same pack — cannot disagree.
    """
    lines = [
        "// " + ONTOLOGY_OUTPUT.name,
        "//",
        "// GENERATED by graph/mine_concept_markers.py --domain " + str(DOMAIN_KEY),
        "// Source of truth: domains/" + str(DOMAIN_KEY) + ".json",
        "//",
        "// The domain nodes. Load this before the concept seed, or let the",
        "// concept seed MERGE them itself — either order works.",
        "//",
        "//   Node label   : OntologyNode { label, description }",
        "//   Relationship : (:OntologyNode)-[:RELATED_TO]->(:OntologyNode)",
        "",
        "CREATE CONSTRAINT ontology_node_label IF NOT EXISTS",
        "FOR (n:OntologyNode) REQUIRE n.label IS UNIQUE;",
        "",
    ]
    for label in pack_labels:
        desc = label.get("description", "").replace(chr(34), "")
        lines.append(
            'MERGE (n:OntologyNode {label: "' + label["name"] + '"})'
            ' SET n.description = "' + desc + '";'
        )
    lines.append("")
    for label in pack_labels:
        for other in label.get("related", []):
            lines.append(
                'MATCH (a:OntologyNode {label: "' + label["name"] + '"}), '
                '(b:OntologyNode {label: "' + other + '"}) '
                'MERGE (a)-[:RELATED_TO]->(b);'
            )
    lines.append("")
    ONTOLOGY_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-docs", type=int, default=MIN_DOCS_DEFAULT)
    ap.add_argument("--min-precision", type=float, default=MIN_PRECISION_DEFAULT)
    ap.add_argument("--min-sources", type=int, default=MIN_SOURCES_DEFAULT,
                    help="distinct corpora a term must appear in (guards against "
                         "learning dataset provenance instead of legal concepts)")
    ap.add_argument("--print-only", action="store_true",
                    help="show what would be written without writing it")
    ap.add_argument("--domain", default="legal",
                    help="domain pack supplying the corpus and output paths")
    args = ap.parse_args()

    resolve_domain(args.domain)
    rows = load_rows()
    markers, label_total = mine(rows, args.min_docs, args.min_precision, args.min_sources)

    print(f"Mined from {len(rows)} labelled samples in {DATASET.name}\n")
    for label in sorted(markers, key=lambda l: -len(markers[l])):
        terms = markers[label]
        print(f"{label}  ({label_total[label]} samples, {len(terms)} markers)")
        print("   " + ", ".join(m["term"] for m in terms[:14]))
        print()

    empty = [l for l in label_total if l not in markers]
    if empty:
        print(f"WARNING: no markers survived for: {', '.join(empty)}")
        print("Lower --min-precision or --min-docs if that domain matters.\n")

    if args.print_only:
        print("(--print-only: nothing written)")
        return

    OUTPUT.write_text(json.dumps({
        "source_dataset": str(DATASET.relative_to(ROOT)),
        "sample_count": len(rows),
        "min_docs": args.min_docs,
        "min_precision": args.min_precision,
        "min_sources": args.min_sources,
        "markers": markers,
    }, indent=1))
    print(f"Wrote {OUTPUT.relative_to(ROOT)} — "
          f"{sum(len(v) for v in markers.values())} markers across {len(markers)} domains.")

    write_cypher(markers)
    print(f"Wrote {CYPHER_OUTPUT.relative_to(ROOT)} — same model as graph nodes and edges.")

    write_ontology_cypher(PACK_LABELS)
    print(f"Wrote {ONTOLOGY_OUTPUT.relative_to(ROOT)} — the {len(PACK_LABELS)} domain nodes.")


if __name__ == "__main__":
    main()
