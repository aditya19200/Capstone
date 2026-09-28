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

Output: graph/concept_markers.json, consumed by
services/ontology_check.py. Re-runnable; nothing is hand-edited in place.

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
DATASET = ROOT / "datasets" / "v1.csv"
OUTPUT = ROOT / "graph" / "concept_markers.json"

MIN_DOCS_DEFAULT = 6
MIN_PRECISION_DEFAULT = 0.60
MIN_SOURCES_DEFAULT = 2
TOP_N_PER_LABEL = 40

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

    for label in markers:
        markers[label].sort(key=lambda m: m["score"], reverse=True)
        markers[label] = markers[label][:TOP_N_PER_LABEL]

    return markers, label_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-docs", type=int, default=MIN_DOCS_DEFAULT)
    ap.add_argument("--min-precision", type=float, default=MIN_PRECISION_DEFAULT)
    ap.add_argument("--min-sources", type=int, default=MIN_SOURCES_DEFAULT,
                    help="distinct corpora a term must appear in (guards against "
                         "learning dataset provenance instead of legal concepts)")
    ap.add_argument("--print-only", action="store_true",
                    help="show what would be written without writing it")
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()
