"""
mine_graph_depth.py — give the concept graph structure below the first level.

The graph as mined is a star. Every Concept points at exactly one
OntologyNode through :INDICATES, nothing points at anything else, and the
whole thing is one hop deep. That is a lookup table drawn as a graph: it
answers "which label does this term indicate" and no question that needs a
path. Neo4j is not earning its place in a system that only ever asks that.

This adds two kinds of edge between Concepts, both derived from evidence
already in the corpus rather than from a hand-written taxonomy.

  (:Concept)-[:NARROWS]->(:Concept)
      "cervical discectomy" is a kind of "discectomy". Taken from whole-word
      containment: a longer term whose words contain a shorter known term is
      a specialisation of it.

  (:Concept)-[:CO_OCCURS {jaccard, documents}]-(:Concept)
      "ankle tourniquet" and "sagittal saw" are different things that appear
      in the same operation notes. Taken from document co-occurrence over the
      domain's own corpus.

NOTHING IN THE SERVING PATH READS THESE EDGES, checked rather than assumed:

  * services/ontology_check.py -- the confidently-wrong detector every
    number in docs/RESULTS.md section 4 comes from -- loads the markers
    JSON and never opens a Neo4j session at all.
  * services/neo4j_service.py, which the /ontology routes do use, queries
    only (:OntologyNode) and the relationship types HAS_CHILD, RELATED_TO
    and HAS_PARENT. This script creates neither: its edges run between
    (:Concept) nodes and are called NARROWS and CO_OCCURS.

So no measured result can move because of anything in this file, and no
existing API response changes shape. That is deliberate. The graph gets
real depth and the claim stays "our graph has structure", never "our graph
improved accuracy". If a later feature does read these edges, it has to be
measured on its own before anything is claimed for it.

Two filters do the real work, and both exist because the obvious version of
this script produces impressive-looking nonsense.

GRAMMAR. Containment alone gives "accused is" -> "accused" and "438 of" ->
"438". The added words are function words, so the pair is one concept with
grammar stuck to it, not a specialisation. Pairs whose extra words are all
stopwords are dropped: 230 raw legal pairs become 84.

ARITHMETIC CO-OCCURRENCE. The strongest co-occurrences are always pairs
where one term contains the other -- "intellectual" and "intellectual
property" co-occur at Jaccard 1.00, necessarily, because every document
holding the long one holds the short one. That is the containment relation
restated as if it were evidence. Pairs related by containment are excluded
from co-occurrence; they are already joined by :NARROWS.

What is left differs sharply by domain, which is worth knowing before anyone
promises this feature generalises:

  medical   2,326 non-trivial pairs, 384 above Jaccard 0.4, and the strong
            ones are real: 'carpal ligament' with 'transverse carpal'
            (carpal tunnel release), 'capsular bag' with 'capsulorrhexis'
            (cataract surgery), 'ankle tourniquet' with 'sagittal saw' --
            two unrelated instruments used in one operation. Procedure notes
            describe a sequence of steps, so the vocabulary of a procedure
            genuinely clusters.

  legal     246 non-trivial pairs, 8 above Jaccard 0.4, and the strongest are
            provenance, not law: 'available at' with 'cci' at 0.60, 'cci'
            with 'please' at 0.50. Those are the footer of a CCI press
            release. Judgments do not share a template the way operation
            notes do, so what co-occurs is boilerplate.

So the default threshold is per domain, for the same reason
ontology_thresholds is: 0.25 for medical, where the signal is real, and 0.45
for legal, where only a handful survive and the rest are the dataset's own
letterhead. Legal keeps the setting rather than the feature being switched
off, because the few that survive are genuine and because a domain pack
should carry its own numbers.
"""

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))

from utils.text import normalize

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent

# Words that do not make a term more specific. A pair whose only extra words
# are these is grammar, not hierarchy.
STOPWORDS = {
    "is", "was", "were", "are", "be", "been", "of", "and", "the", "a", "an",
    "in", "to", "for", "on", "by", "with", "that", "this", "as", "or", "at",
    "from", "it", "its", "has", "have", "had", "not", "no", "any", "all",
    "such", "shall", "may", "which", "who", "he", "she", "his", "her",
    "there", "then", "upon", "where",
}

MIN_CO_DOCS = 10          # below this a Jaccard score is three documents' luck
DEFAULT_JACCARD = {"medical": 0.25, "legal": 0.45}
FALLBACK_JACCARD = 0.35   # an unknown domain gets the cautious middle


def load_terms(markers_path):
    data = json.loads(markers_path.read_text(encoding="utf-8"))["markers"]
    label_of = collections.defaultdict(set)
    for label, entries in data.items():
        for m in entries:
            label_of[m["term"]].add(label)
    return dict(label_of)


def hierarchy(terms):
    """(specific, general) pairs by whole-word containment, grammar removed."""
    known = set(terms)
    out = set()
    for term in terms:
        words = term.split()
        if len(words) < 2:
            continue
        for size in range(1, len(words)):
            for i in range(len(words) - size + 1):
                sub = " ".join(words[i:i + size])
                if sub == term or sub not in known:
                    continue
                if all(w in STOPWORDS for w in words[:i] + words[i + size:]):
                    continue
                out.add((term, sub))
    return sorted(out)


def contains(a, b):
    """True if either term's words contain the other's as a contiguous run."""
    return f" {b} " in f" {a} " or f" {a} " in f" {b} "


def co_occurrence(terms, texts, min_docs, min_jaccard):
    patterns = {t: re.compile(r"\b" + re.escape(t) + r"\b") for t in terms}
    doc_freq = collections.Counter()
    pair_freq = collections.Counter()
    for text in texts:
        present = sorted(t for t, p in patterns.items() if p.search(text))
        doc_freq.update(present)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_freq[(present[i], present[j])] += 1

    out = []
    for (a, b), n in pair_freq.items():
        if n < min_docs or contains(a, b):
            continue
        union = doc_freq[a] + doc_freq[b] - n
        j = n / union if union else 0.0
        if j >= min_jaccard:
            out.append((a, b, round(j, 4), n))
    # deterministic: strongest first, then alphabetical, so the file is stable
    return sorted(out, key=lambda r: (-r[2], r[0], r[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--min-docs", type=int, default=MIN_CO_DOCS)
    ap.add_argument("--min-jaccard", type=float, default=None,
                    help="default is per domain: 0.25 medical, 0.45 legal")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "backend"))
    from config.domains import get_domain

    domain = get_domain(args.domain)
    if domain.markers_path is None or not domain.markers_path.exists():
        sys.exit(f"{args.domain}: no markers file — run mine_concept_markers.py first")

    min_j = args.min_jaccard
    if min_j is None:
        min_j = DEFAULT_JACCARD.get(domain.key, FALLBACK_JACCARD)

    label_of = load_terms(domain.markers_path)
    terms = sorted(label_of)

    csv.field_size_limit(10 ** 7)
    with open(ROOT / domain.dataset, encoding="utf-8") as fh:
        texts = [normalize(r["text"]) for r in csv.DictReader(fh)]

    print(f"domain      : {domain.key}")
    print(f"concepts    : {len(terms)}")
    print(f"corpus      : {len(texts)} documents from {domain.dataset}")
    print(f"min jaccard : {min_j}   min docs: {args.min_docs}\n")

    narrows = hierarchy(terms)
    co = co_occurrence(terms, texts, args.min_docs, min_j)

    print(f"NARROWS   {len(narrows)} edges")
    for a, b in narrows[:6]:
        print(f"    {a!r} -> {b!r}")
    print(f"\nCO_OCCURS {len(co)} edges")
    for a, b, j, n in co[:6]:
        shared = label_of[a] & label_of[b]
        tag = f"both -> {sorted(shared)[0]}" if shared else "across labels"
        print(f"    {j:.2f}  {a!r} <-> {b!r}  ({n} docs, {tag})")

    cross = sum(1 for a, b, _, _ in co if not (label_of[a] & label_of[b]))
    print(f"\n  of those, {cross} join concepts of different labels")

    roots = {b for _, b in narrows}
    children = {a for a, _ in narrows}
    print(f"  hierarchy: {len(roots)} general terms, {len(children)} specialisations")
    print(f"  deepest fan-out: "
          f"{collections.Counter(b for _, b in narrows).most_common(3)}")

    if args.print_only:
        print("\n--print-only: nothing written")
        return

    out = HERE / f"{domain.key}_depth_seed.cypher"
    lines = [
        f"// {out.name}",
        "//",
        "// GENERATED by graph/mine_graph_depth.py — do not hand-edit.",
        f"// domain: {domain.key}   corpus: {domain.dataset}",
        f"// min_jaccard {min_j}   min_docs {args.min_docs}",
        "//",
        "// Structure below the first level: NARROWS is a specialisation taken",
        "// from whole-word containment, CO_OCCURS is document co-occurrence over",
        "// the domain's own corpus. Nothing in the serving path reads either —",
        "// services/ontology_check.py loads the markers JSON, not the graph.",
        "//",
        "// MATCH would silently create nothing if a Concept were absent, so both",
        "// ends are MERGEd. Load the concept seed first all the same.",
        "",
    ]
    for a, b in narrows:
        lines.append(
            f'MERGE (a:Concept {{term: "{a}"}}) '
            f'MERGE (b:Concept {{term: "{b}"}}) '
            f'MERGE (a)-[:NARROWS]->(b);'
        )
    lines.append("")
    for a, b, j, n in co:
        lines.append(
            f'MERGE (a:Concept {{term: "{a}"}}) '
            f'MERGE (b:Concept {{term: "{b}"}}) '
            f'MERGE (a)-[r:CO_OCCURS]-(b) SET r.jaccard = {j}, r.documents = {n};'
        )
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nwrote {out.relative_to(ROOT)}  "
          f"({len(narrows)} NARROWS + {len(co)} CO_OCCURS)")


if __name__ == "__main__":
    main()
