"""
services/dataset_export.py — the annotated corpus, as a file someone can train on.

Why this exists
---------------
The product of this system is not the classifier. It is the labelled corpus
the classifier helps produce: the model pre-labels, the routing layer decides
what a human should look at, and a reviewer confirms or corrects. What comes
out the other end is training data.

That claim is worth very little while the data is only reachable as rows in a
table, so this writes it out in the form a practitioner would actually use —
JSON Lines, one object per example, ready for `datasets.load_dataset("json",
...)` — together with a card describing what is in it.

The guarantee that makes it honest
----------------------------------
`rows()` is built on exactly what `workers.retrain_worker.build_training_data`
consumes: validated annotations, joined to their prediction's text, normalised
with the same `normalize()` every prediction path uses. The export is not a
report *about* the training data, it is the training data. If the two ever
drift, the retraining results stop being evidence for the exported corpus, so
they share this module rather than each assembling their own view.

Provenance travels with every row: which model version predicted it, what it
predicted, and whether the human agreed. A corpus of labels with no record of
how they were produced is worth less than one that says.
"""

import io
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Tuple

from config.domains import active_domain
from services.supabase_service import get_prediction, list_annotations
from utils.text import normalize

logger = logging.getLogger(__name__)


def rows() -> Tuple[List[Dict], int]:
    """
    Every validated annotation as an export record.

    Returns (records, skipped). A record is skipped when its prediction row
    is missing or its label is not in the active domain — the same two
    conditions build_training_data() skips on, for the same reason.
    """
    domain = active_domain()
    known = set(domain.names)
    out: List[Dict] = []
    skipped = 0

    for ann in list_annotations(status="validated"):
        pred = get_prediction(ann["prediction_id"])
        if pred is None:
            skipped += 1
            continue
        label = ann.get("validated_label")
        if label not in known:
            logger.warning(
                "[dataset_export] annotation %s has label %r, not in domain %r",
                ann.get("id"), label, domain.key,
            )
            skipped += 1
            continue

        predicted = pred.get("predicted_label")
        out.append({
            "text": normalize(pred["text_content"]),
            "label": label,
            "label_id": domain.label2id[label],
            "domain": domain.key,
            # Provenance. "corrected" is the interesting flag: those rows are
            # the ones the model got wrong and a human fixed, which is where
            # the value of this corpus concentrates.
            "predicted_label": predicted,
            "confidence": pred.get("confidence"),
            "corrected": predicted is not None and predicted != label,
            "model_version": pred.get("model_version"),
            "annotated_at": ann.get("annotated_at"),
        })

    return out, skipped


def to_jsonl(records: List[Dict]) -> Iterator[str]:
    """One JSON object per line. Streamed so a large corpus is not held twice."""
    for record in records:
        yield json.dumps(record, ensure_ascii=False) + "\n"


def stats(records: List[Dict]) -> Dict:
    """Counts the card is built from, and that a reader would ask for first."""
    by_label = Counter(r["label"] for r in records)
    corrected = sum(1 for r in records if r["corrected"])
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]
    domain = active_domain()
    return {
        "domain": domain.key,
        "display_name": domain.display_name,
        "total": len(records),
        "corrected": corrected,
        "confirmed": len(records) - corrected,
        "label_counts": {name: by_label.get(name, 0) for name in domain.names},
        "labels_unused": [n for n in domain.names if by_label.get(n, 0) == 0],
        "mean_confidence": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "model_versions": sorted(
            {r["model_version"] for r in records if r.get("model_version")}
        ),
    }


def card(records: List[Dict]) -> str:
    """
    A dataset card, in the shape a HuggingFace reader expects.

    Deliberately states the limits. A corpus of 40 rows described as though
    it were a benchmark is worse than one that says what it is, and the
    reader of this card is likely to be the examiner asking how big it
    really is.
    """
    s = stats(records)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pct = (100 * s["corrected"] / s["total"]) if s["total"] else 0.0

    lines = [
        f"# {s['display_name']} — annotated corpus",
        "",
        f"Exported {generated} from the annotation framework "
        f"(domain `{s['domain']}`).",
        "",
        "## What this is",
        "",
        "Human-validated labels. Each document was classified by a fine-tuned",
        "model, routed for review when the model was unsure **or** when the",
        "legal concepts in the text contradicted its answer, and then confirmed",
        "or corrected by a person. Only confirmed rows are here.",
        "",
        "## Size",
        "",
        f"| | |",
        f"|---|---|",
        f"| Examples | **{s['total']}** |",
        f"| Confirmed as predicted | {s['confirmed']} |",
        f"| Corrected by a human | {s['corrected']} ({pct:.1f}%) |",
        f"| Classes | {len(s['label_counts'])} |",
    ]
    if s["mean_confidence"] is not None:
        lines.append(f"| Mean model confidence | {s['mean_confidence']} |")
    lines += [
        "",
        "## Label distribution",
        "",
        "| Label | Examples |",
        "|---|---:|",
    ]
    for name, count in s["label_counts"].items():
        lines.append(f"| {name} | {count} |")

    lines += [
        "",
        "## Fields",
        "",
        "| Field | Meaning |",
        "|---|---|",
        "| `text` | The document, normalised exactly as the model receives it |",
        "| `label` | The human-validated label. **This is the target** |",
        "| `label_id` | Its index in the domain's label order |",
        "| `predicted_label` | What the model said before review |",
        "| `confidence` | How sure the model was |",
        "| `corrected` | True where the human disagreed with the model |",
        "| `model_version` | Which checkpoint produced the prediction |",
        "| `annotated_at` | When it was validated |",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        'ds = load_dataset("json", data_files="dataset.jsonl")["train"]',
        "```",
        "",
        "## Limits — read before using this",
        "",
    ]

    limits = [
        "`corrected: true` rows are where this corpus earns its keep — they are "
        "the cases the model got wrong and a human fixed. They are also, by "
        "construction, **not** a random sample of the domain: they are what the "
        "routing layer chose to show a reviewer.",
        "Documents the model was confident about and that raised no concept "
        "conflict were auto-accepted and never reviewed, so they are absent. "
        "This corpus is biased toward hard cases.",
        "Labels come from the reviewers available to this project. Inter-annotator "
        "agreement has not been measured.",
    ]
    if s["total"] < 200:
        limits.insert(0, f"**This export contains {s['total']} examples.** That is a "
                         "demonstration of the pipeline, not a corpus to train a "
                         "production model on.")
    if s["labels_unused"]:
        limits.append("No examples at all for: " + ", ".join(s["labels_unused"]) + ".")
    if len(s["model_versions"]) > 1:
        limits.append("Predictions come from more than one model version "
                      f"({', '.join(s['model_versions'])}), so `confidence` is not "
                      "comparable across all rows.")

    lines += [f"- {item}" for item in limits]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
#
# JSONL answers "can another system consume this". It does not answer "what
# does an annotation from this framework look like", because nobody reads a
# corpus by scrolling an ndjson file, and a spreadsheet of labels shows the
# answers while hiding everything that produced them.
#
# So: the same records, laid out as documents. Each one shows the text with
# the ontology's concept markers highlighted in place, what the model
# predicted and how sure it was, what the human decided, and whether those
# two agreed. The highlighting is not decoration — those matched terms are
# the evidence the consistency check routes on, so a page where the green
# terms sit under a corrected label is the mechanism of this project visible
# in one glance.

_HL_SUPPORTS = "#cdebc5"      # marker supporting the validated label
_HL_CONTRADICTS = "#ffe0a3"   # marker supporting some other label

# Terms are matched the way ontology_check matches them, on word boundaries,
# so "act" does not fire inside "transaction".
_SPAN_PREFER_LONGEST = True


def _escape(text: str) -> str:
    """XML-escape for reportlab's paragraph markup."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _md_inline(line: str) -> str:
    """
    Render the inline Markdown that card() writes, as reportlab markup.

    The limits are written once, in card(), and shown in both the Markdown
    card and this PDF. Rather than keep two copies of that prose in step,
    the PDF reads the card's lines and translates the only two kinds of
    emphasis they use. Anything else is left as text.
    """
    out = _escape(line)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.5">\1</font>', out)
    return out


def _highlight(text: str, validated_label: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Mark up *text* with the ontology concept terms it contains.

    Returns (markup, {label: [terms]}). Terms that support *validated_label*
    are shaded one colour and everything else another, which is the whole
    point: a document whose highlighted vocabulary all belongs to a label the
    model did not choose is what an ontology conflict actually looks like.
    """
    from services.ontology_check import ontology_check

    evidence = ontology_check.evidence(text) if ontology_check.is_loaded() else {}
    if not evidence:
        return _escape(text), {}

    # (start, end, supports_validated) for every occurrence of every marker.
    spans: List[Tuple[int, int, bool]] = []
    lowered = text.lower()
    for label, detail in evidence.items():
        supports = label == validated_label
        for term in detail["terms"]:
            for match in re.finditer(r"\b" + re.escape(term) + r"\b", lowered):
                spans.append((match.start(), match.end(), supports))

    # Markers overlap ("bail" inside "bail application"). Longest-first, then
    # drop anything that starts inside a span already taken, so the rendered
    # text stays a partition of the original rather than nested tags.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen: List[Tuple[int, int, bool]] = []
    for start, end, supports in spans:
        if chosen and start < chosen[-1][1]:
            continue
        chosen.append((start, end, supports))

    out: List[str] = []
    cursor = 0
    for start, end, supports in chosen:
        out.append(_escape(text[cursor:start]))
        colour = _HL_SUPPORTS if supports else _HL_CONTRADICTS
        out.append(f'<font backColor="{colour}">{_escape(text[start:end])}</font>')
        cursor = end
    out.append(_escape(text[cursor:]))

    found = {label: detail["terms"] for label, detail in evidence.items()}
    return "".join(out), found


def to_pdf(records: List[Dict]) -> bytes:
    """
    The annotated corpus as a document someone can read.

    Built from the same records as to_jsonl(), so the two downloads are the
    same corpus in two shapes rather than two views assembled separately.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    s = stats(records)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"{s['display_name']} — annotated corpus",
        author="XAI Legal Annotation Framework",
    )

    base = getSampleStyleSheet()
    style = {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20,
                                spaceAfter=2, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=9.5,
                                   textColor=colors.HexColor("#5b6472"),
                                   spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12,
                             spaceBefore=14, spaceAfter=6,
                             textColor=colors.HexColor("#1f2937")),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5,
                               leading=14, spaceAfter=6),
        "doc": ParagraphStyle("d", parent=base["Normal"], fontSize=10,
                              leading=15.5, spaceBefore=3, spaceAfter=5,
                              leftIndent=6, rightIndent=6),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=8,
                               leading=11.5, textColor=colors.HexColor("#5b6472"),
                               leftIndent=6),
        "entry": ParagraphStyle("e", parent=base["Normal"], fontSize=9.5,
                                leading=13, spaceBefore=10,
                                textColor=colors.HexColor("#111827")),
    }

    def rule():
        t = Table([[""]], colWidths=[doc.width])
        t.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#d7dce3")),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return t

    def kv_table(pairs):
        t = Table([[Paragraph(f"<b>{k}</b>", style["body"]),
                    Paragraph(str(v), style["body"])] for k, v in pairs],
                  colWidths=[doc.width * 0.42, doc.width * 0.58], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#e6e9ee")),
        ]))
        return t

    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    pct = (100 * s["corrected"] / s["total"]) if s["total"] else 0.0

    flow = [
        Paragraph(f"{s['display_name']} — annotated corpus", style["title"]),
        Paragraph(
            f"Exported {generated} · domain <b>{s['domain']}</b> · "
            f"{s['total']} human-validated documents",
            style["subtitle"],
        ),
        rule(),
        Paragraph("What this is", style["h2"]),
        Paragraph(
            "Every document here was classified by a fine-tuned transformer, "
            "routed to a human when the model was unsure <b>or</b> when the "
            "legal concepts present in the text contradicted its answer, and "
            "then confirmed or corrected by that human. The label on each "
            "entry is the human's, not the model's.",
            style["body"],
        ),
        Paragraph(
            "Concept markers found in the text are shaded. "
            f'<font backColor="{_HL_SUPPORTS}">&nbsp;Green&nbsp;</font> marks '
            "vocabulary belonging to the validated label, "
            f'<font backColor="{_HL_CONTRADICTS}">&nbsp;amber&nbsp;</font> '
            "vocabulary belonging to some other label. Amber under a corrected "
            "entry is the ontology check doing its job: the markers disagreed "
            "with the model, the document was put in front of a reviewer, and "
            "the reviewer changed the label.",
            style["body"],
        ),
        Paragraph("Contents", style["h2"]),
        kv_table([
            ("Documents", f"<b>{s['total']}</b>"),
            ("Confirmed as the model predicted", s["confirmed"]),
            ("Corrected by a human", f"{s['corrected']} ({pct:.1f}%)"),
            ("Classes in this domain", len(s["label_counts"])),
            ("Mean model confidence",
             s["mean_confidence"] if s["mean_confidence"] is not None else "—"),
            ("Model versions represented",
             ", ".join(s["model_versions"]) or "—"),
        ]),
        Paragraph("Label distribution", style["h2"]),
    ]

    dist = [["Label", "Documents"]] + [
        [name, str(count)] for name, count in s["label_counts"].items()
    ]
    dist_table = Table(dist, colWidths=[doc.width * 0.72, doc.width * 0.28],
                       hAlign="LEFT")
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f6")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e6e9ee")),
    ]))
    flow.append(dist_table)

    flow.append(Paragraph("Read this before using the corpus", style["h2"]))
    # The card already works out which caveats apply to this particular
    # export; re-deriving them here would let the two disagree.
    for line in card(records).splitlines():
        if line.startswith("- "):
            flow.append(Paragraph("• " + _md_inline(line[2:]), style["body"]))

    # No forced page break here. The cover's length depends on how many
    # labels the domain has and how many caveats apply to this export, so a
    # break lands on a nearly-empty page about as often as a useful one.
    flow.append(Spacer(1, 10))
    flow.append(Paragraph("The documents", style["h2"]))
    flow.append(Paragraph(
        "One entry per validated annotation, in the order they were exported.",
        style["subtitle"],
    ))

    for i, record in enumerate(records, start=1):
        markup, found = _highlight(record["text"], record["label"])
        corrected = record.get("corrected")
        tag = (
            '<font color="#b3261e"><b>CORRECTED</b></font>' if corrected
            else '<font color="#3f6b2b"><b>CONFIRMED</b></font>'
        )
        confidence = record.get("confidence")
        conf = f"{confidence:.3f}" if isinstance(confidence, (int, float)) else "—"

        decision = (
            f"model said <b>{_escape(str(record.get('predicted_label') or '—'))}</b> "
            f"at {conf} confidence &nbsp;→&nbsp; "
            f"human validated <b>{_escape(record['label'])}</b>"
        )
        provenance = (
            f"model version {_escape(str(record.get('model_version') or '—'))}"
            f" · annotated {_escape(str(record.get('annotated_at') or '—'))[:19]}"
        )
        concept_line = " · ".join(
            f"<b>{_escape(label)}</b>: {_escape(', '.join(terms[:6]))}"
            for label, terms in sorted(found.items(),
                                       key=lambda kv: kv[0] != record["label"])
        ) or "no concept markers matched"

        entry = [
            Paragraph(f"<b>{i}. {_escape(record['label'])}</b> &nbsp; {tag}",
                      style["entry"]),
            Paragraph(markup, style["doc"]),
            Paragraph(decision, style["meta"]),
            Paragraph(f"concepts found — {concept_line}", style["meta"]),
            Paragraph(provenance, style["meta"]),
            Spacer(1, 4),
            rule(),
        ]
        # Short entries are held together, because a two-line judgment split
        # across a page break reads badly and costs nothing to avoid. A long
        # one is allowed to split: keeping it whole would push most of a page
        # of white space ahead of it, and over a corpus that adds up to a
        # document nobody wants to scroll.
        flow.extend([KeepTogether(entry)] if len(record["text"]) < 600 else entry)

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#8b93a1"))
        canvas.drawString(20 * mm, 10 * mm,
                          f"{s['display_name']} — annotated corpus · "
                          f"exported {generated}")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
