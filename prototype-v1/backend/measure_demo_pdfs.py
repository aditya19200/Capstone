"""
measure_demo_pdfs.py — the numbers in sample-data/demo/medical-pdfs/README.md.

Each demo PDF goes through the path an upload takes: pdf_extract.extract()
packing by the tokeniser's own count, utils.text.normalize(), then
model_service.predict_batch() in the batch worker's group size. Predictions at or above the auto-accept bar are run
through the ontology check, as the review queue does. Gold for every passage
is the file's specialty, because each file holds held-out records of one
specialty only.

The README's first numbers were measured by hand, splitting by characters
rather than by tokens as an upload does, so they described passages no upload
produces. This script measures the real path, and is what to re-run whenever
the served medical model changes.

    cd backend
    ACTIVE_DOMAIN=medical MODEL_PATH=./models/<checkpoint> ./venv/bin/python measure_demo_pdfs.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.settings import settings  # noqa: E402
from services import pdf_extract  # noqa: E402
from services.model_service import ACTIVE_DOMAIN, model_service  # noqa: E402
from services.ontology_check import ontology_check  # noqa: E402
from utils.text import count_tokens, normalize  # noqa: E402

PDF_DIR = HERE.parent / "sample-data" / "demo" / "medical-pdfs"
FILES = [
    ("neurology-consults.pdf", "Neurology"),
    ("cardiology-catheterization.pdf", "Cardiovascular / Pulmonary"),
]


def main():
    if ACTIVE_DOMAIN.key != "medical":
        sys.exit("run with ACTIVE_DOMAIN=medical")
    model_service.load()
    ontology_check.load()
    print(f"checkpoint: {model_service.loaded_path}\n")

    bar = settings.CONFIDENCE_HIGH
    for name, gold in FILES:
        result = pdf_extract.extract((PDF_DIR / name).read_bytes(),
                                     token_counter=count_tokens,
                                     max_tokens=settings.MAX_LENGTH)
        texts = [t for t in (normalize(d) for d in result.documents) if t]
        # predict_batch in groups of BATCH_SIZE, as workers/batch_worker.py
        # claims and classifies them: batching pads, which can move a
        # confidence in the third decimal, and 0.85 is a hard line.
        preds = []
        for i in range(0, len(texts), settings.BATCH_SIZE):
            preds += model_service.predict_batch(texts[i:i + settings.BATCH_SIZE])

        right = sum(p["predicted_label"] == gold for p in preds)
        accepted = [(t, p) for t, p in zip(texts, preds) if p["confidence"] >= bar]
        flags = [(p, ontology_check.check(t, p["predicted_label"])) for t, p in accepted]
        flags = [(p, c) for p, c in flags if c]
        caught = sum(p["predicted_label"] != gold and c["suggested_label"] == gold
                     for p, c in flags)
        false_pos = sum(p["predicted_label"] == gold for p, c in flags)

        # The three routing bands (services/active_learning.py), and what the
        # Review queue lists: under REVIEW_THRESHOLD, plus ontology flags.
        middle = sum(settings.CONFIDENCE_LOW <= p["confidence"] < bar for p in preds)
        low = sum(p["confidence"] < settings.CONFIDENCE_LOW for p in preds)
        queue = sum(p["confidence"] < settings.REVIEW_THRESHOLD for p in preds)
        print(f"{name}: {result.page_count} pages -> {len(texts)} docs | "
              f"accuracy {right}/{len(texts)} = {right / len(texts):.1%} | "
              f"avg conf {sum(p['confidence'] for p in preds) / len(preds):.3f}")
        print(f"   auto-accepted {len(accepted)} (wrong among them "
              f"{sum(p['predicted_label'] != gold for _, p in accepted)}) | "
              f"needs explanation {middle} | routed to a reviewer {low} | "
              f"in the Review queue {queue} on confidence + flags below")
        print(f"   ontology flags {len(flags)}: {caught} catch a wrong label, "
              f"{false_pos} on a correct one")
        for p, c in flags:
            print(f"   {p['confidence']:.3f}  model said {p['predicted_label']}  ->  "
                  f"check says {c['suggested_label']}  "
                  f"(evidence {c['evidence_score']:.1f} vs {c['predicted_label_score']:.1f}; "
                  f"{', '.join(c['matched_terms'][:4])})")
        print()


if __name__ == "__main__":
    main()
