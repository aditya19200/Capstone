# ML Pipeline — v1 Model Handoff

## What's in this delivery

| Item | Location | Description |
|------|----------|-------------|
| Trained model | `models/v1/` (Google Drive) | Fine-tuned InLegalBERT, 10-class Indian legal classifier |
| Inference module | `inference.py` | Python module with 4 ready-to-use functions |
| Training dataset | `datasets/v1.csv` | 1701 labeled samples used for training |
| Test metrics | `metrics/v1_metrics.json` | Accuracy, F1, confusion matrix |
| Hard test metrics | `metrics/v1_hard_test.json` | 26-case stress test results |
| Training script | `scripts/retrain.py` | Reproducible training pipeline for future v2/v3 |

## Model performance

- **Test accuracy:** 75% on held-out test set (256 samples the model never saw)
- **Hard test accuracy:** 81% on 26 designed-to-break cases
- **Confidence range:** 0.22 – 0.95 (properly calibrated, NOT stuck at 0.99)
- **Best classes:** IP (F1=0.92), Corporate (F1=0.87), Tax (F1=0.83)
- **Weakest classes:** Contract (F1=0.61), Criminal (F1=0.61), Civil Procedure (F1=0.68)

## Setup

### 1. Download the model folder

Download `models_v1.tar.gz` from the shared Google Drive link. Extract it:

```bash
tar -xzf models_v1.tar.gz -C /path/to/your/backend/
```

This creates a `models/v1/` folder containing `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`.

### 2. Set environment variable

Add to your `.env` file:

```
MODEL_PATH=/absolute/path/to/models/v1
```

### 3. Install dependencies

```bash
pip install torch transformers shap numpy
```

## Using `inference.py`

Copy `inference.py` into your backend. Import and call:

```python
from inference import predict_text, predict_paragraphs, predict_batch, explain_text
```

### `predict_text(text: str) -> dict`

Classify a single piece of legal text. Fast (~50ms on GPU).

```python
result = predict_text("The accused was charged under Section 420 IPC.")
# {
#     "label": "Criminal Law",
#     "label_id": 1,
#     "confidence": 0.751,
#     "all_probabilities": {"Contract Law": 0.03, "Criminal Law": 0.751, ...}
# }
```

### `predict_paragraphs(text: str) -> dict`

Split text by blank lines, classify each paragraph, return aggregate label.

```python
result = predict_paragraphs("Paragraph one...\n\nParagraph two...")
# {
#     "document_label": "Criminal Law",
#     "paragraph_count": 2,
#     "paragraphs": [
#         {"paragraph": "...", "label": "...", "confidence": 0.75, ...},
#         ...
#     ]
# }
```

### `predict_batch(texts: list[str]) -> list[dict]`

Classify a list of texts. For CSV bulk uploads.

```python
results = predict_batch(["text1", "text2", "text3"])
# Returns a list of predict_text outputs
```

### `explain_text(text: str) -> dict`

Generate SHAP token-level importance scores. **SLOW (30-90 seconds).** Use from the XAI worker only, never in the main request path.

```python
result = explain_text("The NCLT admitted the insolvency petition.")
# {
#     "predicted_label": "Corporate / Company Law",
#     "confidence": 0.923,
#     "token_importance": [{"token": "NCLT", "importance": 0.082}, ...],
#     "top_tokens": [top 10 by absolute importance],
#     "summary": "The prediction was primarily driven by: NCLT, insolvency, petition."
# }
```

## Integration with existing backend code

### For `model_service.py`

Your existing `model_service.py` already works with this model. Just update `MODEL_PATH` in `.env` to point at the new `models/v1/` folder. **One change required:** update `max_length=512` to `max_length=256` on line 92. The v1 model was trained at 256 tokens.

### For `shap_service.py`

Your existing `shap_service.py` already works. **One change required:** update `max_length=512` to `max_length=256` on line 36 (inside `_predict_fn`). Same reason as above.

### For `xai_worker.py`

No changes needed. It calls `shap_service.generate_explanation()` which will work with the new model.

## Label mapping

The 10 classes, as stored in `models/v1/config.json`:

| ID | Label |
|----|-------|
| 0 | Contract Law |
| 1 | Criminal Law |
| 2 | Constitutional Law |
| 3 | Corporate / Company Law |
| 4 | Property / Real Estate Law |
| 5 | Family Law |
| 6 | Labour & Employment Law |
| 7 | Intellectual Property Law |
| 8 | Taxation Law |
| 9 | Civil Procedure / Other |

These are also available programmatically via `model.config.id2label` after loading.

## For the Neo4j team

The 10 label names above are your ontology root nodes. They're stored in `models/v1/config.json` under the `id2label` key. Parse that JSON to populate your knowledge graph.

## Known limitations

1. **Max input length is 256 tokens (~200 words).** Longer text is silently truncated. For long documents, use `predict_paragraphs()` which chunks first.
2. **Cross-domain cases are harder.** When legal text spans multiple domains (e.g., a criminal fraud case involving a contract), the model picks the dominant domain but may disagree with human judgment. This is by design — low-confidence predictions route to human reviewers in the active learning loop.
3. **Non-legal text gets classified anyway.** The model has no "unknown" class. Feed it a cooking recipe and it will still output a legal label (with low confidence). Use the confidence threshold (recommend < 0.5) to flag uncertain predictions.
4. **SHAP is slow.** 30-90 seconds per explanation. Must run async, never in the request path.

## Dataset provenance (for the report)

v1.csv was built from 5 sources:
- **ILC** (d0r1h/ILC): 788 real Indian Supreme Court case summaries, labeled via Gemini
- **Original**: 530 hand-written legal text examples from the initial training notebook
- **Synthetic**: 130 examples generated by Gemini 3.1 Pro (70 Tax, 60 IP) to fill class gaps
- **MILPaC CCI**: 129 Competition Commission of India FAQ answers (Corporate Law)
- **MILPaC IP + PDF**: 97 examples from Indian IP FAQ + legal template explanatory prose
