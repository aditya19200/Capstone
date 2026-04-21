"""
Smoke test for SHAP on the v1 model.
Confirms SHAP can compute token-level importance scores.
"""
import os
import torch
import numpy as np
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.getenv("MODEL_PATH", "models/v1")
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

LABEL_MAP = {int(k): v for k, v in model.config.id2label.items()}
LABEL_NAMES = [LABEL_MAP[i] for i in range(10)]


def predict_proba(texts):
    """SHAP's required interface: list of strings -> (n, 10) probability array."""
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    else:
        texts = list(texts)
    
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    return probs.cpu().numpy()


print("Building SHAP explainer...")
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_proba, masker, output_names=LABEL_NAMES)

test_text = (
    "The accused was charged under Section 420 IPC for criminal breach "
    "of trust by misappropriating funds entrusted to him as a trustee. "
    "The Sessions Court rejected the bail application."
)

print(f"\nInput text:\n  {test_text}\n")
print("Computing SHAP values (30-90 seconds expected)...\n")

shap_values = explainer([test_text])

probs = predict_proba([test_text])[0]
pred_id = int(np.argmax(probs))
pred_label = LABEL_NAMES[pred_id]
confidence = float(probs[pred_id])

print(f"Predicted: {pred_label}  (confidence: {confidence:.3f})\n")

print(f"Top 10 tokens pushing TOWARD '{pred_label}':")
print("-" * 60)
tokens = shap_values.data[0]
contributions = shap_values.values[0][:, pred_id]

token_contribs = list(zip(tokens, contributions))
sorted_contribs = sorted(token_contribs, key=lambda x: x[1], reverse=True)

for token, value in sorted_contribs[:10]:
    bar = "█" * int(abs(value) * 200)
    print(f"  {repr(token):25s}  +{value:.4f}  {bar}")

print(f"\nTop 5 tokens pushing AWAY FROM '{pred_label}':")
print("-" * 60)
for token, value in sorted_contribs[-5:]:
    bar = "█" * int(abs(value) * 200)
    print(f"  {repr(token):25s}  {value:+.4f}  {bar}")

print("\nGenerating HTML visualization...")
try:
    html = shap.plots.text(shap_values[0], display=False)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/shap_test_output.html", "w") as f:
        f.write("<html><body>")
        f.write(html if html else "<p>SHAP text plot rendered</p>")
        f.write("</body></html>")
    print("HTML saved to metrics/shap_test_output.html")
    print("Open it in Firefox/Chrome to see the colored token visualization.")
except Exception as e:
    print(f"HTML save skipped: {e}")
    print("(Not critical — the text output above is the important part)")

print("\n=== SHAP TEST COMPLETE ===")
