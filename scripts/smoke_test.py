"""Quick check that the trained v1 model loads and predicts sensibly."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# A test from each class — straightforward cases
TEST_INPUTS = [
    ("The Vendor shall indemnify the Purchaser against all claims arising from breach of warranty under Clause 12.", "Contract"),
    ("The accused was charged under Section 302 IPC for the murder of his neighbour during a property dispute. The Sessions Court rejected bail.", "Criminal"),
    ("The petitioner challenges the constitutional validity of the impugned amendment as violating Article 14 of the Constitution.", "Constitutional"),
    ("The NCLT initiated CIRP against the corporate debtor following default in repayment of financial debt of Rs. 50 crores.", "Corporate"),
    ("The lease deed for the commercial premises was executed for a term of 11 months at a monthly rent of Rs. 50,000.", "Property"),
    ("The petitioner seeks dissolution of marriage on the grounds of cruelty and irretrievable breakdown of the marital relationship.", "Family"),
    ("The Industrial Tribunal held that the dismissal was invalid as the principles of natural justice were violated during the domestic inquiry.", "Labour"),
    ("The plaintiff sued the defendant for trademark infringement of its registered mark 'BrandX' under Section 29 of the Trade Marks Act.", "IP"),
    ("The Assessing Officer reopened the assessment under Section 147 of the Income Tax Act based on tangible material indicating escapement of income.", "Tax"),
    ("The plaint was rejected under Order VII Rule 11 of the CPC as it did not disclose any cause of action against the defendant.", "Civil Procedure"),
]

print(f"\n{'='*80}")
print(f"SMOKE TEST — predicting on 10 prepared examples")
print(f"{'='*80}\n")

correct = 0
for text, expected in TEST_INPUTS:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    conf, pred = torch.max(probs, dim=-1)
    pred_label = model.config.id2label[pred.item()]
    
    is_correct = expected.lower() in pred_label.lower()
    correct += int(is_correct)
    mark = "✓" if is_correct else "✗"
    print(f"{mark} Expected: {expected:20s} | Predicted: {pred_label:30s} | Conf: {conf.item():.3f}")
    print(f"   Text: {text[:90]}...")
    print()

print(f"\n{correct}/10 correct on smoke test")
