"""v1 retraining: balanced multi-source dataset, proper splits, held-out test."""
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

LABEL_MAP = {
    0: "Contract Law", 1: "Criminal Law", 2: "Constitutional Law",
    3: "Corporate / Company Law", 4: "Property / Real Estate Law",
    5: "Family Law", 6: "Labour & Employment Law",
    7: "Intellectual Property Law", 8: "Taxation Law",
    9: "Civil Procedure / Other"
}

# Load data
df = pd.read_csv("datasets/v1.csv")
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(int).tolist()
print(f"Loaded {len(texts)} samples")
print(f"Class distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")

# 70/15/15 stratified split
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.30, random_state=SEED, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
)
print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

# Save the held-out test set for later debugging / demo
pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(
    "datasets/v1_holdout_test.csv", index=False
)

# Model
MODEL_BASE = "law-ai/InLegalBERT"
print(f"\nLoading {MODEL_BASE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=10,
    id2label={str(k): v for k, v in LABEL_MAP.items()},
    label2id={v: k for k, v in LABEL_MAP.items()},
    use_safetensors=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")

class LegalDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=256, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_loader = DataLoader(LegalDataset(train_texts, train_labels), batch_size=16, shuffle=True)
val_loader = DataLoader(LegalDataset(val_texts, val_labels), batch_size=16)
test_loader = DataLoader(LegalDataset(test_texts, test_labels), batch_size=16)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

def evaluate(loader):
    model.eval()
    preds, true, confs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            preds.extend(pred.cpu().tolist())
            true.extend(batch["labels"].cpu().tolist())
            confs.extend(conf.cpu().tolist())
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average=None, labels=list(range(10)), zero_division=0)
    return acc, f1, preds, true, confs

EPOCHS = 3
training_loss = []
print("\n=== Training ===")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += outputs.loss.item()
    avg_loss = total_loss / len(train_loader)
    val_acc, val_f1, _, _, val_confs = evaluate(val_loader)
    training_loss.append(round(avg_loss, 4))
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Mean Val Confidence: {np.mean(val_confs):.4f}")

# Final test on held-out set
test_acc, test_f1, test_preds, test_true, test_confs = evaluate(test_loader)
print(f"\n=== HELD-OUT TEST SET (model never saw these) ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"Mean Confidence: {np.mean(test_confs):.4f}")
print(f"Confidence range: {min(test_confs):.4f} - {max(test_confs):.4f}")
print("\nPer-class F1:")
for i, name in LABEL_MAP.items():
    print(f"  {i} {name:35s} F1={test_f1[i]:.4f}")

# Save model
os.makedirs("models/v1", exist_ok=True)
model.save_pretrained("models/v1")
tokenizer.save_pretrained("models/v1")
print("\nModel saved to models/v1/")

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics = {
    "model_version": "v1",
    "trained_at": pd.Timestamp.now().isoformat(),
    "sample_count": len(texts),
    "train_count": len(train_texts),
    "val_count": len(val_texts),
    "test_count": len(test_texts),
    "overall_accuracy": round(test_acc, 4),
    "mean_confidence": round(float(np.mean(test_confs)), 4),
    "f1_per_class": {LABEL_MAP[i]: round(float(test_f1[i]), 4) for i in range(10)},
    "confusion_matrix": confusion_matrix(test_true, test_preds).tolist(),
    "training_loss": training_loss,
}
with open("metrics/v1_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to metrics/v1_metrics.json")
