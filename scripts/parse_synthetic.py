"""Parse Gemini synthetic .txt outputs into a CSV."""
import os
import re
import pandas as pd

LABEL_MAP = {
    0: "Contract Law", 1: "Criminal Law", 2: "Constitutional Law",
    3: "Corporate / Company Law", 4: "Property / Real Estate Law",
    5: "Family Law", 6: "Labour & Employment Law",
    7: "Intellectual Property Law", 8: "Taxation Law",
    9: "Civil Procedure / Other"
}

rows = []
syn_dir = "datasets/synthetic"

for fname in sorted(os.listdir(syn_dir)):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(syn_dir, fname)
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Match pattern: ("text", N)  — handles escaped quotes and multi-line
    matches = re.findall(r'\("(.+?)"\s*,\s*(\d)\s*\)', content, re.DOTALL)
    file_count = 0
    for text, label in matches:
        label = int(label)
        # Unescape quotes and normalize whitespace
        text = text.replace('\\"', '"').replace("\\'", "'")
        text = re.sub(r'\s+', ' ', text).strip()
        if text and 0 <= label <= 9 and len(text.split()) >= 20:
            rows.append({
                "text": text, "label": label,
                "label_name": LABEL_MAP[label], "source": "synthetic"
            })
            file_count += 1
    print(f"  {fname}: parsed {file_count} examples")

print(f"\nTotal synthetic: {len(rows)}")
out_df = pd.DataFrame(rows)
print(f"\nClass distribution:")
print(out_df['label'].value_counts().sort_index().to_string())

out_df.to_csv("datasets/synthetic_labeled.csv", index=False)
print("\nSaved to datasets/synthetic_labeled.csv")
