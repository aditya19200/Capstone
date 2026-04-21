"""Merge Gemini's labels with the original summaries into one labeled CSV."""
import pandas as pd
import re
import os

df = pd.read_csv("datasets/ilc_raw/ilc_summaries.csv").head(1000).reset_index(drop=True)

# Truncate same way as before
df['Summary'] = df['Summary'].apply(lambda t: " ".join(str(t).split()[:400]))

BATCH_SIZE = 20
all_labels = []

batch_files = sorted(os.listdir("datasets/labeling_results"))
print(f"Found {len(batch_files)} label files")

for batch_file in batch_files:
    path = f"datasets/labeling_results/{batch_file}"
    with open(path) as f:
        content = f.read()
    
    # Parse lines like "1: 2"
    batch_labels = {}
    for line in content.strip().split("\n"):
        line = line.strip()
        match = re.match(r"^\s*(\d+)\s*:\s*(\d)\s*$", line)
        if match:
            local_idx = int(match.group(1))
            label = int(match.group(2))
            if 0 <= label <= 9 and 1 <= local_idx <= BATCH_SIZE:
                batch_labels[local_idx] = label
    
    # Fill batch
    for i in range(1, BATCH_SIZE + 1):
        all_labels.append(batch_labels.get(i, None))

# Trim to actual length
all_labels = all_labels[:len(df)]
df['label'] = all_labels

# Stats
print(f"\nTotal summaries: {len(df)}")
print(f"Labeled successfully: {df['label'].notna().sum()}")
print(f"Failed/missing labels: {df['label'].isna().sum()}")

# Drop rows where Gemini didn't return a parseable label
df = df.dropna(subset=['label']).copy()
df['label'] = df['label'].astype(int)

LABEL_MAP = {
    0: "Contract Law", 1: "Criminal Law", 2: "Constitutional Law",
    3: "Corporate / Company Law", 4: "Property / Real Estate Law",
    5: "Family Law", 6: "Labour & Employment Law",
    7: "Intellectual Property Law", 8: "Taxation Law",
    9: "Civil Procedure / Other"
}
df['label_name'] = df['label'].map(LABEL_MAP)

print(f"\nClass distribution:")
print(df['label'].value_counts().sort_index().to_string())

# Save labeled data
df[['Summary', 'label', 'label_name']].rename(columns={'Summary': 'text'}).to_csv(
    "datasets/ilc_labeled.csv", index=False
)
print(f"\nSaved {len(df)} labeled examples to datasets/ilc_labeled.csv")
