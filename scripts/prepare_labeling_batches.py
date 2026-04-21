"""Split ILC summaries into batches of 20 for Gemini labeling."""
import pandas as pd
import os

df = pd.read_csv("datasets/ilc_raw/ilc_summaries.csv")
print(f"Loaded {len(df)} summaries")

# Take first 1000 — enough for our needs, leaves headroom for retry/discard
df = df.head(1000).reset_index(drop=True)

# Truncate any very long summaries to first 400 words to keep prompts manageable
def truncate_summary(text, max_words=400):
    words = str(text).split()
    return " ".join(words[:max_words])

df['Summary'] = df['Summary'].apply(truncate_summary)

# Write batches of 20 to text files
os.makedirs("datasets/labeling_batches", exist_ok=True)
BATCH_SIZE = 20
n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(df))
    batch = df.iloc[start:end]
    
    lines = []
    for i, row in enumerate(batch.itertuples(), start=1):
        lines.append(f"--- Summary {i} ---")
        lines.append(row.Summary)
        lines.append("")
    
    batch_text = "\n".join(lines)
    
    with open(f"datasets/labeling_batches/batch_{batch_idx:03d}.txt", "w") as f:
        f.write(batch_text)

print(f"Created {n_batches} batches in datasets/labeling_batches/")
print(f"You'll send each batch to Gemini and save the labels to datasets/labeling_results/batch_NNN.txt")
