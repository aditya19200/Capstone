"""Download the d0r1h/ILC dataset and save summaries to CSV for labeling."""
from datasets import load_dataset
import pandas as pd
import os

print("Downloading d0r1h/ILC dataset from HuggingFace...")
dataset = load_dataset("d0r1h/ILC")

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"Train rows: {len(train_df)}")
print(f"Test rows:  {len(test_df)}")
print(f"Columns:    {train_df.columns.tolist()}")
print(f"\nFirst row sample:")
print(f"  Title: {train_df.iloc[0]['Title'][:100]}")
print(f"  Summary length (words): {len(train_df.iloc[0]['Summary'].split())}")
print(f"  Case length (words):    {len(train_df.iloc[0]['Case'].split())}")

# Stats on summary length
summary_word_counts = train_df['Summary'].str.split().str.len()
print(f"\nSummary word count distribution:")
print(summary_word_counts.describe())

# Save just what we need
os.makedirs("datasets/ilc_raw", exist_ok=True)
train_df[['Title', 'Summary']].to_csv("datasets/ilc_raw/ilc_summaries.csv", index=False)
print(f"\nSaved {len(train_df)} summaries to datasets/ilc_raw/ilc_summaries.csv")
