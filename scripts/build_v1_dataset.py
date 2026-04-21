"""Merge ILC + original + supplementary + synthetic → datasets/v1.csv"""
import sys
import pandas as pd
import random

sys.path.insert(0, "datasets")
from original_data import original_data

random.seed(42)

LABEL_MAP = {
    0: "Contract Law", 1: "Criminal Law", 2: "Constitutional Law",
    3: "Corporate / Company Law", 4: "Property / Real Estate Law",
    5: "Family Law", 6: "Labour & Employment Law",
    7: "Intellectual Property Law", 8: "Taxation Law",
    9: "Civil Procedure / Other"
}

# 1. Original hand-written
orig_df = pd.DataFrame(original_data, columns=['text', 'label'])
orig_df['label_name'] = orig_df['label'].map(LABEL_MAP)
orig_df['source'] = 'original'
print(f"Original: {len(orig_df)}")

# 2. ILC labeled
ilc_df = pd.read_csv("datasets/ilc_labeled.csv")
ilc_df['source'] = 'ilc'
print(f"ILC: {len(ilc_df)}")

# 3. Supplementary (PDF + MILPaC)
supp_df = pd.read_csv("datasets/supplementary_labeled.csv")
print(f"Supplementary (PDF+MILPaC): {len(supp_df)}")

# 4. Synthetic (Gemini)
syn_df = pd.read_csv("datasets/synthetic_labeled.csv")
print(f"Synthetic: {len(syn_df)}")

# Merge all
all_df = pd.concat([orig_df, ilc_df, supp_df, syn_df], ignore_index=True)
print(f"\nBefore dedup: {len(all_df)}")
all_df = all_df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(f"After dedup: {len(all_df)}")

# Downsample Criminal Law (label 1) to max 200 to balance dataset
crim_df = all_df[all_df['label'] == 1]
other_df = all_df[all_df['label'] != 1]
if len(crim_df) > 200:
    crim_sample = crim_df.sample(n=200, random_state=42)
    print(f"\nDownsampled Criminal Law: {len(crim_df)} → 200")
    all_df = pd.concat([other_df, crim_sample], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFINAL TOTAL: {len(all_df)}")
print(f"\nFinal class distribution:")
dist = all_df.groupby(['label', 'label_name']).size()
for (lbl, name), cnt in dist.items():
    bar = '█' * (cnt // 10)
    print(f"  {lbl} {name:35s} {cnt:4d}  {bar}")

print(f"\nSource distribution:")
print(all_df['source'].value_counts().to_string())

all_df.to_csv("datasets/v1.csv", index=False)
print(f"\nSaved datasets/v1.csv")
