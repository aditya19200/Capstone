"""Try loading several Indian legal datasets to see which are available."""
from datasets import load_dataset

candidates = [
    "opennyaiorg/InJudgements_Dataset",
    "opennyaiorg/aalap_instruction_dataset",
    "viber1/indian-law-dataset",
    "ninadn/indian-legal",
    "kshitij230/Indian-Law",
    "jonathanli/legal-advice-reddit",  # not Indian but has IP, contract, tax labels
    "lex_glue",  # multi-jurisdiction legal NLP benchmark
    "coastalcph/lex_glue",
]

for name in candidates:
    print(f"\n{'='*60}")
    print(f"Trying: {name}")
    try:
        ds = load_dataset(name, split="train", streaming=True)
        # Just peek at first row
        first = next(iter(ds))
        print(f"  ✓ Available")
        print(f"  Columns: {list(first.keys())[:8]}")
        # Print first 200 chars of first text-like field
        for key, val in first.items():
            if isinstance(val, str) and len(val) > 50:
                print(f"  Sample [{key}]: {val[:200]}...")
                break
    except Exception as e:
        err = str(e)[:150]
        print(f"  ✗ Failed: {err}")
