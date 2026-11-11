
from pathlib import Path
from collections import Counter
import sys

from preprocess import clean_line

path = Path("data/raw/AE_wiki_34.txt")

counts = Counter()
examples = {}

with path.open("r", encoding="utf-8-sig", errors="replace") as f:
    for line_no, raw in enumerate(f, 1):
        cleaned, reason = clean_line(
            raw,
            remove_english=False,
            remove_symbols=True,
        )

        if cleaned is None:
            key = reason or "unknown"
            counts[key] += 1
            examples.setdefault(key, []).append(
                (line_no, raw.rstrip("\n")[:250])
            )

print("Rejection counts:")
for reason, count in counts.most_common():
    print(f"{reason}: {count}")

print("\nExamples:")
for reason, items in examples.items():
    print(f"\n--- {reason} ---")
    for line_no, text in items[:5]:
        print(f"Line {line_no}: {text!r}")
