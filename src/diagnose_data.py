
from pathlib import Path
import random
import statistics
import re

FILES = [
    Path("data/processed/train.txt"),
    Path("data/processed/validation.txt"),
    Path("data/processed/test.txt"),
]

random.seed(42)

for path in FILES:
    if not path.exists():
        print(f"\nSKIPPED: {path} not found")
        continue

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    lengths = [len(line) for line in lines]
    unique_lines = set(lines)

    all_chars = "".join(lines)
    tamil_chars = sum(
        1 for ch in all_chars
        if "\u0B80" <= ch <= "\u0BFF"
    )
    tamil_ratio = tamil_chars / max(len(all_chars), 1)

    print(f"\n{'=' * 50}")
    print("File:", path)
    print("Non-empty lines:", len(lines))
    print("Unique lines:", len(unique_lines))
    print("Duplicate ratio: {:.2f}%".format(
        100 * (1 - len(unique_lines) / len(lines))
    ))
    print("Characters per line — median:", statistics.median(lengths))
    print("Characters per line — min/max:", min(lengths), max(lengths))
    print("Tamil Unicode character ratio: {:.1f}%".format(
        tamil_ratio * 100
    ))

    print("\nSample lines:")
    for line in random.sample(lines, min(5, len(lines))):
        print("-", line[:250])
