
from pathlib import Path

files = [
    Path("data/processed/train.txt"),
    Path("data/processed/validation.txt"),
    Path("data/processed/test.txt"),
]

for path in files:
    if not path.exists():
        continue

    print(f"\n{'=' * 60}")
    print(path)

    lines = path.read_text(encoding="utf-8").splitlines()

    suspicious = [
        (i, line)
        for i, line in enumerate(lines, start=1)
        if len(line.strip()) > 1000
    ]

    print("Total lines:", len(lines))
    print("Lines over 1,000 characters:", len(suspicious))

    for line_no, line in suspicious[:10]:
        print(f"\nLine {line_no}: {len(line):,} characters")
        print(repr(line[:300]))
        print("...")
        print(repr(line[-200:]))
