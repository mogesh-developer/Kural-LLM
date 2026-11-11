
from pathlib import Path
import re

path = Path("data/processed/train.txt")
lines = path.read_text(encoding="utf-8").splitlines()

# Tamil letter + optional vowel sign/virama + space +
# Tamil consonant. This is a review heuristic, not a
# definitive Tamil grammar rule.
pattern = re.compile(
    r"([\u0B95-\u0BB9][\u0BBE-\u0BCD]?) "
    r"([\u0B95-\u0BB9])"
)

shown = 0

for line_no, line in enumerate(lines, start=1):
    for match in pattern.finditer(line):
        start = max(0, match.start() - 35)
        end = min(len(line), match.end() + 35)

        print(f"Line {line_no}: {line[start:end]!r}")
        shown += 1

        if shown >= 30:
            break

    if shown >= 30:
        break

print(f"\nExamples shown: {shown}")
