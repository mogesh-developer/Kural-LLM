
from pathlib import Path
import unicodedata
import re

path = Path("data/processed/train.txt")
lines = path.read_text(encoding="utf-8").splitlines()

# Longest non-empty lines
ranked = sorted(
    ((len(line), i + 1, line) for i, line in enumerate(lines) if line.strip()),
    reverse=True,
)

print("TOP 5 LONGEST LINES")
for length, line_no, line in ranked[:5]:
    print(f"\nLine {line_no}: {length:,} characters")
    print("START:", repr(line[:300]))
    print("END:  ", repr(line[-200:]))
    print("Newline-like separators inside:", len(re.findall(r"[\r\n]", line)))
    print("Runs of 5+ spaces:", len(re.findall(r" {5,}", line)))
    print("Replacement characters:", line.count("\uFFFD"))

# Unicode normalization check
changed_lines = 0
changed_chars = 0

for line in lines:
    normalized = unicodedata.normalize("NFC", line)
    if normalized != line:
        changed_lines += 1
        changed_chars += sum(a != b for a, b in zip(line, normalized))
        # Account for length differences too.
        changed_chars += abs(len(line) - len(normalized))

print("\nUNICODE CHECK")
print("Lines changed by NFC normalization:", changed_lines)
print("Approximate changed character positions:", changed_chars)

# Find lines containing unusually long whitespace runs
print("\nLINES WITH 5+ CONSECUTIVE SPACES")
count = 0
for i, line in enumerate(lines, start=1):
    if re.search(r" {5,}", line):
        print(f"Line {i}: {repr(line[:250])}")
        count += 1
        if count >= 10:
            break

# Inspect actual Tamil combining marks in one sample.
print("\nUNICODE CODE POINTS IN FIRST LINE")
for ch in next((x for x in lines if x.strip()), "")[:80]:
    print(repr(ch), f"U+{ord(ch):04X}", unicodedata.name(ch, "UNKNOWN"))
