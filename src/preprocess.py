
import argparse
import glob
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


# Tamil digits -> English digits
TAMIL_DIGITS = str.maketrans(
    "௦௧௨௩௪௫௬௭௮௯",
    "0123456789",
)

TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
NUM_PLACEHOLDER_RE = re.compile(r"(?<!\w)NUM(?!\w)", re.IGNORECASE)

# Common web/blog footer & social media noise patterns
BOILERPLATE_PATTERNS = [
    re.compile(r"\bposted\s+by\b.*", re.IGNORECASE),
    re.compile(r"\bshare\s+on\s+(facebook|twitter|whatsapp|telegram)\b.*", re.IGNORECASE),
    re.compile(r"\bfollow\s+us\s+on\b.*", re.IGNORECASE),
    re.compile(r"\bread\s+more\s*[:\-]?", re.IGNORECASE),
    re.compile(r"\b(subscribe|comments?|all rights reserved|copyright)\b.*", re.IGNORECASE),
]

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_BLOCK_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
HTML_TAG_RE = re.compile(r"<[^>]*>")

# Wikipedia dump markers
DOC_START_RE = re.compile(r"^\s*<doc\b", re.IGNORECASE)
DOC_END_RE = re.compile(r"</doc\s*>", re.IGNORECASE)

CONTROL_RE = re.compile(
    r"[\u200B-\u200F\uFEFF\u0000-\u0008\u000B\u000C"
    r"\u000E-\u001F\u007F]"
)

ALLOWED_RE = re.compile(
    r"[^\u0B80-\u0BFFA-Za-z0-9\s"
    r".,;:!?'\u2018\u2019\"\u201C\u201D"
    r"\-\u2013\u2014()/\[\]{}%+&=#]"
)

REPEATED_PUNCT_RE = re.compile(r"([!?,.:;])\1{2,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    text = CONTROL_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_unmatched_brackets(text: str) -> str:
    """Remove unmatched brackets while preserving balanced/nested ones."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {closing: opening for opening, closing in pairs.items()}

    stack = []
    remove_positions = set()

    for i, char in enumerate(text):
        if char in pairs:
            stack.append((char, i))
        elif char in closers:
            if stack and stack[-1][0] == closers[char]:
                stack.pop()
            else:
                remove_positions.add(i)

    for _, i in stack:
        remove_positions.add(i)

    return "".join(
        char for i, char in enumerate(text)
        if i not in remove_positions
    )


def clean_record(
    raw: str,
    remove_english: bool = False,
    remove_symbols: bool = True,
) -> tuple[str | None, str | None]:
    """Clean a complete article or a normal text record."""

    text = html.unescape(raw)
    text = unicodedata.normalize("NFC", text)

    if NUM_PLACEHOLDER_RE.search(text):
        return None, "placeholder"

    text = HTML_COMMENT_RE.sub(" ", text)
    text = HTML_BLOCK_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)

    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)

    # Strip out common blog/social media trailing metadata
    for bp in BOILERPLATE_PATTERNS:
        text = bp.sub(" ", text)

    if remove_english:
        text = LATIN_CHAR_RE.sub(" ", text)

    if remove_symbols:
        text = ALLOWED_RE.sub(" ", text)

    text = text.translate(TAMIL_DIGITS)
    text = clean_unmatched_brackets(text)

    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    text = normalize_text(text)
    text = text.strip(" \t,;:-")
    text = normalize_text(text)

    if not text:
        return None, "empty"

    return text, None


def quality_check(
    text: str,
    min_chars: int,
    min_words: int,
    min_tamil_ratio: float,
    max_digit_ratio: float,
    max_latin_ratio: float = 0.20,
) -> tuple[bool, str | None, dict]:
    words = text.split()
    non_space = [char for char in text if not char.isspace()]

    tamil_count = len(TAMIL_CHAR_RE.findall(text))
    latin_count = len(LATIN_CHAR_RE.findall(text))
    digit_count = sum(char.isdigit() for char in text)

    tamil_ratio = tamil_count / max(1, len(non_space))
    latin_ratio = latin_count / max(1, len(non_space))
    digit_ratio = digit_count / max(1, len(non_space))

    stats = {
        "characters": len(text),
        "words": len(words),
        "tamil_ratio": round(tamil_ratio, 4),
        "latin_ratio": round(latin_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
    }

    if len(text) < min_chars:
        return False, "too_short", stats

    if len(words) < min_words:
        return False, "too_few_words", stats

    if tamil_ratio < min_tamil_ratio:
        return False, "low_tamil_ratio", stats

    if latin_ratio > max_latin_ratio:
        return False, "high_latin_ratio", stats

    if digit_ratio > max_digit_ratio:
        return False, "high_digit_ratio", stats

    return True, None, stats


def is_wikipedia_dump(path: Path) -> bool:
    """Detect a Wikipedia <doc> dump from its opening lines."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for _ in range(100):
            line = f.readline()
            if not line:
                break
            if DOC_START_RE.match(line):
                return True
            if line.strip():
                # Keep checking: introductory lines may precede the first doc.
                continue
    return False


def iter_wikipedia_articles(path: Path):
    """
    Yield one record per <doc>...</doc> block.
    Article title metadata is not included; article body is preserved.
    """
    inside_doc = False
    article_parts = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")

            if not inside_doc:
                if DOC_START_RE.match(line):
                    inside_doc = True
                    article_parts = []

                    # Handle unusual dumps where content follows the opening tag.
                    start_end = line.find(">")
                    remainder = line[start_end + 1:] if start_end >= 0 else ""
                    if remainder.strip():
                        article_parts.append(remainder)
                continue

            if DOC_END_RE.search(line):
                before_end = DOC_END_RE.split(line, maxsplit=1)[0]
                if before_end.strip():
                    article_parts.append(before_end)

                article = " ".join(part.strip() for part in article_parts if part.strip())
                if article.strip():
                    yield article

                inside_doc = False
                article_parts = []
                continue

            # Preserve article text, combining physical lines into one record.
            if line.strip():
                article_parts.append(line.strip())

    # Do not silently discard a final article if the closing marker is missing.
    if inside_doc and article_parts:
        article = " ".join(part.strip() for part in article_parts if part.strip())
        if article.strip():
            yield article


def iter_plain_records(path: Path):
    """For ordinary text files, use each non-empty physical line as a record."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if line.strip():
                yield line.rstrip("\r\n")


def iter_records(path: Path):
    """Automatically choose Wikipedia article mode or plain-text mode."""
    if is_wikipedia_dump(path):
        yield from iter_wikipedia_articles(path)
    else:
        yield from iter_plain_records(path)


def expand_inputs(input_args: list[str]) -> list[Path]:
    files = []

    for item in input_args:
        path = Path(item)

        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                p for p in path.rglob("*.txt") if p.is_file()
            )
        else:
            files.extend(
                Path(p) for p in glob.glob(item, recursive=True)
                if Path(p).is_file()
            )

    unique = {
        str(path.resolve()).casefold(): path
        for path in files
    }

    return sorted(unique.values(), key=lambda p: str(p).casefold())


def main():
    parser = argparse.ArgumentParser(
        description="Clean Tamil corpora, including Wikipedia <doc> dumps."
    )

    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-words", type=int, default=15)
    parser.add_argument("--min-tamil-ratio", type=float, default=0.50)
    parser.add_argument("--max-latin-ratio", type=float, default=0.15)
    parser.add_argument("--max-digit-ratio", type=float, default=0.25)

    parser.add_argument("--remove-english", action="store_true")
    parser.add_argument("--keep-symbols", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)

    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        parser.error("Train, validation and test ratios must sum to 1.")

    if args.min_chars < 1 or args.min_words < 1:
        parser.error("Minimum character and word counts must be positive.")

    if not 0 <= args.min_tamil_ratio <= 1:
        parser.error("--min-tamil-ratio must be between 0 and 1.")

    if not 0 <= args.max_digit_ratio <= 1:
        parser.error("--max-digit-ratio must be between 0 and 1.")

    input_files = expand_inputs(args.input)
    if not input_files:
        parser.error("No input files found.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_resolved = output_dir.resolve()

    input_files = [
        p for p in input_files
        if output_resolved not in p.resolve().parents
        and p.resolve() != output_resolved
    ]

    if not input_files:
        parser.error("No input files remain after excluding output directory.")

    seen = set()
    records = []
    rejection_counts = Counter()
    per_file = {}

    total_records_read = 0
    duplicate_count = 0

    print("Input files:")
    for path in input_files:
        mode = "Wikipedia articles" if is_wikipedia_dump(path) else "plain-text records"
        print(f"  - {path} [{mode}]")

    print("\nCleaning and combining datasets...\n")

    for path in input_files:
        file_read = 0
        file_kept = 0
        file_duplicates = 0

        for raw_record in iter_records(path):
            total_records_read += 1
            file_read += 1

            text, reason = clean_record(
                raw_record,
                remove_english=args.remove_english,
                remove_symbols=not args.keep_symbols,
            )

            if text is None:
                rejection_counts[reason or "empty"] += 1
                continue

            good, reason, _ = quality_check(
                text,
                min_chars=args.min_chars,
                min_words=args.min_words,
                min_tamil_ratio=args.min_tamil_ratio,
                max_digit_ratio=args.max_digit_ratio,
                max_latin_ratio=args.max_latin_ratio,
            )

            if not good:
                rejection_counts[reason or "quality"] += 1
                continue

            dedup_key = normalize_text(text).casefold()

            if dedup_key in seen:
                duplicate_count += 1
                file_duplicates += 1
                continue

            seen.add(dedup_key)
            records.append(text)
            file_kept += 1

        per_file[str(path)] = {
            "records_read": file_read,
            "unique_records_kept": file_kept,
            "duplicates": file_duplicates,
            "format": (
                "wikipedia_dump"
                if is_wikipedia_dump(path)
                else "plain_text"
            ),
        }

    random.Random(args.seed).shuffle(records)

    total_kept = len(records)
    train_end = int(total_kept * args.train_ratio)
    val_end = train_end + int(total_kept * args.val_ratio)

    train_records = records[:train_end]
    val_records = records[train_end:val_end]
    test_records = records[val_end:]

    outputs = {
        "cleaned.txt": records,
        "train.txt": train_records,
        "validation.txt": val_records,
        "test.txt": test_records,
    }

    for filename, items in outputs.items():
        with (output_dir / filename).open(
            "w", encoding="utf-8", newline="\n"
        ) as f:
            for item in items:
                f.write(item + "\n")

    word_counts = Counter()
    for record in records:
        word_counts.update(record.split())

    source_info = []
    for path in input_files:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)

        source_info.append({
            "path": str(path),
            "sha256": digest.hexdigest(),
        })

    rejected_total = sum(rejection_counts.values())

    report = {
        "sources": source_info,
        "per_file": per_file,
        "records_read": total_records_read,
        "unique_records_kept": total_kept,
        "kept_percentage": round(
            total_kept / total_records_read * 100, 2
        ) if total_records_read else 0,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "options": {
            "remove_english": args.remove_english,
            "remove_symbols": not args.keep_symbols,
            "min_chars": args.min_chars,
            "min_words": args.min_words,
            "min_tamil_ratio": args.min_tamil_ratio,
            "max_digit_ratio": args.max_digit_ratio,
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "rejections": {
            **dict(rejection_counts),
            "duplicate": duplicate_count,
            "rejected_total": rejected_total,
        },
        "top_words": word_counts.most_common(50),
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("========== DATASET REPORT ==========")
    print(f"Input files       : {len(input_files):,}")
    print(f"Records read      : {total_records_read:,}")
    print(f"Unique records    : {total_kept:,}")
    print(f"Kept percentage   : {report['kept_percentage']}%")
    print(f"Duplicates        : {duplicate_count:,}")
    print(f"Rejected records  : {rejected_total:,}")

    print("\nRejection breakdown:")
    for reason, count in sorted(rejection_counts.items()):
        print(f"  - {reason:<22}: {count:,}")

    print("\nSplits saved:")
    print(f"  Train      : {len(train_records):,} -> {output_dir / 'train.txt'}")
    print(f"  Validation : {len(val_records):,} -> {output_dir / 'validation.txt'}")
    print(f"  Test       : {len(test_records):,} -> {output_dir / 'test.txt'}")
    print(f"\nReport      : {output_dir / 'report.json'}")
    print("====================================")


if __name__ == "__main__":
    main()
