
from pathlib import Path
import random
import sentencepiece as spm

SOURCE = Path("data/processed/train.txt")
OUT = Path("data/processed/tokenizer_eval")
MODEL_DIR = Path("checkpoints/tokenizer_eval")

VOCAB_SIZE = 8000
SEED = 42
VALIDATION_FRACTION = 0.10
CHUNK_CHARS = 6000


def read_records(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def split_records(records):
    if len(records) < 100:
        raise ValueError("Need at least 100 records for this evaluation.")

    # Split at record boundaries, not within documents.
    shuffled = records.copy()
    random.Random(SEED).shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * VALIDATION_FRACTION))
    validation = shuffled[:n_val]
    training = shuffled[n_val:]

    return training, validation


def write_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record + "\n")


def chunk_record(text, max_chars=CHUNK_CHARS):
    """Chunk long records at whitespace boundaries."""
    words = text.split()
    chunk = ""

    for word in words:
        candidate = f"{chunk} {word}".strip()

        if len(candidate) > max_chars and chunk:
            yield chunk
            chunk = word
        else:
            chunk = candidate

    if chunk:
        yield chunk


def write_training_chunks(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            for chunk in chunk_record(record):
                f.write(chunk + "\n")
                count += 1

    print(f"Tokenizer training chunks: {count:,}")


def train_tokenizer(corpus, prefix, model_type):
    prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=VOCAB_SIZE,
        model_type=model_type,
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        split_digits=False,
        byte_fallback=True,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        max_sentence_length=16384,
        hard_vocab_limit=False,
        num_threads=4,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )

    return Path(f"{prefix}.model")


def tamil_char_count(text):
    return sum("\u0B80" <= c <= "\u0BFF" for c in text)


def evaluate(model_path, records, label):
    sp = spm.SentencePieceProcessor(
        model_file=str(model_path)
    )

    total_chars = 0
    total_tokens = 0
    total_words = 0
    exact_chunks = 0
    total_chunks = 0
    unknown_ids = 0
    tamil_single_pieces = 0
    total_pieces = 0

    tamil_chars = 0
    all_chars = 0

    for record in records:
        all_chars += len(record)
        tamil_chars += tamil_char_count(record)

        for chunk in chunk_record(record):
            ids = sp.encode(chunk, out_type=int)
            pieces = sp.encode(chunk, out_type=str)
            decoded = sp.decode(ids)

            total_chars += len(chunk)
            total_tokens += len(ids)
            total_words += len(chunk.split())
            unknown_ids += ids.count(sp.unk_id())
            total_pieces += len(pieces)
            exact_chunks += int(decoded == chunk)
            total_chunks += 1

            for piece in pieces:
                visible = piece.removeprefix("▁")
                is_byte = visible.startswith("<0x")
                is_single_tamil = (
                    len(visible) == 1
                    and "\u0B80" <= visible <= "\u0BFF"
                )

                if not is_byte and is_single_tamil:
                    tamil_single_pieces += 1

    result = {
        "label": label,
        "vocab": sp.get_piece_size(),
        "tokens": total_tokens,
        "chars_per_token": total_chars / max(total_tokens, 1),
        "tokens_per_word": total_tokens / max(total_words, 1),
        "single_tamil_pct": (
            100 * tamil_single_pieces / max(total_pieces, 1)
        ),
        "exact_pct": 100 * exact_chunks / max(total_chunks, 1),
        "unknown": unknown_ids,
        "tamil_ratio": tamil_chars / max(all_chars, 1),
    }

    print(f"\n--- {label} ---")
    print(f"Vocabulary: {result['vocab']:,}")
    print(f"Characters/token: {result['chars_per_token']:.3f}")
    print(f"Tokens/word: {result['tokens_per_word']:.3f}")
    print(f"Single-char Tamil pieces: {result['single_tamil_pct']:.2f}%")
    print(f"Exact reconstruction: {result['exact_pct']:.2f}%")
    print(f"Unknown IDs: {result['unknown']}")
    print(f"Tamil character ratio: {result['tamil_ratio']:.2%}")

    return result


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    records = read_records(SOURCE)
    training, validation = split_records(records)

    # Persist the split so it can be reproduced and reused.
    write_records(training, OUT / "train.txt")
    write_records(validation, OUT / "validation.txt")

    print(f"Original records: {len(records):,}")
    print(f"Training records: {len(training):,}")
    print(f"Validation records: {len(validation):,}")
    print(f"Longest original record: {max(map(len, records)):,} chars")

    corpus = OUT / "tokenizer_training_chunks.txt"
    write_training_chunks(training, corpus)

    results = []

    for model_type in ("unigram", "bpe"):
        prefix = MODEL_DIR / f"tamil_{model_type}_8k"
        model_path = train_tokenizer(corpus, prefix, model_type)

        # Evaluate on records that were excluded from tokenizer training.
        results.append(
            evaluate(model_path, validation, model_type.upper())
        )

    print("\n=== HELD-OUT COMPARISON ===")
    print(
        f"{'Model':<10} {'Vocab':>7} {'Chars/tok':>10} "
        f"{'Tok/word':>10} {'1-char Tamil':>13} {'Exact %':>9}"
    )

    for r in results:
        print(
            f"{r['label']:<10} {r['vocab']:>7,} "
            f"{r['chars_per_token']:>10.3f} "
            f"{r['tokens_per_word']:>10.3f} "
            f"{r['single_tamil_pct']:>12.2f}% "
            f"{r['exact_pct']:>8.2f}%"
        )


if __name__ == "__main__":
    main()
