
from pathlib import Path
import unicodedata

from tokenizers import Tokenizer, models, trainers
from tokenizers import normalizers, pre_tokenizers, decoders


# ==================================================
# CONFIGURATION
# ==================================================

DATA_FILE = Path("data/processed/train.txt")
OUTPUT_FILE = Path("tokenizer.json")

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[PAD]",
]


# ==================================================
# TRAIN TOKENIZER
# ==================================================

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.resolve()}"
        )

    # Read and normalize Tamil Unicode text.
    lines = DATA_FILE.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    lines = [
        unicodedata.normalize("NFC", line.strip())
        for line in lines
        if line.strip()
    ]

    if len(lines) < 2:
        raise ValueError("Not enough non-empty training lines.")

    print("=" * 55)
    print("TAMIL BPE TOKENIZER TRAINING")
    print("=" * 55)
    print("Dataset:", DATA_FILE)
    print("Non-empty lines:", f"{len(lines):,}")
    print("Target vocabulary:", VOCAB_SIZE)

    # BPE model.
    tokenizer = Tokenizer(
        models.BPE(unk_token="[UNK]")
    )

    # Normalize equivalent Unicode representations.
    tokenizer.normalizer = normalizers.NFC()

    # Metaspace represents word boundaries using ▁.
    # This helps preserve spaces when decoding generated text.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    tokenizer.decoder = decoders.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Train directly from the cleaned text file.
    tokenizer.train(
        files=[str(DATA_FILE)],
        trainer=trainer,
    )

    # Save tokenizer in the project root.
    tokenizer.save(str(OUTPUT_FILE))

    print("\nTOKENIZER TRAINED")
    print("-" * 55)
    print("Saved:", OUTPUT_FILE.resolve())
    print("Actual vocabulary:", tokenizer.get_vocab_size())

    for token in SPECIAL_TOKENS:
        print(f"{token}: {tokenizer.token_to_id(token)}")

    # ----------------------------------------------
    # Test Tamil words and sentences.
    # ----------------------------------------------
    test_texts = [
        "தமிழ்",
        "தமிழ்நாடு",
        "தமிழ் மொழி மிகவும் அழகானது.",
        "இந்தியாவின் வரலாறு மிகவும் பழமையானது.",
        "கணினி அறிவியல் மற்றும் செயற்கை நுண்ணறிவு",
    ]

    print("\nTOKENIZATION TEST")
    print("-" * 55)

    for text in test_texts:
        encoded = tokenizer.encode(text)

        print("\nText:", text)
        print("Tokens:", encoded.tokens)
        print("Token count:", len(encoded.ids))
        print("Decoded:", tokenizer.decode(encoded.ids))

    # ----------------------------------------------
    # Basic round-trip test.
    # ----------------------------------------------
    sample = "தமிழ் மொழி மிகவும் அழகானது."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    if decoded != sample:
        print("\nWARNING: Exact round-trip mismatch.")
        print("Original:", repr(sample))
        print("Decoded :", repr(decoded))
        print("Inspect normalization and whitespace behavior.")
    else:
        print("\nRound-trip test: PASSED")

    print("\nNext: verify tokenizer quality before GPT training.")


if __name__ == "__main__":
    main()
