"""
PDF to Tamil Dataset Converter (Raw Text & Alpaca SFT Format)
============================================================
Converts Tamil PDF documents into:
1. Clean plain text (for Pre-training)
2. Q&A / Paragraph pairs (Alpaca SFT JSON format for Chatbot Fine-Tuning)

Usage:
  python -m src.pdf_converter --pdf <path_to_pdf_or_folder> [--out-format sft|raw]
"""

import os
import re
import json
import argparse
from pathlib import Path
from pypdf import PdfReader


def clean_tamil_text(text: str) -> str:
    """Cleans up PDF extraction artifacts while preserving Tamil Unicode"""
    # Replace multiple spaces / newlines
    text = re.sub(r'[\r\f\v]', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove standalone non-Tamil special symbols / page numbers
    lines = [l.strip() for l in text.split('\n')]
    cleaned_lines = [l for l in lines if len(l) > 2 and not l.isdigit()]
    return '\n'.join(cleaned_lines)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts raw Tamil text from all pages in a PDF file"""
    print(f"📄 Processing: {pdf_path.name}...")
    reader = PdfReader(str(pdf_path))
    full_text = []

    for page_num, page in enumerate(reader.pages, 1):
        try:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
        except Exception as e:
            print(f"   ⚠️ Warning on page {page_num}: {e}")

    raw = "\n".join(full_text)
    return clean_tamil_text(raw)


def convert_text_to_sft_alpaca(cleaned_text: str, source_title: str) -> list:
    """
    Splits text into meaningful paragraphs and structures them into Alpaca SFT format:
    - Instruction: Explain / Summarize / Comprehend the paragraph
    - Input: The Tamil context
    - Output: The factual content
    """
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if len(p.strip()) > 80]
    sft_data = []

    for p in paragraphs:
        # Create a reading comprehension / informational SFT pair
        first_line = p.split('.')[0] if '.' in p else p[:60]
        instruction = f"பின்வரும் பத்தியைப் படித்து, அதன் முக்கிய கருத்தை விளக்குங்கள்:"
        
        # Or factual instruction based on first sentence
        if "என்பது" in first_line or "ஆகும்" in first_line:
            topic = first_line.split("என்பது")[0] if "என்பது" in first_line else first_line.split("ஆகும்")[0]
            instruction = f"{topic.strip()} பற்றி விளக்குக."
        
        sft_data.append({
            "instruction": instruction,
            "input": f"ஆதாரம்: {source_title}",
            "output": p,
        })

    return sft_data


def process_pdfs(input_path_str: str, out_format: str = "sft", output_dir: str = "data/pdf_converted"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = Path(input_path_str)
    pdf_files = []

    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        pdf_files.append(target_path)
    elif target_path.is_dir():
        pdf_files.extend(list(target_path.glob("**/*.pdf")))
    else:
        print(f"❌ Error: Path '{target_path}' not found!")
        return

    print("=" * 65)
    print(f"📚 TAMIL PDF TO DATASET CONVERTER ({len(pdf_files)} PDFs Found)")
    print(f"🎯 Target Output Format: {out_format.upper()}")
    print("=" * 65)

    all_raw_text = []
    all_sft_data = []

    for pdf in pdf_files:
        text = extract_text_from_pdf(pdf)
        if not text:
            print(f"⚠️ No text could be extracted from {pdf.name} (Might be scanned image/OCR needed)")
            continue

        all_raw_text.append(text)
        if out_format == "sft":
            pairs = convert_text_to_sft_alpaca(text, source_title=pdf.stem)
            all_sft_data.extend(pairs)

    # 1. Save Raw Text
    raw_out = out_dir / "extracted_tamil_corpus.txt"
    with open(raw_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_raw_text))
    print(f"\n✅ Clean Plain Text Saved: {raw_out} ({raw_out.stat().st_size / 1024:.1f} KB)")

    # 2. Save SFT JSON Format
    if out_format == "sft":
        sft_out = out_dir / "converted_alpaca_sft.json"
        with open(sft_out, "w", encoding="utf-8") as f:
            json.dump(all_sft_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Alpaca SFT Dataset Saved: {sft_out} ({len(all_sft_data):,} Q&A Pairs created!)")

    print("\n🎉 CONVERSION COMPLETED!")


def main():
    parser = argparse.ArgumentParser(description="Convert Tamil PDFs into LLM Dataset")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file or folder containing PDFs")
    parser.add_argument("--format", type=str, default="sft", choices=["sft", "raw"], help="Format: sft or raw")
    parser.add_argument("--out-dir", type=str, default="data/pdf_converted", help="Output directory")

    args = parser.parse_args()
    process_pdfs(args.pdf, out_format=args.format, output_dir=args.out_dir)


if __name__ == "__main__":
    main()
