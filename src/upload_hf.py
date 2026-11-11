"""
Kural-LLM Hugging Face Model Uploader
====================================
Uploads model checkpoints and tokenizer directly to Hugging Face Model Hub.

Usage:
  python -m src.upload_hf --repo-id <username>/<model-name> --token <hf_write_token> [--checkpoint <path_to_model>]
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login


def upload_to_hf(repo_id: str, token: str, checkpoint_path: str, tokenizer_path: str = "tokenizer.json"):
    print("=" * 65)
    print(f"🚀 KURAL-LLM: UPLOADING TO HUGGING FACE ({repo_id})")
    print("=" * 65)

    # 1. Login / Verify Token
    login(token=token, add_to_git_credential=False)
    api = HfApi()

    # Create repo if it doesn't already exist
    try:
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"📁 Repository verified/ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ℹ️ Repo check notice: {e}")

    # 2. Upload Tokenizer
    tok_file = Path(tokenizer_path)
    if tok_file.exists():
        print(f"⏳ Uploading tokenizer: {tok_file} ({tok_file.stat().st_size / 1024:.1f} KB)...")
        api.upload_file(
            path_or_fileobj=str(tok_file),
            path_in_repo="tokenizer.json",
            repo_id=repo_id,
            token=token,
        )
        print("✅ Tokenizer uploaded successfully!")
    else:
        print(f"⚠️ Warning: Tokenizer file '{tok_file}' not found, skipping.")

    # 3. Upload Model Checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        # Fallback search if path doesn't exist directly
        fallbacks = [
            Path("checkpoints_v4_1_best.pt"),
            Path("checkpoints_v3_stage4/last.pt"),
            Path("checkpoints_v4_warmstart.pt"),
        ]
        for fb in fallbacks:
            if fb.exists():
                ckpt_file = fb
                break

    if ckpt_file.exists():
        size_mb = ckpt_file.stat().st_size / (1024 * 1024)
        print(f"⏳ Uploading Model Weights: {ckpt_file} ({size_mb:.2f} MB)...")
        print("   (This might take a couple of minutes depending on your internet upload speed)")
        api.upload_file(
            path_or_fileobj=str(ckpt_file),
            path_in_repo="pytorch_model.pt",
            repo_id=repo_id,
            token=token,
        )
        print("🎉 Model weights uploaded successfully!")
    else:
        print(f"❌ Error: Model checkpoint file '{checkpoint_path}' not found!")
        return

    # 4. Upload Model Card (README.md)
    readme_path = Path("README.md")
    if readme_path.exists():
        print("⏳ Uploading Model Card (README.md)...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print("✅ Model card uploaded successfully!")

    print("\n" + "=" * 65)
    print("🌟 ALL UPLOADS COMPLETED SUCCESSFULLY!")
    print(f"👉 View your model live at: https://huggingface.co/{repo_id}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Upload Kural-LLM to Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mogesh-developer/Kural-LLM",
        help="Hugging Face repo id, e.g. 'mogesh-developer/Kural-LLM'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face Write Access Token (hf_...)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_v3_stage4/last.pt",
        help="Path to checkpoint file to upload (e.g. checkpoints_v3_stage4/last.pt or checkpoints_v4_1_best.pt)",
    )

    args = parser.parse_args()

    if not args.token:
        args.token = input("🔑 Enter your Hugging Face Write Token (hf_...): ").strip()

    upload_to_hf(
        repo_id=args.repo_id,
        token=args.token,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
