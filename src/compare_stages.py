import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from train import GPTConfig, MiniGPT
from generate import generate

PROMPTS = [
    "தமிழ்நாடு ஒரு",
    "இயற்கை என்பது",
    "பண்டைய காலத்தில் மக்கள்",
    "அறிவியல் வளர்ச்சி மனித",
    "கல்வியின் முக்கியத்துவம் என்னவென்றால்",
]

def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
    config = GPTConfig(**cfg_dict)
    model = MiniGPT(config)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    val_loss = checkpoint.get("best_val_loss")
    epoch = checkpoint.get("epoch")
    return model, val_loss, epoch

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer.json")

    s2_path = Path("checkpoints_v3_stage2/best.pt")
    s3_path = Path("checkpoints_v3_stage3/best.pt")

    print(f"Device: {device}")
    
    s2_model, s2_loss, s2_epoch = load_model(s2_path, device)
    print(f"Stage 2 Checkpoint: Epoch {s2_epoch}, Best Val Loss: {s2_loss}")

    s3_model, s3_loss, s3_epoch = (None, None, None)
    if s3_path.exists():
        s3_model, s3_loss, s3_epoch = load_model(s3_path, device)
        print(f"Stage 3 Checkpoint: Epoch {s3_epoch}, Best Val Loss: {s3_loss}")
    
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION: Stage 2 vs Stage 3")
    print("=" * 80)

    for i, p in enumerate(PROMPTS, 1):
        print(f"\n--- PROMPT {i}: \"{p}\" ---")
        
        torch.manual_seed(42)
        out_s2 = generate(s2_model, tokenizer, p, device, max_new_tokens=60, temperature=0.6)
        print(f"[Stage 2 Output]:\n{out_s2}\n")

        if s3_model:
            torch.manual_seed(42)
            out_s3 = generate(s3_model, tokenizer, p, device, max_new_tokens=60, temperature=0.6)
            print(f"[Stage 3 Output]:\n{out_s3}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()

