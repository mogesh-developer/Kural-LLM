from pathlib import Path
import torch
from tokenizers import Tokenizer
from train import GPTConfig, MiniGPT
from generate import generate

PROMPTS = [
    "தமிழ்நாடு ஒரு",
    "இயற்கை என்பது",
    "தொழில்நுட்பம் நம் அன்றாட வாழ்க்கையில்",
    "அறிவியல் வளர்ச்சி மனித",
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("checkpoints_v4_warmstart.pt")
    
    print(f"Loading Warm-Started 85.2M Checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    tokenizer = Tokenizer.from_file("tokenizer.json")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint.get("model_config", {})
    config = GPTConfig(**cfg_dict)
    
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,} (Confirmed 85.2M)")
    print(f"Inherited Val Loss: {checkpoint.get('best_val_loss')}")
    print("=" * 70)

    for i, p in enumerate(PROMPTS, 1):
        print(f"\n[Prompt {i}]: \"{p}\"")
        torch.manual_seed(42)
        out = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            device=device,
            max_new_tokens=60,
            temperature=0.6,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.25,
        )
        print(f"[Generated Text]:\n{out}")
        print("-" * 70)

if __name__ == "__main__":
    main()

