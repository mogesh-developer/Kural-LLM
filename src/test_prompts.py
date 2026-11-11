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
    "வானிலை மாற்றம் காரணமாக",
    "தொழில்நுட்பம் நம் அன்றாட வாழ்க்கையில்",
]

def load_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
    config = GPTConfig(**cfg_dict)
    model = MiniGPT(config)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("best_val_loss", "?")
    global_step = checkpoint.get("global_step", "?")
    return model, epoch, val_loss, global_step

def test_checkpoint(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    
    tokenizer_path = Path("tokenizer.json")
    if not tokenizer_path.exists():
        print("tokenizer.json not found!")
        return
        
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model, epoch, val_loss, step = load_checkpoint(ckpt_path, device)
    
    print(f"Loaded Epoch: {epoch} | Best Val Loss: {val_loss} | Global Updates: {step}")
    print("=" * 70)
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[Prompt {i}]: \"{prompt}\"")
        try:
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=80,
                min_new_tokens=20,
                temperature=0.6,
                top_k=30,
                top_p=0.85,
                repetition_penalty=1.25,
            )
            print(f"[Generated Text]:\n{output}")
        except Exception as e:
            print(f"[Error Generating]: {e}")
        print("-" * 70)

def main():
    s4_last = Path("checkpoints_v3_stage4/last.pt")
    s4_best = Path("checkpoints_v3_stage4/best.pt")
    s3_best = Path("checkpoints_v3_stage3/best.pt")
    
    target_ckpt = None
    if s4_best.exists():
        target_ckpt = s4_best
    elif s4_last.exists():
        target_ckpt = s4_last
    elif s3_best.exists():
        target_ckpt = s3_best
        print("Notice: Stage 4 not found, falling back to Stage 3.")
        
    if target_ckpt:
        test_checkpoint(target_ckpt)
    else:
        print("No valid checkpoints found.")

if __name__ == "__main__":
    main()

