import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.compare_v3_v4 import MiniGPT, GPTConfigV4

def main():
    tok = Tokenizer.from_file('tokenizer.json')
    m = MiniGPT(GPTConfigV4())
    ckpt = torch.load('checkpoints_v4_stage2/best.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['model'])
    m.eval()

    prompts = [
        "தமிழ்நாடு ஒரு",
        "இயற்கை என்பது",
        "தொழில்நுட்பம் நம் அன்றாட வாழ்க்கையில்",
        "பண்டைய காலத்தில் மக்கள்",
        "அறிவியல் வளர்ச்சி மனித"
    ]

    print("=" * 75)
    print("🌟 NEW RECORD BEST MODEL (VAL LOSS: 3.5170 | STEP: 36,390) 🌟")
    print("=" * 75)

    for p in prompts:
        ids = tok.encode(p).ids
        x = torch.tensor([ids], dtype=torch.long)
        for _ in range(55):
            logits = m(x[:, -512:])[:, -1, :] / 0.65
            for tid in torch.unique(x[0]):
                if logits[0, tid] > 0:
                    logits[0, tid] /= 1.2
                else:
                    logits[0, tid] *= 1.2
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        print(f"\n🔹 [Prompt]: \"{p}\"\n[Generated]: {tok.decode(x[0].tolist())}\n" + "-" * 65)

if __name__ == "__main__":
    main()

