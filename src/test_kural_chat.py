"""
Kural-Chat 98M (1024 Context) Comprehensive Evaluation Test
===========================================================
Tests the fine-tuned SFT Chat model with:
- Repetition Penalty (1.3)
- Top-p (Nucleus) sampling
- Diverse practical prompts (Geography, Science, Health, General QA)
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.compare_v3_v4 import MiniGPT
from src.train_sft import SFTGPTConfig


def generate_response(model, tokenizer, question, device, max_tokens=90, temperature=0.6, rep_penalty=1.35, top_p=0.85):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x[:, -1024:]
            logits = model(x_cond)[:, -1, :] / temperature

            # Repetition penalty
            for tid in torch.unique(x[0]):
                if logits[0, tid] > 0:
                    logits[0, tid] /= rep_penalty
                else:
                    logits[0, tid] *= rep_penalty

            # Top-p (nucleus) filtering
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            next_tok = torch.multinomial(probs, 1)
            if next_tok.item() == eos_id:
                break
            x = torch.cat([x, next_tok], dim=1)

    full_text = tokenizer.decode(x[0].tolist())
    response = full_text.split("### Response:\n")[-1].strip()
    return response


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🤖 KURAL-CHAT 98M (POST-SFT) LIVE CONVERSATION TEST")
    print(f"⚡ Device: {device} | Checkpoint: checkpoints_kural_chat/best.pt")
    print("=" * 70)

    tok = Tokenizer.from_file("tokenizer.json")
    cfg = SFTGPTConfig()
    model = MiniGPT(cfg).to(device)

    ckpt = torch.load("checkpoints_kural_chat/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"✅ SFT Checkpoint Loaded cleanly! (Val Loss: {ckpt.get('best_val_loss', 'N/A'):.4f} | Epoch: {ckpt.get('epoch')})")
    print("-" * 70)

    test_queries = [
        "தமிழ்நாட்டின் தலைநகரம் எது?",
        "ஆரோக்கியமாக வாழ சில குறிப்புகளை கூறுங்கள்.",
        "கணினியின் முக்கிய பயன்கள் என்ன?",
        "சூரிய குடும்பத்தில் உள்ள கோள்கள் பற்றி கூறுங்கள்.",
        "தண்ணீர் குடிப்பதால் உடலுக்கு கிடைக்கும் நன்மைகள் என்ன?",
        "செயற்கை நுண்ணறிவு (AI) என்றால் என்ன?",
    ]

    for q in test_queries:
        ans = generate_response(model, tok, q, device)
        print(f"\n👤 [User]: {q}")
        print(f"🤖 [Kural-Chat]:\n{ans}")
        print("-" * 70)


if __name__ == "__main__":
    main()

