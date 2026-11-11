import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tokenizers import Tokenizer

# ----------------- V3 (54.68M) Config -----------------
@dataclass
class GPTConfigV3:
    vocab_size: int = 8000
    context_length: int = 512
    n_embd: int = 640
    n_heads: int = 10
    n_layers: int = 10
    dropout: float = 0.1

# ----------------- V4 (91.59M) Config -----------------
@dataclass
class GPTConfigV4:
    vocab_size: int = 8000
    context_length: int = 512
    n_embd: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.05

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_embedding(x) + self.position_embedding(pos)
        h = self.dropout(h)
        for b in self.blocks:
            h = b(h)
        return self.lm_head(self.final_ln(h))

def main():
    tok = Tokenizer.from_file("tokenizer.json")

    # Load Model V3 (54.68M)
    print("Loading V3 (54.68M)...")
    m_v3 = MiniGPT(GPTConfigV3())
    ckpt_v3 = torch.load("checkpoints_v3_stage4/last.pt", map_location="cpu", weights_only=False)
    m_v3.load_state_dict(ckpt_v3["model"])
    m_v3.eval()

    # Load Model V4 (91.59M)
    print("Loading V4 (91.59M)...")
    m_v4 = MiniGPT(GPTConfigV4())
    ckpt_v4 = torch.load("last_model.pt", map_location="cpu", weights_only=False)
    m_v4.load_state_dict(ckpt_v4["model"])
    m_v4.eval()

    prompts = [
        "தமிழ்நாடு ஒரு",
        "இயற்கை என்பது",
        "தொழில்நுட்பம் நம் அன்றாட வாழ்க்கையில்",
        "பண்டைய காலத்தில் மக்கள்",
        "அறிவியல் வளர்ச்சி மனித",
    ]

    def generate(m, prompt, steps=50, temp=0.65):
        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        for _ in range(steps):
            logits = m(x[:, -512:])[:, -1, :] / temp
            for tid in torch.unique(x[0]):
                if logits[0, tid] > 0:
                    logits[0, tid] /= 1.2
                else:
                    logits[0, tid] *= 1.2
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            x = torch.cat([x, nxt], dim=1)
        return tok.decode(x[0].tolist())

    print("\n" + "=" * 80)
    print("⚔️ KURAL-LLM COMPARISON: V3 (54.68M) vs V4 (91.59M) ⚔️")
    print("=" * 80)

    for p in prompts:
        print(f"\n🔹 [PROMPT]: \"{p}\"")
        res_v3 = generate(m_v3, p)
        res_v4 = generate(m_v4, p)
        print(f"   [V3 - 54.68M]:\n   {res_v3}\n")
        print(f"   [V4 - 91.59M]:\n   {res_v4}")
        print("-" * 80)

if __name__ == "__main__":
    main()

