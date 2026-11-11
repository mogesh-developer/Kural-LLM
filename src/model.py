
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 4000
    context_length: int = 512
    n_embd: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, seq_len, channels = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(
            batch, seq_len, channels
        )

        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
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
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or GPTConfig()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.n_embd
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_ln = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        # Share token embedding and output weights
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        batch, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.config.context_length}"
            )

        positions = torch.arange(
            seq_len, device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig()
    model = MiniGPT(config).to(device)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Device:", device)
    print("Parameters:", f"{total_params:,}")
    print("Vocabulary:", config.vocab_size)
    print("Context length:", config.context_length)

    # Smoke test: forward and backward pass
    batch_size = 2
    seq_len = 128

    x = torch.randint(
        0, config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )
    y = torch.randint(
        0, config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    model.train()
    logits, loss = model(x, y)
    loss.backward()

    print("Logits shape:", tuple(logits.shape))
    print("Test loss:", round(loss.item(), 4))
    print("Backward pass: SUCCESS")

    if device == "cuda":
        print(
            "Peak GPU memory:",
            round(
                torch.cuda.max_memory_allocated() / 1024**3,
                3,
            ),
            "GB",
        )

    print("MODEL TEST PASSED")
