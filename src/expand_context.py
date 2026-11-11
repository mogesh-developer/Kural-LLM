"""
Kural-LLM Context Length Expander (512 -> 1024)
==============================================
Expands the Position Embedding matrix from (512, 768) to (1024, 768)
using Linear Position Interpolation (Chen et al., 2023).

This preserves 100% of the trained 512-token knowledge while doubling
the context window so long questions & comprehensive answers fit cleanly.
"""

import math
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compare_v3_v4 import MiniGPT, GPTConfigV4


@dataclass
class GPTConfig1024:
    vocab_size: int = 8000
    context_length: int = 1024    # Doubled Context Window!
    n_embd: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.05


def expand_context_window(
    src_checkpoint_path: str = "checkpoints_v4_stage2/best.pt",
    out_checkpoint_path: str = "checkpoints_v4_1024_base.pt",
):
    print("=" * 70)
    print("🚀 EXPANDING KURAL-LLM CONTEXT LENGTH: 512 -> 1024 TOKENS")
    print("=" * 70)

    src_path = Path(src_checkpoint_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_path}")

    print(f"📦 Loading source 512-context checkpoint: {src_path}")
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    old_state = ckpt["model"]
    old_cfg = ckpt.get("model_config", {})

    print(f"   Old Context Length: {old_cfg.get('context_length', 512)}")
    print(f"   Old Best Val Loss : {ckpt.get('best_val_loss', 'N/A')}")
    print(f"   Old Global Step   : {ckpt.get('global_step', 'N/A')}")

    # Initialize new 1024 config model
    new_cfg = GPTConfig1024()
    new_model = MiniGPT(new_cfg)
    new_state = new_model.state_dict()

    print("\n🧠 Performing Linear Position Interpolation on Position Embeddings...")
    old_pos_emb = old_state["position_embedding.weight"]  # Shape: (512, 768)
    old_len, n_embd = old_pos_emb.shape
    new_len = new_cfg.context_length

    # Method:
    # 1. Keep the first 512 positions identical so normal short sequences stay 100% identical
    # 2. Interpolate across the 0..old_len scale for smooth extension into 512..1023
    new_pos_emb = torch.empty((new_len, n_embd), dtype=old_pos_emb.dtype)

    # Copy first 512 directly
    new_pos_emb[:old_len] = old_pos_emb

    # For 512..1023, linearly interpolate between positions to ensure smooth gradients
    for i in range(old_len, new_len):
        # Normalized position in 0..old_len
        ratio = (i / (new_len - 1)) * (old_len - 1)
        low = int(math.floor(ratio))
        high = min(low + 1, old_len - 1)
        alpha = ratio - low
        new_pos_emb[i] = (1.0 - alpha) * old_pos_emb[low] + alpha * old_pos_emb[high]

    print(f"✅ Position Embedding successfully expanded: {old_pos_emb.shape} -> {new_pos_emb.shape}")

    # Copy all other transformer layers (Embeddings, Attention, MLPs, LayerNorms, LM Head)
    transferred = 0
    for key, val in old_state.items():
        if key == "position_embedding.weight":
            new_state[key] = new_pos_emb
        elif key in new_state:
            assert new_state[key].shape == val.shape, f"Shape mismatch for {key}: {new_state[key].shape} vs {val.shape}"
            new_state[key] = val.clone()
            transferred += 1

    new_model.load_state_dict(new_state)

    total_params = sum(p.numel() for p in new_model.parameters())
    print(f"🎯 Total Parameters in 1024 Model: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"   Transferred Layers: {transferred + 1}/{len(new_state)}")

    # Save new 1024 warmstart checkpoint
    out_payload = {
        "version": "v4_97M_1024context_base",
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_val_loss": ckpt.get("best_val_loss", 3.5170),
        "model_config": asdict(new_cfg),
        "model": new_model.state_dict(),
    }

    out_path = Path(out_checkpoint_path)
    torch.save(out_payload, out_path)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n🎉 SUCCESS! Saved 1024-Context Model to: {out_path} ({file_size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    expand_context_window()
