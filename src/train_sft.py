"""
Kural-Chat-98M: Supervised Fine-Tuning (SFT) Pipeline
=====================================================
Fine-tunes the 1024-context Kural-LLM base model on the 52K Tamil Alpaca dataset
using Instruction Masking (Loss calculated ONLY on the Assistant Response).

Hardware Optimized for NVIDIA RTX 4050 (6GB VRAM) with Native BF16 & FlashAttention.
"""

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import pyarrow.parquet as pq

from src.compare_v3_v4 import MiniGPT, GPTConfigV4


# ============================================================
# 1. SFT CONFIGURATION
# ============================================================

@dataclass
class SFTGPTConfig:
    vocab_size: int = 8000
    context_length: int = 1024       # Full 1024 Context!
    n_embd: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.05            # Low dropout for clean factual recall


@dataclass
class SFTTrainConfig:
    data_file: str = "data/sft/train-00000-of-00001-e96a6e4e7f400cce.parquet"
    tokenizer_file: str = "tokenizer.json"
    base_checkpoint: str = "checkpoints_v4_1024_base.pt"
    output_dir: str = "checkpoints_kural_chat"

    epochs: int = 3                  # 3 Epochs is optimal for SFT (prevents catastrophic forgetting)
    batch_size: int = 4              # Fits comfortably in 6GB VRAM at 1024 context
    grad_accum_steps: int = 16       # Effective batch size = 4 * 16 = 64

    # SFT Fine-tuning Learning Rate (Gentle adaptation)
    learning_rate: float = 3.0e-5
    min_learning_rate: float = 3.0e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 40

    val_split: float = 0.04          # ~2,000 validation samples
    num_workers: int = 0             # Safe for Windows
    log_every: int = 15


# ============================================================
# 2. SFT DATASET WITH INSTRUCTION MASKING
# ============================================================

class SFTDataset(Dataset):
    """
    Constructs prompt pairs:
    ### Instruction:
    {instruction}

    ### Input: (if any)
    {input}

    ### Response:
    {output}

    Masks out the instruction tokens with label -100 so CrossEntropyLoss
    is computed ONLY on the assistant's response!
    """
    def __init__(self, records, tokenizer, max_length=1024):
        self.samples = []
        eos_id = tokenizer.token_to_id("[EOS]")
        pad_id = tokenizer.token_to_id("[PAD]") or 0

        for r in records:
            inst = r.get("instruction", "").strip()
            inp = r.get("input", "").strip() if r.get("input") else ""
            out = r.get("output", "").strip()

            if not inst or not out:
                continue

            if inp:
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

            prompt_ids = tokenizer.encode(prompt).ids
            answer_ids = tokenizer.encode(out).ids + [eos_id]

            input_ids = prompt_ids + answer_ids

            # Skip sequences that exceed context window
            if len(input_ids) > max_length:
                # Truncate answer gently while preserving prompt
                remaining = max_length - len(prompt_ids)
                if remaining < 15:
                    continue
                answer_ids = answer_ids[:remaining - 1] + [eos_id]
                input_ids = prompt_ids + answer_ids

            # Labels: -100 for prompt tokens, actual IDs for response tokens
            labels = ([-100] * len(prompt_ids)) + answer_ids

            self.samples.append((input_ids, labels))

        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate_fn(batch):
    """Pad batch dynamically to longest sequence in batch"""
    max_len = max(len(item[0]) for item in batch)
    input_ids_batch = []
    labels_batch = []

    for input_ids, labels in batch:
        pad_len = max_len - len(input_ids)
        input_ids_batch.append(input_ids + [0] * pad_len)
        labels_batch.append(labels + [-100] * pad_len)

    return (
        torch.tensor(input_ids_batch, dtype=torch.long),
        torch.tensor(labels_batch, dtype=torch.long),
    )


def load_sft_data(cfg, gpt_cfg):
    print("\n" + "=" * 65)
    print("📦 LOADING TAMIL ALPACA SFT PARQUET DATASET...")
    print("=" * 65)

    tok = Tokenizer.from_file(cfg.tokenizer_file)

    table = pq.read_table(cfg.data_file)
    df_rows = []
    for i in range(table.num_rows):
        df_rows.append({
            "instruction": table["instruction"][i].as_py(),
            "input": table["input"][i].as_py(),
            "output": table["output"][i].as_py(),
        })

    random.seed(42)
    random.shuffle(df_rows)

    val_size = int(len(df_rows) * cfg.val_split)
    train_rows = df_rows[val_size:]
    val_rows = df_rows[:val_size]

    print(f"Total Raw Rows     : {len(df_rows):,}")
    print(f"Training Rows      : {len(train_rows):,}")
    print(f"Validation Rows    : {len(val_rows):,}")

    train_ds = SFTDataset(train_rows, tok, max_length=gpt_cfg.context_length)
    val_ds = SFTDataset(val_rows, tok, max_length=gpt_cfg.context_length)

    print(f"Usable Train Samples: {len(train_ds):,}")
    print(f"Usable Val Samples  : {len(val_ds):,}")
    return tok, train_ds, val_ds


# ============================================================
# 3. CHAT GENERATION TEST
# ============================================================

def test_chat(model, tokenizer, device):
    test_questions = [
        "தமிழ்நாட்டின் தலைநகரம் எது?",
        "ஆரோக்கியமாக இருக்க என்ன செய்ய வேண்டும்?",
        "திருக்குறள் பற்றி சுருக்கமாக கூறுங்கள்.",
        "கணினியின் பயன்கள் என்ன?",
    ]
    model.eval()
    eos_id = tokenizer.token_to_id("[EOS]")

    print("\n" + "=" * 70)
    print("💬 KURAL-CHAT 98M CONVERSATION TEST 💬")
    print("=" * 70)

    with torch.no_grad():
        for q in test_questions:
            prompt = f"### Instruction:\n{q}\n\n### Response:\n"
            ids = tokenizer.encode(prompt).ids
            x = torch.tensor([ids], dtype=torch.long, device=device)

            for _ in range(90):
                x_cond = x[:, -1024:]
                logits = model(x_cond)[:, -1, :] / 0.65
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                if next_tok.item() == eos_id:
                    break
                x = torch.cat([x, next_tok], dim=1)

            full_text = tokenizer.decode(x[0].tolist())
            response_part = full_text.split("### Response:\n")[-1].strip()
            print(f"\n👤 [User]: {q}")
            print(f"🤖 [Kural-Chat]: {response_part}\n" + "-" * 55)

    model.train()


# ============================================================
# 4. SFT TRAINING LOOP
# ============================================================

def main():
    cfg = SFTTrainConfig()
    gpt_cfg = SFTGPTConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    tokenizer, train_ds, val_ds = load_sft_data(cfg, gpt_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sft_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=sft_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 1. Initialize 1024 Model & Load Weights
    print(f"\n🧠 Initializing 98.13M Architecture (1024 Context)...")
    model = MiniGPT(gpt_cfg).to(device)

    base_ckpt_path = Path(cfg.base_checkpoint)
    if base_ckpt_path.exists():
        print(f"📥 Loading Pretrained Base Checkpoint: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print("✅ Base Model weights loaded cleanly!")
    else:
        raise FileNotFoundError(f"Base checkpoint {base_ckpt_path} not found!")

    # 2. Optimizer & Cosine Schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.epochs
    global_step = 0
    best_val_loss = float("inf")

    def get_lr(step):
        if step < cfg.warmup_steps:
            return cfg.learning_rate * (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"🔥 Mixed Precision: {amp_dtype}")
    print(f"🚀 SFT Training across {cfg.epochs} Epochs | Effective Batch: {cfg.batch_size * cfg.grad_accum_steps}")
    print("=" * 65)

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, targets) in enumerate(train_loader):
            x, targets = x.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                # Next-token prediction requires shifting logits and targets!
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )
                raw_loss_val = loss.item()
                loss = loss / cfg.grad_accum_steps

            # Safety Guard: Skip corrupted / exploding gradients
            if not math.isfinite(raw_loss_val) or raw_loss_val > 15.0:
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            total_train_loss += raw_loss_val

            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Verify gradients are finite before optimizer step
                valid_grads = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        valid_grads = False
                        break

                if valid_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    lr = get_lr(global_step)
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                    optimizer.step()
                    global_step += 1
                else:
                    print(f"⚠️ Skipped step at batch {batch_idx + 1} due to non-finite gradient.")

                optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) % (cfg.log_every * cfg.grad_accum_steps) == 0:
                avg_loss = total_train_loss / (batch_idx + 1)
                print(f"Epoch {epoch + 1:02d}/{cfg.epochs} | Batch {batch_idx + 1:04d}/{len(train_loader)} | SFT Loss {avg_loss:.4f} | LR {lr:.2e}")

        # Validation Step
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    v_logits = model(vx)
                    shift_v_logits = v_logits[:, :-1, :].contiguous()
                    shift_vy = vy[:, 1:].contiguous()
                    v_loss = F.cross_entropy(
                        shift_v_logits.view(-1, shift_v_logits.size(-1)).float(),
                        shift_vy.view(-1),
                        ignore_index=-100,
                    )
                if torch.isfinite(v_loss):
                    val_loss += v_loss.item()
                    val_count += 1
                if val_count >= 50:
                    break

        avg_val_loss = val_loss / max(1, val_count)
        print(f"\n--> Epoch {epoch + 1:02d} Complete | Val SFT Loss: {avg_val_loss:.4f}")

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        payload = {
            "version": "kural_chat_98M_1024",
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model_config": asdict(gpt_cfg),
            "model": model.state_dict(),
        }

        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)
            print(f"⭐ Saved New Best Chat Model! (SFT Loss: {best_val_loss:.4f})")

        print("-" * 65)

    print("\n🎉 SFT CHAT TRAINING FINISHED SUCCESSFULLY!")
    test_chat(model, tokenizer, device)


if __name__ == "__main__":
    main()

