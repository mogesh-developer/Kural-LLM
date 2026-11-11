
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int = 0
    context_length: int = 512
    n_embd: int = 640
    n_heads: int = 10
    n_layers: int = 10
    dropout: float = 0.1


@dataclass
class TrainConfig:
    train_file: str = "data/processed/train.txt"
    val_file: str = "data/processed/validation.txt"
    tokenizer_file: str = "tokenizer.json"

    # Load clean v3 Stage 3 best checkpoint to continue training for Stage 4
    resume_from: str = "checkpoints_v3_stage3/best.pt"

    # Output directory for v3 Stage 4 (preserving Stage 2 and Stage 3 checkpoints safely)
    output_dir: str = "checkpoints_v3_stage4"

    # Automatically continue v3 Stage 4 if interrupted
    resume_v3_stage: bool = True

    seed: int = 42
    epochs: int = 10

    # Hardware-optimized for RTX 4050 Laptop (6GB VRAM)
    # Effective batch size = batch_size * grad_accum_steps = 16 * 4 = 64
    batch_size: int = 16
    grad_accum_steps: int = 4

    # Stage 4 fine-tuning learning rate (gentle polishing schedule)
    learning_rate: float = 1.5e-5
    min_learning_rate: float = 1.5e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 50

    eval_batches: int = 50
    num_workers: int = 0
    log_every: int = 25


# ============================================================
# 2. TRANSFORMER MODEL
# Keep architecture identical to Stage 1.
# ============================================================

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
            q,
            k,
            v,
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
    def __init__(self, config):
        super().__init__()
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

        self.apply(self._init_weights)

        # Tie token embedding and output weights.
        self.lm_head.weight = self.token_embedding.weight

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
            raise ValueError("Sequence exceeds context length.")

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
            # Cast logits to float32 before cross_entropy for maximum numerical stability
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
            )

        return logits, loss


# ============================================================
# 3. DATASET
# ============================================================

class PackedDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.chunk_size = context_length + 1

        if len(token_ids) < self.chunk_size:
            raise ValueError(
                f"Only {len(token_ids)} tokens; "
                f"need at least {self.chunk_size}."
            )

        usable = (
            len(token_ids) // self.chunk_size
        ) * self.chunk_size

        self.tokens = torch.tensor(
            token_ids[:usable], dtype=torch.long
        )

    def __len__(self):
        return len(self.tokens) // self.chunk_size

    def __getitem__(self, index):
        start = index * self.chunk_size
        chunk = self.tokens[start:start + self.chunk_size]
        return chunk[:-1], chunk[1:]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_records(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        records = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def encode_records(records, tokenizer, eos_id):
    token_ids = []

    for record in records:
        ids = tokenizer.encode(record).ids
        token_ids.extend(ids)
        token_ids.append(eos_id)

    return token_ids


def load_data(cfg, gpt_cfg):
    tokenizer_path = Path(cfg.tokenizer_file)

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path.resolve()}"
        )

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    train_records = read_records(cfg.train_file)
    val_records = read_records(cfg.val_file)

    overlap = set(train_records).intersection(val_records)

    if overlap:
        raise ValueError(
            f"{len(overlap)} identical records exist in both "
            "training and validation files."
        )

    eos_id = tokenizer.token_to_id("[EOS]")

    if eos_id is None:
        raise ValueError("Tokenizer has no [EOS] token.")

    train_ids = encode_records(train_records, tokenizer, eos_id)
    val_ids = encode_records(val_records, tokenizer, eos_id)

    gpt_cfg.vocab_size = tokenizer.get_vocab_size()

    if (
        min(train_ids + val_ids) < 0
        or max(train_ids + val_ids) >= gpt_cfg.vocab_size
    ):
        raise ValueError("Token ID is outside vocabulary range.")

    train_ds = PackedDataset(
        train_ids, gpt_cfg.context_length
    )
    val_ds = PackedDataset(
        val_ids, gpt_cfg.context_length
    )

    tokenizer_hash = sha256_file(tokenizer_path)

    dataset_hashes = {
        "train": sha256_file(cfg.train_file),
        "validation": sha256_file(cfg.val_file),
    }

    print("\nSTAGE 2 DATASET")
    print("-" * 55)
    print(f"Training records   : {len(train_records):,}")
    print(f"Validation records : {len(val_records):,}")
    print(f"Vocabulary size    : {gpt_cfg.vocab_size:,}")
    print(f"Training tokens    : {len(train_ids):,}")
    print(f"Validation tokens  : {len(val_ids):,}")
    print(f"Training chunks    : {len(train_ds):,}")
    print(f"Validation chunks  : {len(val_ds):,}")

    return (
        tokenizer,
        train_ds,
        val_ds,
        tokenizer_hash,
        dataset_hashes,
    )


# ============================================================
# 4. EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss(
    model, loader, device, amp_dtype, use_amp, max_batches
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            _, loss = model(x, y)

        if loss is not None and torch.isfinite(loss) and loss.item() < 20.0:
            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        return float("nan")

    return total_loss / total_batches


# ============================================================
# 5. CHECKPOINTS AND RANDOM-STATE RECOVERY
# ============================================================

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    return state


def restore_rng_state(state):
    if not state:
        return

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())

    if (
        torch.cuda.is_available()
        and state.get("cuda") is not None
    ):
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def build_checkpoint(
    model,
    optimizer,
    scaler,
    gpt_cfg,
    cfg,
    epoch,
    global_step,
    stage_step,
    best_val_loss,
    tokenizer_hash,
    dataset_hashes,
    schedule_total_steps,
    schedule_warmup_steps,
):
    return {
        "version": "v3",
        "stage": 2,
        "epoch": epoch,
        "global_step": global_step,
        "stage_step": stage_step,
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(gpt_cfg),
        "train_config": asdict(cfg),
        "tokenizer_sha256": tokenizer_hash,
        "dataset_sha256": dataset_hashes,
        "schedule_total_steps": schedule_total_steps,
        "schedule_warmup_steps": schedule_warmup_steps,
        "rng_state": capture_rng_state(),
    }


# ============================================================
# 6. MAIN TRAINING
# ============================================================

def main():
    cfg = TrainConfig()
    gpt_cfg = GPTConfig()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    if use_amp and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    scaler_enabled = (
        use_amp and amp_dtype == torch.float16
    )

    print("Device          :", device)
    print("Mixed precision :", amp_dtype)
    print("GradScaler      :", scaler_enabled)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"

    (
        _,
        train_ds,
        val_ds,
        tokenizer_hash,
        dataset_hashes,
    ) = load_data(cfg, gpt_cfg)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training or validation dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_amp,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_amp,
    )

    model = MiniGPT(gpt_cfg).to(device)

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print(f"\nMODEL PARAMETERS: {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(device.type == "cuda"),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=scaler_enabled,
    )

    # Default state: starting from v3 Stage 1.
    global_step = 0
    v3_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    updates_per_epoch = math.ceil(
        len(train_loader) / cfg.grad_accum_steps
    )

    schedule_total_steps = max(
        1, updates_per_epoch * cfg.epochs
    )

    schedule_warmup_steps = min(
        cfg.warmup_steps,
        max(0, schedule_total_steps - 1),
    )

    # --------------------------------------------------------
    # Restore v3 Stage 4 if a previous checkpoint exists in output_dir.
    # Otherwise, continue on top of previous stage (Stage 3).
    # --------------------------------------------------------

    v3_checkpoint_exists = (
        cfg.resume_v3_stage and last_path.exists()
    )

    if v3_checkpoint_exists:
        checkpoint_path = last_path
        print("\nResuming previous v3 Stage 4 training...")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("model_config") != asdict(gpt_cfg):
            raise ValueError(
                "Model configuration mismatch with resumed checkpoint."
            )

        if checkpoint.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError(
                "Tokenizer mismatch. Use the exact tokenizer.json."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

        global_step = checkpoint.get("global_step", 0)
        v3_step = checkpoint.get("stage_step", 0)
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        schedule_total_steps = checkpoint.get(
            "schedule_total_steps", schedule_total_steps
        )
        schedule_warmup_steps = checkpoint.get(
            "schedule_warmup_steps", schedule_warmup_steps
        )

        restore_rng_state(checkpoint.get("rng_state"))

        print("Resumed checkpoint :", checkpoint_path)
        print("Next epoch          :", start_epoch + 1)
        print("v3 Stage 4 updates  :", v3_step)
        print("Global updates      :", global_step)
        print("Best validation loss:", best_val_loss)
    elif cfg.resume_from and Path(cfg.resume_from).exists():
        checkpoint_path = Path(cfg.resume_from)
        print("\nLoading weights from checkpoint:", checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("model_config") != asdict(gpt_cfg):
            raise ValueError(
                "Model configuration mismatch with resumed checkpoint."
            )
        model.load_state_dict(checkpoint["model"])
        # Do not load old optimizer state to allow fresh AdamW buffers for Stage 4
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Loaded weights successfully | Global updates: {global_step} | Prior best val loss: {best_val_loss:.4f}")
        print("v3 Stage 4 starts with fresh AdamW momentum buffers and refined learning rate.")
    else:
        print("\nStarting fresh training for v3 architecture (54.68M parameters).")

    # --------------------------------------------------------
    # Learning-rate schedule
    # --------------------------------------------------------

    def get_lr(step):
        if (
            schedule_warmup_steps > 0
            and step < schedule_warmup_steps
        ):
            return (
                cfg.learning_rate
                * (step + 1)
                / schedule_warmup_steps
            )

        decay_steps = max(
            1, schedule_total_steps - schedule_warmup_steps
        )

        progress = min(
            1.0,
            max(
                0.0,
                (step - schedule_warmup_steps) / decay_steps,
            ),
        )

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            cfg.min_learning_rate
            + cosine
            * (cfg.learning_rate - cfg.min_learning_rate)
        )

    if start_epoch >= cfg.epochs:
        print(
            "\nConfigured v3 epochs are already complete. "
            "Increase cfg.epochs to train further."
        )
        return

    print("\nV3 ARCHITECTURE (54.68M) TRAINING")
    print("-" * 60)
    print("Epochs configured :", cfg.epochs)
    print("Starting epoch    :", start_epoch + 1)
    print("Batch size        :", cfg.batch_size)
    print("Gradient accumulate:", cfg.grad_accum_steps)
    print("Checkpoint folder :", output_dir)
    print("-" * 60)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_train_loss = 0.0
        accumulated = 0
        epoch_updates = 0

        for batch_index, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            group_start = (
                batch_index // cfg.grad_accum_steps
            ) * cfg.grad_accum_steps

            group_size = min(
                cfg.grad_accum_steps,
                len(train_loader) - group_start,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                _, loss = model(x, y)

                if (
                    loss is None
                    or not torch.isfinite(loss).item()
                    or loss.item() > 20.0
                ):
                    print(
                        f"\n[WARNING] Skipping anomalous loss at batch {batch_index + 1}: "
                        f"{loss.item() if loss is not None else 'None'}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    continue

                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()

            total_train_loss += loss.item()
            accumulated += 1

            should_update = (
                accumulated == cfg.grad_accum_steps
                or batch_index == len(train_loader) - 1
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip,
                )

                lr = get_lr(v3_step)

                for group in optimizer.param_groups:
                    group["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                v3_step += 1
                epoch_updates += 1
                accumulated = 0

            if (
                (batch_index + 1) % cfg.log_every == 0
                or batch_index + 1 == len(train_loader)
            ):
                avg_loss = (
                    total_train_loss / (batch_index + 1)
                )

                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} | "
                    f"Batch {batch_index + 1}/{len(train_loader)} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}"
                )

        train_loss = (
            total_train_loss / max(1, len(train_loader))
        )

        val_loss = estimate_loss(
            model,
            val_loader,
            device,
            amp_dtype,
            use_amp,
            cfg.eval_batches,
        )

        if not math.isfinite(val_loss):
            raise RuntimeError(
                "Validation loss is not finite. "
                "Check the validation dataset and training stability."
            )

        print(
            f"\nEpoch {epoch + 1}/{cfg.epochs} complete | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Updates: {epoch_updates}"
        )

        # Update best loss BEFORE saving latest checkpoint.
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            gpt_cfg=gpt_cfg,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            stage_step=v3_step,
            best_val_loss=best_val_loss,
            tokenizer_hash=tokenizer_hash,
            dataset_hashes=dataset_hashes,
            schedule_total_steps=schedule_total_steps,
            schedule_warmup_steps=schedule_warmup_steps,
        )

        # Always save latest completed epoch.
        save_checkpoint(last_path, payload)
        print("Latest checkpoint:", last_path)

        # Save the best validation-loss model separately.
        if is_best:
            save_checkpoint(best_path, payload)
            print("New best checkpoint:", best_path)

        if device.type == "cuda":
            peak_gb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 3)
            )
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        print("-" * 60)

    print("\nV3 ARCHITECTURE TRAINING FINISHED")
    print("Best model          :", best_path)
    print("Latest model        :", last_path)
    print("Best validation loss:", f"{best_val_loss:.4f}")
    print("Total optimizer updates:", global_step)


if __name__ == "__main__":
    main()
