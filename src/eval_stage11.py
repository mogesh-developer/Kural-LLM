import math
from pathlib import Path
import torch
from tokenizers import Tokenizer
from train import GPTConfig, MiniGPT, read_records, encode_records, PackedDataset, estimate_loss
from generate import generate

def evaluate_checkpoint(ckpt_path_str, tokenizer_path_str="tokenizer.json"):
    ckpt_path = Path(ckpt_path_str)
    if not ckpt_path.exists():
        print(f"File not found: {ckpt_path}")
        return

    print("=" * 70)
    print(f"EVALUATION: {ckpt_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # 1. Parameter size & Architecture details
    cfg_dict = ckpt.get("model_config") or ckpt.get("config", {})
    gpt_cfg = GPTConfig(**cfg_dict)
    
    model = MiniGPT(gpt_cfg)
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if "wte" in n or "wpe" in n)
    transformer_params = total_params - embedding_params

    print("\n--- Model Architecture & Parameter Counts ---")
    print(f"Vocab Size          : {gpt_cfg.vocab_size:,}")
    print(f"Context Length      : {gpt_cfg.context_length}")
    print(f"Embedding Dim (d)   : {gpt_cfg.n_embd}")
    print(f"Attention Heads     : {gpt_cfg.n_heads}")
    print(f"Transformer Layers  : {gpt_cfg.n_layers}")
    print(f"Total Parameters    : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Embedding Parameters: {embedding_params:,} ({embedding_params / 1e6:.2f}M, {embedding_params/total_params*100:.1f}%)")
    print(f"Non-Embedding Params: {transformer_params:,} ({transformer_params / 1e6:.2f}M, {transformer_params/total_params*100:.1f}%)")
    
    # Checkpoint training metadata
    print("\n--- Training Checkpoint Metadata ---")
    print(f"Stage               : {ckpt.get('stage', 'N/A')}")
    print(f"Completed Epoch     : {ckpt.get('epoch', 'N/A')}")
    print(f"Global Updates      : {ckpt.get('global_step', 'N/A')}")
    print(f"Stage Updates       : {ckpt.get('stage_step', 'N/A')}")
    stored_val_loss = ckpt.get("best_val_loss", None)
    if stored_val_loss is not None:
        print(f"Recorded Best Val Loss: {stored_val_loss:.4f} (Perplexity: {math.exp(stored_val_loss):.2f})")

    # 2. Validation Loss & Perplexity on current validation set
    val_file = "data/processed/validation.txt"
    tokenizer = Tokenizer.from_file(tokenizer_path_str)
    eos_id = tokenizer.token_to_id("[EOS]")

    if Path(val_file).exists():
        val_records = read_records(val_file)
        val_ids = encode_records(val_records, tokenizer, eos_id)
        val_ds = PackedDataset(val_ids, gpt_cfg.context_length)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

        amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        use_amp = device.type == "cuda"

        eval_val_loss = estimate_loss(model, val_loader, device, amp_dtype, use_amp, max_batches=100)
        perplexity = math.exp(eval_val_loss)
        print(f"\n--- Validation Set Metrics (100 batches) ---")
        print(f"Validation Loss     : {eval_val_loss:.4f}")
        print(f"Validation Perplexity: {perplexity:.2f}")

    # 3. Qualitative Discourse & Coherence Generation Samples
    prompts = [
        "தமிழ்நாடு",
        "இயற்கை சூழல்",
        "கல்வி என்பது",
        "வரலாற்று சிறப்புமிக்க",
        "அறிவியல் வளர்ச்சி"
    ]

    print("\n--- Qualitative Discourse & Coherence Evaluation Samples ---")
    for prompt in prompts:
        print(f"\n[Prompt]: {prompt}")
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=60,
            min_new_tokens=20,
            temperature=0.55,
            top_k=25,
            top_p=0.85,
            repetition_penalty=1.25,
        )
        print(f"[Generated]: {output}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    evaluate_checkpoint("checkpoints_stage11/best.pt")

