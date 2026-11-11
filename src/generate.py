
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train import GPTConfig, MiniGPT


CHECKPOINT = (
    Path("checkpoints_v3_stage4/best.pt")
    if Path("checkpoints_v3_stage4/best.pt").exists()
    else (
        Path("checkpoints_v3_stage3/best.pt")
        if Path("checkpoints_v3_stage3/best.pt").exists()
        else (
            Path("checkpoints_v3_stage2/best.pt")
            if Path("checkpoints_v3_stage2/best.pt").exists()
            else (
                Path("checkpoints_v3/best.pt")
                if Path("checkpoints_v3/best.pt").exists()
                else (
                    Path("checkpoints_stage12/best.pt")
                    if Path("checkpoints_stage12/best.pt").exists()
                    else (
                        Path("checkpoints_stage11/best.pt")
                        if Path("checkpoints_stage11/best.pt").exists()
                        else Path("checkpoints/best.pt")
                    )
                )
            )
        )
    )
)
TOKENIZER_PATH = Path("tokenizer.json")

MAX_NEW_TOKENS = 100
MIN_NEW_TOKENS = 25
TEMPERATURE = 0.55
TOP_K = 25
TOP_P = 0.85
REPETITION_PENALTY = 1.25


def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=MAX_NEW_TOKENS,
    min_new_tokens=MIN_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
):
    model.eval()

    input_ids = tokenizer.encode(prompt).ids
    if not input_ids:
        input_ids = [tokenizer.token_to_id("[BOS]")]

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")
    context_length = model.config.context_length

    with torch.no_grad():
        for step in range(max_new_tokens):
            x_cond = x[:, -context_length:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty on previously generated/input tokens
            if repetition_penalty != 1.0:
                seen_tokens = torch.unique(x[0])
                for token_id in seen_tokens:
                    val = logits[0, token_id]
                    if val > 0:
                        logits[0, token_id] = val / repetition_penalty
                    else:
                        logits[0, token_id] = val * repetition_penalty

            # Prevent premature stopping before min_new_tokens
            if step < min_new_tokens and eos_id is not None:
                logits[0, eos_id] = float("-inf")

            # Top-K filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                threshold = values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            if eos_id is not None and token_id == eos_id:
                break

            x = torch.cat([x, next_token], dim=1)

    generated_ids = x[0, len(input_ids):].tolist()
    continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Ensure clean separation between prompt and generated continuation if needed
    if prompt and continuation and not prompt.endswith(" ") and not continuation.startswith(" "):
        full_text = prompt + " " + continuation
    else:
        full_text = prompt + continuation

    return full_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    # Support both "model_config" / "config" and "model" / "model_state_dict" keys
    cfg_dict = checkpoint.get("model_config") or checkpoint.get("config", {})
    config = GPTConfig(**cfg_dict)
    model = MiniGPT(config)

    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)

    print("Checkpoint:", CHECKPOINT)
    print("Device:", device)
    print("Enter a Tamil prompt. Type 'exit' to quit.\n")

    while True:
        prompt = input("Prompt: ").strip()

        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        print("\nGenerated text:")
        print(generate(model, tokenizer, prompt, device))
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
