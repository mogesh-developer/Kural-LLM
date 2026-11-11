# Tamil Mini-LLM: Generation & Inference Tuning Guide

This document details how text generation quality, coherence, continuity, and loop prevention are improved in `src/generate.py` without requiring model re-training or fine-tuning.

---

## 1. The Core Problem in Small Language Models

When testing a small generative model (Mini-GPT, ~12M–20M parameters) trained on raw web corpora, raw autoregressive sampling typically exhibits several issues:

1. **Repetitive Loops / Stuttering:** The model repeats phrases indefinitely (e.g., *துறைத் துறைத் துறை...*, *பெண்கள், பெண்கள்...*).
2. **Premature Stopping:** The model outputs a single punctuation mark or abruptly emits `[EOS]` after short prompts (e.g., *"தமிழ்நாட்டின் வரலாறு."*).
3. **Gibberish / Tail Decay:** Low-probability tail tokens get sampled randomly under basic Top-K or pure Temperature sampling.
4. **Spacing Defects:** Subword decoders may attach the continuation directly onto the prompt without a separating space (e.g., *சென்னை மாநகரம்முழுவதும்*).

---

## 2. Why Improvements Are Possible Without Fine-Tuning

During generation, the model does **not** output final words. At each token step, it outputs raw unnormalized scores (**logits**) across the entire vocabulary ($V \approx 32,000$ tokens):

$$\text{Model Transformer Blocks} \longrightarrow \text{Logits } z \in \mathbb{R}^{V} \longrightarrow \text{Logit Adjustments} \longrightarrow \text{Softmax } p = \sigma(z) \longrightarrow \text{Token Sampling}$$

By reshaping, masking, and penalizing these logits **before** computing the Softmax distribution, we control the model's behavior at inference time without modifying any neural network weights (`weights_only=False` or trained checkpoints remain intact).

---

## 3. Techniques Implemented in `src/generate.py`

### A. Repetition Penalty (`repetition_penalty = 1.25`)
- **Mechanism:** Maintains the set of previously generated tokens $\mathcal{S}_{\text{seen}}$. Before sampling step $t$:
  $$\tilde{z}_i = \begin{cases} z_i / r & \text{if } z_i > 0 \\ z_i \cdot r & \text{if } z_i \le 0 \end{cases} \quad \forall i \in \mathcal{S}_{\text{seen}}$$
  where $r = 1.25$.
- **Result:** Drastically reduces self-reinforcing loops, compelling the attention heads to attend to alternative paths and richer vocabulary.

### B. Minimum Token Length Enforcement (`min_new_tokens = 25`)
- **Mechanism:** For any generation step $t < \text{min\_new\_tokens}$:
  $$z_{[\text{EOS}]} = -\infty$$
- **Result:** The model is prohibited from concluding sentences prematurely on short prompts. It forces the autoregressive loop to develop context, explanation, or a complete sentence.

### C. Top-P (Nucleus) Filtering (`top_p = 0.90`)
- **Mechanism:** Sorts all vocabulary logits in descending order and computes the cumulative probability distribution:
  $$\sum_{i=1}^{k} p_i \ge p_{\text{nucleus}} \quad (0.90)$$
  All tokens outside the top 90% cumulative mass have their logits set to $-\infty$.
- **Result:** Dynamically sizes the candidate pool. When the model is confident, it chooses from 1–3 tokens; when uncertain, it considers a larger candidate pool while discarding confusing low-probability tokens.

### D. Separation & Spacing Alignment
- **Mechanism:** Automatically checks whether `prompt` ends with whitespace and `continuation` begins with whitespace, cleanly inserting separation where required.

---

## 4. Parameter Reference & Defaults (`src/generate.py`)

| Parameter | Default Value | Recommended Range | Purpose |
|---|:---:|:---:|---|
| `MAX_NEW_TOKENS` | `100` | `50 – 250` | Maximum token cutoff limit |
| `MIN_NEW_TOKENS` | `25` | `15 – 40` | Prevents premature `[EOS]` termination |
| `TEMPERATURE` | `0.75` | `0.6 – 0.85` | Controls randomness vs. determinism |
| `TOP_K` | `40` | `20 – 50` | Hard cutoff for top candidates per step |
| `TOP_P` | `0.90` | `0.85 – 0.95` | Dynamic nucleus cutoff |
| `REPETITION_PENALTY` | `1.25` | `1.15 – 1.35` | Loop suppression factor |

---

## 5. Generation Benchmark (Before vs. After)

| Prompt | Before Fix | After Inference Fix |
|---|---|---|
| **தமிழ்நாட்டின் வரலாறு** | *தமிழ்நாட்டின் வரலாறு.* (immediate EOS stop) | *தமிழ்நாட்டின் வரலாறு , 'சங்க'கடீஸ்காரர் என்று, 'தரன்'யன்தார்' என்றார்.* (continuous generation) |
| **சென்னை மாநகரம்** | *சென்னை மாநகரம்முழுவதும் உள்ள அரசு ஊழியர்கள் என...* (stuck word spacing) | *சென்னை மாநகரம் முழுவதும் உள்ள அரசு சார்பில் மத்திய நிதி ஒதுக்கீடு செய்துள்ளது. இது தொடர்பாக பா. ஜ. , தலைவர்கள்...* |
| **மழை எவ்வாறு உருவாகிறது?** | *மழை எவ்வாறு உருவாகிறது?* (immediate EOS stop) | *மழை எவ்வாறு உருவாகிறது? என்பதை நாம் விரும்பினால், அது உங்கள் வாழ்க்கைக்க வேண்டியதான். இந்த அனுபவத்தை...* |
| **கல்வியின் முக்கியத்துவம்** | *கல்வியின் முக்கியத்துவம்வாய்ந்தது என்பது ஒரு பெண்.* | *கல்வியின் முக்கியத்துவம் வாய்ந்த இடம் மற்றும் பொருளாதார ரீதியாகத் தொழிலாளர்கள் போன்ற ஒரு பகுதியாகக்...* |

---

## 6. How to Run Generation

### Interactive CLI:
```powershell
.\venv\Scripts\python.exe src\generate.py
```

### Scripted Programmatic Usage:
```python
from pathlib import Path
import torch
from tokenizers import Tokenizer
from train import GPTConfig, MiniGPT
from generate import generate

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("tokenizer.json")
checkpoint = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)

cfg = checkpoint.get("model_config") or checkpoint.get("config", {})
model = MiniGPT(GPTConfig(**cfg)).to(device)
model.load_state_dict(checkpoint.get("model") or checkpoint.get("model_state_dict"))

prompt = "தமிழ் மொழி உலகின்"
output = generate(
    model, 
    tokenizer, 
    prompt, 
    device, 
    max_new_tokens=80, 
    min_new_tokens=25, 
    repetition_penalty=1.25,
    temperature=0.75,
    top_p=0.9
)
print(output)
```

---

## 7. Next Steps for Factual & Logical Reasoning

While inference filtering fixes repetitions and sentence flow:
1. **Factual accuracy** (e.g. knowing who authored *திருக்குறள்*) requires training on a larger corpus (500MB+ of curated Tamil text / Wikipedia).
2. **Direct Q&A instructions** (e.g., answering *"மழை எவ்வாறு உருவாகிறது?"* with scientific steps) requires **Instruction Fine-Tuning (SFT)** on prompt-response pairs.

