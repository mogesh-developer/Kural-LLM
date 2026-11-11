# 📜 Kural-LLM: Decoder-Only Tamil Language Model

**Kural-LLM** is an open-source, decoder-only causal language model trained specifically for the Tamil language from scratch. Built with a custom BPE tokenizer optimized for Tamil unicode morphology and a scalable Transformer architecture.

---

## 🏛️ Model Specifications

| Attribute | Kural-LLM (v3 Base) | Kural-LLM (v4 Scaled) |
| :--- | :--- | :--- |
| **Parameters** | **54.68 Million** | **91.59 Million** |
| **Hidden Dim ($d_{model}$)** | 640 | 768 |
| **Layers ($N$)** | 10 | 12 |
| **Attention Heads ($H$)** | 10 | 12 |
| **Context Length** | 512 tokens | 512 tokens |
| **Vocab Size** | 16,000 (Byte-level BPE) | 16,000 (Byte-level BPE) |
| **Activation** | GELU (tanh approximate) | GELU (tanh approximate) |
| **Attention** | FlashAttention (PyTorch SDPA) | FlashAttention (PyTorch SDPA) |

---

## 🚀 Key Features

- **Custom Tamil Tokenizer:** Custom byte-level Byte-Pair Encoding (BPE) trained natively on Tamil text with proper vowel-consonant composition.
- **Pre-training from Scratch:** Step-by-step progressive pre-training with Cosine Annealing learning rate schedules and gradient clipping.
- **Net2Net Architecture Scaling:** Successfully expanded from 54M to 91.6M parameters preserving existing representations.
- **VRAM Optimized:** Configured for mixed-precision (FP16/TF32) with gradient accumulation for low VRAM requirements.

---

## 📁 Repository Structure

```text
├── configs/              # Model and training configuration files
├── src/
│   ├── dataset.py        # Tokenized chunk packing & PyTorch DataLoaders
│   ├── model.py          # Causal Self-Attention, MLP & Transformer blocks
│   ├── train.py          # Training loop with anomaly checks & AMP
│   ├── generate.py       # Text sampling (Top-k, temperature, repetition penalty)
│   ├── preprocess.py     # Text normalization and deduplication
│   └── expand_model.py   # Net2Net parameter expansion script
├── tokenizer.json        # Trained 16K BPE Tokenizer
├── requirements.txt      # Project dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/<your-username>/Kural-LLM.git
cd Kural-LLM
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Tamil Text
```bash
python -m src.generate --checkpoint checkpoints_v3_stage4/last.pt --prompt "தமிழ்நாடு ஒரு"
```

---

## 📜 License
MIT License. Free to use for research, commercial, and educational purposes.

