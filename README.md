---
language:
- ta
license: mit
library_name: pytorch
tags:
- causal-lm
- text-generation
- tamil
- indic-nlp
- gpt
- from-scratch
datasets:
- custom
metrics:
- perplexity
- loss
pipeline_tag: text-generation
widget:
- text: "தமிழ்நாடு ஒரு"
- text: "இயற்கை என்பது"
- text: "தொழில்நுட்பம் நம் அன்றாட வாழ்க்கையில்"
---

# 📜 Kural-LLM: Decoder-Only Tamil Language Model

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Kural--LLM-yellow)](https://huggingface.co/mogesh2004/Kural-LLM)
[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Kural-LLM** is an open-source, decoder-only causal language model trained natively from scratch for the Tamil language. Built with a custom BPE tokenizer optimized for Tamil unicode morphology and a scalable Transformer architecture.

🔗 **Pre-trained Weights & Model Hub:** [huggingface.co/mogesh2004/Kural-LLM](https://huggingface.co/mogesh2004/Kural-LLM)

---

## 🏛️ Model Specifications

| Attribute | Kural-LLM (v3 Base - Current Release) | Kural-LLM (v4 Scaled) |
| :--- | :--- | :--- |
| **Parameters** | **54.68 Million** | **91.59 Million** |
| **Hidden Dim ($d_{model}$)** | 640 | 768 |
| **Layers ($N$)** | 10 | 12 |
| **Attention Heads ($H$)** | 10 | 12 |
| **Context Length** | 512 tokens | 512 tokens |
| **Vocab Size** | 16,000 (Byte-level BPE) | 16,000 (Byte-level BPE) |
| **Best Val Loss** | **3.6623** (Global Step 18,645) | In Training |
| **Activation** | GELU (tanh approximate) | GELU (tanh approximate) |
| **Attention** | FlashAttention (PyTorch SDPA) | FlashAttention (PyTorch SDPA) |

---

## 🚀 Key Features

- **Native Tamil Tokenizer:** Custom byte-level Byte-Pair Encoding (BPE) trained natively on Tamil text with full vowel-consonant composition.
- **Pre-training from Scratch:** Step-by-step progressive pre-training with Cosine Annealing learning rate schedules and gradient clipping.
- **High Stability:** Trained locally across multiple stages with strict gradient anomaly checks, reaching a stable validation loss of **3.6623**.
- **Net2Net Architecture Scaling:** Successfully expanded to 91.6M parameters preserving existing representations.

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
git clone https://github.com/mogesh-developer/Kural-LLM.git
cd Kural-LLM
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Tamil Text with this Model
```bash
python -m src.generate --checkpoint pytorch_model.pt --prompt "தமிழ்நாடு ஒரு"
```

---

## 📜 License
MIT License. Free to use for research, commercial, and educational purposes.
