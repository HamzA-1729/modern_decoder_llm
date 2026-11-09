# Modern Decoder LLM from Scratch (Llama-3 Architecture)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, first-principles implementation of a modern decoder-only language model built in PyTorch from scratch. Incorporates the exact core architectural building blocks of frontier open LLMs (Llama 3, Qwen 2.5): RMSNorm, RoPE with `rotate_half`, SwiGLU FFN, Grouped-Query Attention (GQA), and a preallocated static KV Cache for $O(1)$ prefix token generation.*

---

## 📌 Project Status & Roadmap

> [!NOTE]
> **Status:** 🚧 Active Implementation. Architecture specification and configuration complete. Building core components and unit test suite.

- [x] **Project Scaffolding & MLOps Layout** (`pyproject.toml`, `configs/tinystories.yaml`)
- [ ] **Core Architectural Blocks** (`src/modern_decoder_llm/model.py`)
  - [ ] RMSNorm (bias-free with learnable $\gamma$)
  - [ ] RoPE (Rotary Position Embeddings with `rotate_half` layout)
  - [ ] SwiGLU Feed-Forward Network
  - [ ] Grouped-Query Attention (8 query heads, 2 KV heads, causal mask with position offset)
  - [ ] Weight Tying (LM head tied to input token embeddings $\implies$ 5.57M parameters)
  - [ ] Scaled Output Projection Initialization ($1/\sqrt{2 \cdot \text{depth}}$)
- [ ] **Component & Causality Tests** (`tests/test_components.py`, `tests/test_causality.py`)
- [ ] **Preallocated Static KV Cache & Sampler** (`src/modern_decoder_llm/kv_cache.py`, `sampler.py`)
- [ ] **KV Cache Verification Test** (`tests/test_kv_cache.py`)
- [ ] **Custom Byte-Level BPE Tokenizer (vocab 4096)** (`src/modern_decoder_llm/tokenizer.py`)
- [ ] **Memory-Mapped Dataset Loader** (`src/modern_decoder_llm/dataset.py`)
- [ ] **Training Pipeline & 1-Batch Overfit Test** (`src/modern_decoder_llm/train.py`, `tests/test_overfit.py`)
- [ ] **KV Cache Throughput (tok/s) & Analytical Memory Benchmark** (`src/modern_decoder_llm/benchmark.py`)

---

## 📐 Architecture: Llama-3-Style Decoder Stack

```
Input Token IDs: (B, T)
       │
       ▼
[ Token Embedding Table: Vocab (4096) x Dim (256) ]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ Transformer Decoder Layer (Repeated 6 times)           │
│                                                        │
│  x = x + GQA( RMSNorm(x) )                             │
│      ├── Q: 8 heads, K/V: 2 heads (4:1 GQA grouping)   │
│      ├── RoPE applied to Q and K (before caching K)    │
│      └── Preallocated Static KV Cache (un-repeated)    │
│                                                        │
│  x = x + SwiGLU( RMSNorm(x) )                          │
│      └── (SiLU(x·W_gate) ⊙ (x·W_up)) · W_down          │
│          with explicit ffn_dim = 768                   │
└────────────────────────────────────────────────────────┘
       │
       ▼
[ Final RMSNorm ]
       │
       ▼
[ Output LM Head (Tied to Token Embeddings) ] ──► Logits: (B, T, 4096)
```

---

## 📚 Mathematical Formulations & Complexity

### 1. RMSNorm (Root Mean Square Normalization)
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$
Where $\gamma \in \mathbb{R}^d$ is a learnable scaling vector initialized to 1.0, and $\epsilon = 10^{-6}$. Unlike LayerNorm, RMSNorm does not subtract the mean, saving compute while providing comparable empirical training stability.

### 2. RoPE (Rotary Position Embeddings with `rotate_half`)
To match Llama-3's implementation and allow compatibility with open weights, dimensions are split into two halves:
$$\text{rotate\_half}(x) = \text{cat}\left( [-x[:, :, :, d_k/2:], \; x[:, :, :, :d_k/2]], \quad \text{dim}=-1 \right)$$
$$\text{RoPE}(x, m) = x \odot \cos(m \theta) + \text{rotate\_half}(x) \odot \sin(m \theta)$$
Where $\theta_i = \theta_{\text{base}}^{-2i/d_k}$. For query at position $m$ and key at position $n$, the dot product $\mathbf{q}_m^T \mathbf{k}_n$ depends strictly on relative position $m - n$.

### 3. SwiGLU FFN (Bias-Free)
All linear projections in the FFN are bias-free:
$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$
Where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$ with explicit $d_{\text{ffn}} = 768$.

### 4. Grouped-Query Attention (GQA) & Un-repeated KV Caching
* Query projection: $H_Q = 8$, Key/Value projections: $H_{KV} = 2$ (4:1 grouping).
* **Cache Storage:** Only the raw, un-repeated keys and values ($H_{KV} = 2$) are stored in the cache.
* **Repetition:** After reading the active cache slice, `repeat_interleave(repeats=4, dim=1)` expands the keys and values to 8 heads to compute attention against $Q$. This preserves a strict $4\times$ KV cache memory saving.
* **Causal Mask Offset:** When prefilling with $\text{cur\_len} > 0$, the mask position offset is $\text{cur\_len} + i$.

### 5. Algorithmic Complexity: KV Cache vs. Naive Generation
* **Without KV Cache (Full Prefix Recomputation):**
  At step $t$, the model recomputes projections and FFN for all past $t$ tokens:
  $$\text{FLOPs} = \sum_{t=1}^T \mathcal{O}(t \cdot d^2 + t^2 \cdot d) = \mathcal{O}(T^2 d^2 + T^3 d)$$
* **With KV Cache (Preallocated Static Buffer):**
  Only the new token passes through projections and FFN:
  $$\text{FLOPs} = \sum_{t=1}^T \mathcal{O}(d^2 + t \cdot d) = \mathcal{O}(T d^2 + T^2 d)$$
  *The KV cache eliminates redundant prefix projections and FFN passes. Attention per token remains $\mathcal{O}(t)$.*

### 6. Analytical KV Cache Memory Footprint
$$\text{Memory (bytes)} = 2 \times n_{\text{layers}} \times H_{KV} \times d_k \times L_{\text{seq}} \times \text{bytes\_per\_value}$$
* For our 5.57M model ($n_{\text{layers}}=6, H_{KV}=2, d_k=32, L=256$, FP32): $\approx \mathbf{0.75 \text{ MB}}$ per sequence.
* For Llama-3-8B ($n_{\text{layers}}=32, H_{KV}=8, d_k=128, L=8192$, FP16):
  * With GQA ($H_{KV}=8$): $\approx \mathbf{1.0 \text{ GB}}$ per sequence.
  * With MHA ($H_Q=32$): $\approx \mathbf{4.0 \text{ GB}}$ per sequence ($4\times$ memory bloat!).

---

## ⚖️ Architectural Decisions & Justifications

| Component | Choice Made | Scientific / Mathematical Justification |
|---|---|---|
| **Normalization** | **RMSNorm** | Omits mean-subtraction, saving floating-point reductions while providing comparable training stability. |
| **Position Encoding** | **RoPE (`rotate_half`)** | Relative distance property $q_m^T k_n = f(m-n)$ enables length generalization; layout matches open weights. |
| **Feed-Forward** | **SwiGLU ($d_{\text{ffn}} = 768$)** | Gated non-linear units provide higher gradient quality and capacity per parameter than standard GELU/ReLU MLPs. |
| **Attention** | **GQA (8 Q-heads, 2 KV-heads)** | Exercises real 4:1 grouping code; achieves $4\times$ KV cache reduction with minimal quality loss. |
| **Weight Tying** | **LM Head $\equiv$ Embeddings** | Ties output linear weights to input embedding table, saving 1.05M parameters (19% of model size) for faster convergence. |
| **Tokenizer** | **Custom Byte-Level BPE (4096 vocab)** | Byte fallback eliminates OOV errors; small vocabulary avoids massive embedding tables on CPU/small GPUs. |
| **Weight Init** | **Scaled Output Projections** | All linear layers initialized from $\mathcal{N}(0, 0.02^2)$ with `o_proj` and `down_proj` scaled by $1/\sqrt{2 \cdot \text{depth}}$. |

---

## 📁 Repository Structure

```text
modern_decoder_llm/
├── README.md                      # Mathematical specification & findings
├── pyproject.toml                 # uv package configuration & dependencies
├── configs/
│   ├── cpu_smoke_test.yaml        # 2-layer config for instant smoke tests
│   └── tinystories.yaml           # 6-layer 5.57M production config
├── data/                          # Downloaded TinyStories text (git-ignored)
├── checkpoints/                   # Saved model weights & optimizers (git-ignored)
├── outputs/                       # Generation samples & benchmark plots (git-ignored)
├── src/modern_decoder_llm/
│   ├── __init__.py                # Package exports
│   ├── tokenizer.py               # Custom Byte-Level BPE (vocab 4096)
│   ├── model.py                   # RMSNorm, RoPE, SwiGLU, GQA, DecoderStack
│   ├── kv_cache.py                # Preallocated Static KVCache with cur_len pointer
│   ├── sampler.py                 # Autoregressive generation with Temperature & Top-p
│   ├── dataset.py                 # Memory-mapped uint16 dataset loader
│   ├── train.py                   # Training loop with weight-decay filtering & grad accum
│   └── benchmark.py               # Throughput (tok/s) & KV memory analysis
└── tests/
    ├── __init__.py
    ├── test_tokenizer.py          # Round-trip encode/decode, special tokens, zero OOV
    ├── test_components.py         # RMSNorm, RoPE relative invariance, SwiGLU, GQA==MHA
    ├── test_causality.py          # Causal masking future-leakage verification
    ├── test_kv_cache.py           # Cached vs uncached logit allclose & teacher-forced match
    └── test_overfit.py            # 1-batch overfit convergence test (>95% drop)
```

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/HamzA-1729/modern_decoder_llm.git
cd modern_decoder_llm
uv sync
```

### 2. Run Test Suite
```bash
uv run pytest -v
```

---

## 📜 License
Distributed under the [MIT License](LICENSE).

