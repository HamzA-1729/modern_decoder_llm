# Modern Decoder LLM from Scratch (Llama-3 Architecture)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*A clean, first-principles implementation of a modern decoder-only language model built in PyTorch from scratch. Incorporates the exact core architectural building blocks of frontier open LLMs (Llama 3, Qwen 2.5): RMSNorm, RoPE with `rotate_half`, SwiGLU FFN, Grouped-Query Attention (GQA), and a preallocated static KV Cache eliminating redundant prefix recomputations (attention still $\mathcal{O}(t)$ per token).*

---

## 📌 Project Status & Roadmap

> [!NOTE]
> **Status:** ✅ **Completed & Evaluated.** The complete Llama-3-style decoder architecture (RMSNorm, RoPE with `rotate_half`, SwiGLU, 4:1 GQA, tied weights $\implies$ **5,573,888** parameters) has been implemented from mathematical first principles, verified across all 16 unit tests (including strict causality test and 1-batch overfit with $>95\%$ loss drop), benchmarked across context lengths (yielding **$2.10\times$** decoding speedup), and validated against analytical memory scaling laws.

- [x] **Project Scaffolding & MLOps Layout** (`pyproject.toml`, `configs/tinystories.yaml`)
- [x] **Core Architectural Blocks** (`src/modern_decoder_llm/model.py`)
  - [x] RMSNorm (bias-free with learnable $\gamma$)
  - [x] RoPE (Rotary Position Embeddings with `rotate_half` layout)
  - [x] SwiGLU Feed-Forward Network ($d_{\text{ffn}} = 768$)
  - [x] Grouped-Query Attention (8 query heads, 2 KV heads, causal mask with position offset)
  - [x] Weight Tying (LM head tied to input token embeddings $\implies$ 5.57M parameters)
  - [x] Scaled Output Projection Initialization ($1/\sqrt{2 \cdot \text{depth}}$)
- [x] **Component & Causality Tests** (`tests/test_components.py`, `tests/test_causality.py`)
- [x] **Preallocated Static KV Cache & Sampler** (`src/modern_decoder_llm/kv_cache.py`, `sampler.py`)
- [x] **KV Cache Verification Tests** (`tests/test_kv_cache.py`)
- [x] **Custom Byte-Level BPE Tokenizer (vocab 4096, zero OOV)** (`src/modern_decoder_llm/tokenizer.py`, `tests/test_tokenizer.py`)
- [x] **Memory-Mapped uint16 Dataset Loader** (`src/modern_decoder_llm/dataset.py`, `tests/test_dataset.py`)
- [x] **Training Pipeline & Single-Batch Overfit Test** (`src/modern_decoder_llm/train.py`, `tests/test_overfit.py`)
- [x] **KV Cache Throughput & Analytical Memory Benchmark** (`src/modern_decoder_llm/benchmark.py`)
- [x] **Interactive CLI Text Generation Script** (`src/modern_decoder_llm/generate.py`)

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
Where $\theta_i = \theta_{\text{base}}^{-2i/d_k}$. For query at position $m$ and key at position $n$, the inner product $\mathbf{q}_m^T \mathbf{k}_n$ depends strictly on the relative distance $m - n$.

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
  * With GQA ($H_{KV}=8$): $\approx \mathbf{1.0 \text{ GiB}}$ per sequence.
  * With MHA ($H_Q=32$): $\approx \mathbf{4.0 \text{ GiB}}$ per sequence ($4\times$ memory bloat!).

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

## 📊 Empirical Benchmarks & Findings

### 1. Autoregressive Decoding Throughput (PyTorch CPU, 4 Threads)

| Generated Tokens | Uncached Decode | Dynamic `torch.cat` Cache | Static Preallocated Cache | Speedup vs. Uncached |
| :--- | :---: | :---: | :---: | :---: |
| **32 tokens** | $183.6 \text{ tok/s}$ | $291.8 \text{ tok/s}$ | $269.1 \text{ tok/s}$ | **$1.47\times$** |
| **64 tokens** | $163.1 \text{ tok/s}$ | $284.0 \text{ tok/s}$ | $276.6 \text{ tok/s}$ | **$1.70\times$** |
| **128 tokens** | $130.0 \text{ tok/s}$ | $285.5 \text{ tok/s}$ | $273.0 \text{ tok/s}$ | **$2.10\times$** |

> [!NOTE]
> **Key Finding on Prefix Recomputation vs Attention Cost**:
> Without a KV cache, decoding throughput degrades quadratically as sequence length grows ($183.6 \to 130.0\text{ tok/s}$) because full linear projections and SwiGLU FFN blocks must be recomputed for all past $t$ tokens at every single emission step. With a KV cache, cached decoding maintains steady throughput ($\sim 273\text{ tok/s}$), achieving a **$2.10\times$ speedup** at 128 tokens.
> 
> At small model dimensions ($0.75\text{ MB}$ cache), the throughput gap between dynamic `torch.cat` and preallocated static cache is small, but static preallocation provides deterministic zero-copy memory access and prevents heap fragmentation during long-running serving.

### 2. Analytical KV Cache Memory Scaling

$$\text{KV Memory Footprint} = 2 \times \text{depth} \times B \times n_{\text{kv\_heads}} \times T \times d_k \times \text{bytes\_per\_elem}$$

| Model Architecture | Layers | KV Heads ($H_{KV}$) | Head Dim ($d_k$) | Context ($T$) | Precision | KV Cache Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TinyStories LLM** | 6 | 2 | 32 | 256 | fp32 (4B) | **$768.00\text{ KB}$ ($0.75\text{ MB}$)** |
| **Llama-3-8B** (Batch 1) | 32 | 8 | 128 | 8,192 | fp16 (2B) | **$1.00\text{ GiB}$** |
| **Llama-3-8B** (Batch 16) | 32 | 8 | 128 | 8,192 | fp16 (2B) | **$16.00\text{ GiB}$** |
| **Llama-3-8B** (Batch 1) | 32 | 8 | 128 | 131,072 | fp16 (2B) | **$16.00\text{ GiB}$** |

---

## 📁 Repository Structure

```text
modern_decoder_llm/
├── README.md                      # Mathematical specification & empirical findings
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
│   ├── train.py                   # Training loop with selective weight decay & grad accum
│   ├── benchmark.py               # Throughput (tok/s) & analytical memory analysis
│   └── generate.py                # Standalone CLI text generation interface
└── tests/
    ├── __init__.py
    ├── test_tokenizer.py          # Round-trip encode/decode, special tokens, zero OOV
    ├── test_components.py         # RMSNorm, RoPE relative invariance, SwiGLU, GQA==MHA
    ├── test_causality.py          # Strict causal masking future-leakage verification
    ├── test_kv_cache.py           # Cached vs uncached logit equivalence & teacher-forced match
    ├── test_dataset.py            # Memory-mapped uint16 zero-copy slicing & causal pairs
    └── test_overfit.py            # 1-batch overfit convergence test (>95% drop, >99% memorization)
```

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://github.com/HamzA-1729/modern_decoder_llm.git
cd modern_decoder_llm
uv sync
```

### 2. Run Test Suite (16 Automated Tests)
```bash
uv run pytest -v
```

### 3. Run Inference & KV Cache Benchmark
```bash
uv run python -m modern_decoder_llm.benchmark
```

### 4. Interactive Text Generation CLI
```bash
uv run python -m modern_decoder_llm.generate \
  --prompt "Once upon a time, there was a little bird" \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-p 0.9
```

### 5. TinyStories Pretraining Pipeline
```bash
uv run python -m modern_decoder_llm.train
```

---

## 📜 License
Distributed under the [MIT License](LICENSE).
