# Modern Decoder LLM from Scratch (Llama-3 Architecture)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.14+](https://img.shields.io/badge/PyTorch-2.14%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, first-principles PyTorch implementation of a modern decoder-only language model built from scratch. Incorporates the exact architectural building blocks of frontier open LLMs (Llama 3, Qwen 2.5): RMSNorm, RoPE with `rotate_half`, SwiGLU FFN, Grouped-Query Attention (GQA), and a preallocated static KV Cache that eliminates redundant prefix recomputations (attention remains O(t) per token).

---

## 📌 Project Overview & Highlights

> [!NOTE]
> **Status:** ✅ **Completed & Evaluated.** The complete Llama-3-style decoder architecture (RMSNorm, RoPE with `rotate_half`, SwiGLU, 4:1 GQA, tied weights with **5,573,888** parameters) has been implemented from mathematical first principles, verified across 16 unit tests (including strict causality test and 1-batch overfit with >95% loss drop), benchmarked across context lengths (yielding **2.10x** decoding speedup), and validated against analytical memory scaling laws.

### Core Capabilities & Engineering Deliverables
* **Modern Decoder Backbone:** Full Llama-3 stack with bias-free linear projections and scaled residual initialization ($1/\sqrt{2 \cdot \text{depth}}$).
* **Preallocated Static KV Cache:** Zero-copy static memory buffer storing un-repeated Key/Value heads (4:1 grouping ratio), achieving a **4x memory saving** and **2.10x faster decoding**.
* **Zero Out-of-Vocabulary (OOV) Byte-Level BPE:** Custom tokenizer with 4,096 vocabulary built on a 256-byte foundation, guaranteeing lossless encoding and decoding for any UTF-8, emoji, or non-Latin text.
* **Compact uint16 Dataset Pipeline:** Memory-mapped binary token storage (2 bytes/token) backed by `numpy.memmap` for zero-RAM page streaming during causal pretraining.
* **Selective Weight Decay Optimizer:** AdamW configured with strict parameter partitioning (weight decay = 0.1 on 2D linear matrices only; 0.0 on RMSNorm gains and embeddings).
* **Strict Causality Verification:** Automated unit test proving that future sequence perturbations produce 0.00e+00 output divergence on historical prefixes.
* **Deterministic Single-Batch Overfitting:** Verifies theoretical initial loss ($\approx \ln(4096) \approx 8.32$) and confirms >95% loss drop with >99% greedy token memorization.

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

Where $\gamma \in \mathbb{R}^d$ is a learnable scaling parameter initialized to 1.0, and $\epsilon = 10^{-6}$. Unlike standard LayerNorm, RMSNorm omits mean-subtraction, saving floating-point reductions while providing comparable empirical training stability.

### 2. RoPE (Rotary Position Embeddings with `rotate_half`)
To match Llama-3's implementation and ensure open-weight alignment, embedding dimensions are split into two contiguous halves:
$$\text{rotate\_half}(x) = \text{cat}\left( [-x[..., d_k/2:], \; x[..., :d_k/2]], \quad \text{dim}=-1 \right)$$
$$\text{RoPE}(x, m) = x \odot \cos(m \theta) + \text{rotate\_half}(x) \odot \sin(m \theta)$$

Where $\theta_i = \theta_{\text{base}}^{-2i/d_k}$ with base frequency $\theta_{\text{base}} = 10,000.0$. For query at position $m$ and key at position $n$, the inner product $\mathbf{q}_m^T \mathbf{k}_n$ depends strictly on the relative distance $m - n$.

### 3. SwiGLU Feed-Forward Network
All linear layers in the FFN are bias-free:
$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

Where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$ with explicit $d_{\text{ffn}} = 768$.

### 4. Grouped-Query Attention (GQA) & Un-repeated KV Storage
* **Query heads:** $H_Q = 8$, **Key/Value heads:** $H_{KV} = 2$ (grouping ratio $G = 4$).
* **Cache Storage:** Only the raw, un-repeated keys and values ($H_{KV} = 2$) are stored in memory.
* **Repetition:** After reading from cache, `torch.repeat_interleave(repeats=4, dim=1)` expands $K$ and $V$ to 8 heads immediately before computing attention against $Q$. This enforces a strict $4\times$ memory compression.
* **Causal Mask Offset:** When prefilling with $\text{cur\_len} > 0$, the mask position offset is $\text{cur\_len} + i$.

### 5. Algorithmic Complexity: KV Cache vs. Naive Generation
* **Without KV Cache (Full Prefix Recomputation):**
  At step $t$, the model recomputes projections and FFN for all past $t$ tokens:
  $$\text{FLOPs} = \sum_{t=1}^T \mathcal{O}(t \cdot d^2 + t^2 \cdot d) = \mathcal{O}(T^2 d^2 + T^3 d)$$
* **With KV Cache (Preallocated Static Buffer):**
  Only the new token passes through linear projections and FFN:
  $$\text{FLOPs} = \sum_{t=1}^T \mathcal{O}(d^2 + t \cdot d) = \mathcal{O}(T d^2 + T^2 d)$$
  *The KV cache eliminates redundant prefix projections and FFN passes. Attention per token remains O(t).*

### 6. Analytical KV Cache Memory Scaling
$$\text{KV Memory (bytes)} = 2 \times n_{\text{layers}} \times H_{KV} \times d_k \times L_{\text{seq}} \times \text{bytes\_per\_value}$$

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
| **32 tokens** | 183.6 tok/s | 291.8 tok/s | 269.1 tok/s | **1.47x** |
| **64 tokens** | 163.1 tok/s | 284.0 tok/s | 276.6 tok/s | **1.70x** |
| **128 tokens** | 130.0 tok/s | 285.5 tok/s | 273.0 tok/s | **2.10x** |

> [!NOTE]
> **Key Finding on Prefix Recomputation vs Attention Cost**:
> Without a KV cache, decoding throughput degrades quadratically as sequence length grows ($183.6 \to 130.0\text{ tok/s}$) because full linear projections and SwiGLU FFN blocks must be recomputed for all past $t$ tokens at every single emission step. With a KV cache, cached decoding maintains steady throughput ($\sim 273\text{ tok/s}$), achieving a **2.10x speedup** at 128 tokens.
> 
> At small model dimensions (0.75 MB cache), the throughput gap between dynamic `torch.cat` and preallocated static cache is small, but static preallocation provides deterministic zero-copy memory access and prevents heap fragmentation during long-running serving.

### 2. Analytical KV Cache Memory Scaling

$$\text{KV Memory Footprint} = 2 \times \text{depth} \times B \times n_{\text{kv\_heads}} \times T \times d_k \times \text{bytes\_per\_elem}$$

| Model Architecture | Layers | KV Heads ($H_{KV}$) | Head Dim ($d_k$) | Context ($T$) | Precision | KV Cache Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TinyStories LLM** | 6 | 2 | 32 | 256 | fp32 (4B) | **768.00 KB (0.75 MB)** |
| **Llama-3-8B** (Batch 1) | 32 | 8 | 128 | 8,192 | fp16 (2B) | **1.00 GiB** |
| **Llama-3-8B** (Batch 16) | 32 | 8 | 128 | 8,192 | fp16 (2B) | **16.00 GiB** |
| **Llama-3-8B** (Batch 1) | 32 | 8 | 128 | 131,072 | fp16 (2B) | **16.00 GiB** |

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

## 📖 References

1. **The Llama 3 Herd of Models** — Meta AI, 2024 ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783))
2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al., 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864))
3. **GLU Variants Improve Transformer (SwiGLU)** — Shazeer, 2020 ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202))
4. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — Ainslie et al., 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245))
5. **Root Mean Square Layer Normalization (RMSNorm)** — Zhang & Sennrich, 2019 ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467))
6. **TinyStories: How Small Can Language Models Be and Still Speak Coherent English?** — Eldan & Li, 2023 ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759))

---

## ✍️ Author & Citation

Built by **Hamza Sajid** ([@HamzA-1729](https://github.com/HamzA-1729)) as part of the *Deep Learning Foundations & Frontier Implementations from Scratch* series.

```bibtex
@misc{sajid2024moderndecoderllm,
  author = {Sajid, Hamza},
  title = {Modern Decoder LLM from Scratch: Llama-3 Architecture in PyTorch},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/HamzA-1729/modern_decoder_llm}}
}
```

---

## 📜 License
Distributed under the [MIT License](LICENSE).
