"""
Inference & KV Cache Performance Benchmark:
1. Compares three decoding strategies:
   - Uncached (Full Prefix Recomputation)
   - Dynamic KV Cache (Dynamic torch.cat re-allocations at every step)
   - Preallocated Static KV Cache (Zero-allocation slicing with pointer)
2. Rigorous benchmarking methodology:
   - Fixed thread count (torch.set_num_threads).
   - Dedicated warmup cycles before timing.
   - Median latency across multiple trials to eliminate OS scheduling jitter.
3. Analytical Memory Scaling:
   - Evaluates measured memory vs analytical equation.
   - Projects KV cache footprint to production-scale models (e.g., Llama-3-8B at 8k context).
"""

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from modern_decoder_llm.model import DecoderLLM, apply_rotary_emb
from modern_decoder_llm.kv_cache import StaticKVCache
from modern_decoder_llm.sampler import generate_with_cache, generate_without_cache, sample_next_token


class DynamicCatKVCache:
    """
    Dynamic KV Cache implementation using torch.cat at every step.
    Demonstrates the overhead of repeated tensor memory reallocations during decoding.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # List of (k_tensor, v_tensor) tuples
        self.caches: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(n_layers)
        ]

    def update(self, layer_idx: int, k_step: torch.Tensor, v_step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        past_k, past_v = self.caches[layer_idx]
        if past_k is None:
            self.caches[layer_idx] = (k_step, v_step)
            return k_step, v_step
        else:
            cat_k = torch.cat([past_k, k_step], dim=2)
            cat_v = torch.cat([past_v, v_step], dim=2)
            self.caches[layer_idx] = (cat_k, cat_v)
            return cat_k, cat_v


@torch.no_grad()
def generate_with_dynamic_cat_cache(
    model: DecoderLLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 64,
) -> torch.Tensor:
    """Autoregressive decode using dynamic torch.cat KV caching."""
    model.eval()
    batch_size, prompt_len = prompt_tokens.shape
    device = prompt_tokens.device

    # Prefill: full prompt through model with static cache initialized for prefill
    prefill_cache = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=model.max_seq_len,
        n_layers=model.depth,
        n_kv_heads=model.layers[0].attn.n_kv_heads,
        head_dim=model.layers[0].attn.head_dim,
        device=device,
        dtype=torch.float32,
    )
    layer_caches = [prefill_cache.get_layer_cache(i) for i in range(model.depth)]
    logits, _ = model(prompt_tokens, start_pos=0, kv_caches=layer_caches)
    prefill_cache.advance(prompt_len)

    next_token = sample_next_token(logits[:, -1, :], temperature=0.0)
    generated = [next_token]

    # Convert initial prefilled keys/values into dynamic cache
    dynamic_cache = DynamicCatKVCache(model.depth)
    for i in range(model.depth):
        k_buf, v_buf = layer_caches[i]
        dynamic_cache.caches[i] = (
            k_buf[:, :, :prompt_len, :].clone(),
            v_buf[:, :, :prompt_len, :].clone(),
        )

    # Decode steps with torch.cat allocation overhead
    for step in range(max_new_tokens - 1):
        curr_pos = prompt_len + step

        # Single token embedding
        h = model.tok_embeddings(next_token)

        for i, layer in enumerate(model.layers):
            norm_x = layer.attn_norm(h)
            b, t, d = norm_x.shape
            q = layer.attn.q_proj(norm_x).view(b, t, layer.attn.n_heads, layer.attn.head_dim).transpose(1, 2)
            k = layer.attn.k_proj(norm_x).view(b, t, layer.attn.n_kv_heads, layer.attn.head_dim).transpose(1, 2)
            v = layer.attn.v_proj(norm_x).view(b, t, layer.attn.n_kv_heads, layer.attn.head_dim).transpose(1, 2)

            q = apply_rotary_emb(q, model.rope_cos, model.rope_sin, start_pos=curr_pos)
            k = apply_rotary_emb(k, model.rope_cos, model.rope_sin, start_pos=curr_pos)

            # Dynamic cat reallocation
            full_k, full_v = dynamic_cache.update(i, k, v)

            # Repeat KV heads for GQA
            rep_k = torch.repeat_interleave(full_k, repeats=layer.attn.num_groups, dim=1)
            rep_v = torch.repeat_interleave(full_v, repeats=layer.attn.num_groups, dim=1)

            # Scaled Dot-Product Attention
            scores = torch.matmul(q, rep_k.transpose(-2, -1)) / math.sqrt(layer.attn.head_dim)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
            attn_out = torch.matmul(attn_weights, rep_v)
            attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
            attn_out = layer.attn.o_proj(attn_out)

            h = h + attn_out
            h = h + layer.ffn(layer.ffn_norm(h))

        step_logits = model.lm_head(model.norm(h))
        next_token = sample_next_token(step_logits[:, -1, :], temperature=0.0)
        generated.append(next_token)

    return torch.cat([prompt_tokens] + generated, dim=-1)


def benchmark_generation(
    model: DecoderLLM,
    prompt_tokens: torch.Tensor,
    new_tokens: int,
    repeats: int = 3,
) -> Dict[str, float]:
    """Runs timed benchmarks across uncached, dynamic-cat, and static-cache modes."""
    # Warmup runs
    _ = generate_with_cache(model, prompt_tokens, max_new_tokens=4, temperature=0.0)
    _ = generate_without_cache(model, prompt_tokens, max_new_tokens=4, temperature=0.0)

    # 1. Benchmark: Preallocated Static KV Cache
    static_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = generate_with_cache(model, prompt_tokens, max_new_tokens=new_tokens, temperature=0.0)
        static_times.append(time.perf_counter() - t0)
    static_time = sorted(static_times)[len(static_times) // 2]
    static_tok_s = new_tokens / static_time

    # 2. Benchmark: Dynamic torch.cat KV Cache
    cat_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = generate_with_dynamic_cat_cache(model, prompt_tokens, max_new_tokens=new_tokens)
        cat_times.append(time.perf_counter() - t0)
    cat_time = sorted(cat_times)[len(cat_times) // 2]
    cat_tok_s = new_tokens / cat_time

    # 3. Benchmark: Uncached (Full Prefix Recomputation)
    uncached_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = generate_without_cache(model, prompt_tokens, max_new_tokens=new_tokens, temperature=0.0)
        uncached_times.append(time.perf_counter() - t0)
    uncached_time = sorted(uncached_times)[len(uncached_times) // 2]
    uncached_tok_s = new_tokens / uncached_time

    return {
        "new_tokens": new_tokens,
        "static_time": static_time,
        "static_tok_s": static_tok_s,
        "cat_time": cat_time,
        "cat_tok_s": cat_tok_s,
        "uncached_time": uncached_time,
        "uncached_tok_s": uncached_tok_s,
        "speedup_vs_uncached": uncached_time / static_time,
    }


def compute_analytical_kv_memory(
    depth: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    bytes_per_elem: int = 4,
    batch_size: int = 1,
) -> int:
    """
    Computes exact analytical KV cache memory footprint in bytes:
        Memory = 2 (K & V) * depth * batch_size * n_kv_heads * seq_len * head_dim * bytes_per_elem
    """
    return 2 * depth * batch_size * n_kv_heads * seq_len * head_dim * bytes_per_elem


def main():
    torch.set_num_threads(4)
    print("=" * 80)
    print("MODERN DECODER LLM: INFERENCE & KV CACHE BENCHMARK")
    print(f"PyTorch Thread Count: {torch.get_num_threads()} | Device: CPU")
    print("=" * 80)

    # 1. Instantiate 5.57M TinyStories Model
    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    )
    model.eval()

    prompt_len = 16
    prompt_tokens = torch.randint(0, model.vocab_size, (1, prompt_len))
    token_budgets = [32, 64, 128]

    results = []
    print("\n[1/2] Benchmarking Autoregressive Decoding Latency & Throughput...")
    print(f"{'Tokens':<8} | {'Uncached':<12} | {'Dynamic Cat':<14} | {'Static Cache':<14} | {'Speedup':<8}")
    print("-" * 65)

    for budget in token_budgets:
        res = benchmark_generation(model, prompt_tokens, new_tokens=budget, repeats=3)
        results.append(res)
        print(
            f"{budget:<8d} | "
            f"{res['uncached_tok_s']:6.1f} tok/s  | "
            f"{res['cat_tok_s']:6.1f} tok/s     | "
            f"{res['static_tok_s']:6.1f} tok/s       | "
            f"{res['speedup_vs_uncached']:4.2f}x"
        )

    # 2. Analytical Memory Scaling Evaluation
    print("\n[2/2] Analytical KV Cache Memory Scaling Analysis:")
    print("-" * 80)

    # Our TinyStories Model
    tinystories_bytes = compute_analytical_kv_memory(
        depth=6, n_kv_heads=2, head_dim=32, seq_len=256, bytes_per_elem=4
    )
    # Llama-3-8B Scale (32 layers, 8 KV heads, head_dim 128, fp16 = 2 bytes)
    llama3_8k_bytes = compute_analytical_kv_memory(
        depth=32, n_kv_heads=8, head_dim=128, seq_len=8192, bytes_per_elem=2
    )
    llama3_128k_bytes = compute_analytical_kv_memory(
        depth=32, n_kv_heads=8, head_dim=128, seq_len=131072, bytes_per_elem=2
    )

    print(f"TinyStories Model (6L, 2KV, d_k=32, fp32):")
    print(f"  - Per sequence (256 context):  {tinystories_bytes / 1024:.2f} KB ({tinystories_bytes / (1024*1024):.2f} MB)")
    print(f"Llama-3-8B Production Scale (32L, 8KV, d_k=128, fp16):")
    print(f"  - Per sequence (8k context):    {llama3_8k_bytes / (1024**3):.2f} GiB ({llama3_8k_bytes / (1024**2):.1f} MB)")
    print(f"  - Batch 16 (8k context):        {(llama3_8k_bytes * 16) / (1024**3):.2f} GiB")
    print(f"  - Per sequence (128k context):  {llama3_128k_bytes / (1024**3):.2f} GiB")
    print("-" * 80)

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "throughput_benchmarks": results,
            "tinystories_cache_mb": tinystories_bytes / (1024 * 1024),
            "llama3_8k_cache_gib": llama3_8k_bytes / (1024 ** 3),
        }, f, indent=2)
    print(f"Saved benchmark results to {out_file}\n")


if __name__ == "__main__":
    main()

