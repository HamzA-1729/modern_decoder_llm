"""
Unit Tests for Static Key-Value Cache:
1. Logit Equivalence: Validates that autoregressive decoding with StaticKVCache matches
   the full-sequence uncached forward pass within floating-point tolerance (atol=1e-4).
   (Note: float32 matmul summation order differs between (1 x d) and (t x d) shapes).
2. Teacher-Forced Greedy Matching: Verifies that greedy token choices match between
   cached and uncached decoding paths under identical context.
3. Cache Lifecycle & Memory Footprint: Verifies pointer advancement, buffer reset,
   and exact analytical memory scaling (4x memory savings from storing un-repeated KV heads).
"""

import pytest
import torch
from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.kv_cache import StaticKVCache
from modern_decoder_llm.sampler import generate_with_cache, generate_without_cache


@pytest.fixture
def test_model() -> DecoderLLM:
    """Fixture providing a deterministic DecoderLLM for caching verification."""
    torch.manual_seed(42)
    model = DecoderLLM(
        vocab_size=4096,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        ffn_dim=768,
        max_seq_len=256,
        rope_theta=10000.0,
        norm_eps=1e-5,
    )
    model.eval()
    return model


def test_kv_cache_logit_equivalence_incremental(test_model: DecoderLLM):
    """
    Validates that feeding tokens incrementally with StaticKVCache produces logits
    equivalent to feeding the full sequence at once without cache.
    
    Tolerances:
    Floating-point summation order in (1 x d) @ (d x t) differs from (t x d) @ (d x t),
    so atol=1e-4 is asserted rather than exact bitwise equality.
    """
    torch.manual_seed(1337)
    batch_size = 2
    prompt_len = 4
    decode_steps = 8
    total_seq_len = prompt_len + decode_steps

    # Generate synthetic input sequence
    tokens = torch.randint(0, test_model.vocab_size, (batch_size, total_seq_len))

    # --- 1. Ground Truth: Uncached full sequence forward pass ---
    with torch.no_grad():
        logits_uncached, _ = test_model(tokens, start_pos=0, kv_caches=None)

    # --- 2. Incremental Cached Forward Pass ---
    kv_cache_mgr = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=test_model.max_seq_len,
        n_layers=test_model.depth,
        n_kv_heads=test_model.layers[0].attn.n_kv_heads,
        head_dim=test_model.layers[0].attn.head_dim,
        device=tokens.device,
        dtype=torch.float32,
    )
    layer_caches = [kv_cache_mgr.get_layer_cache(i) for i in range(test_model.depth)]

    # Phase A: Prefill prompt tokens
    with torch.no_grad():
        logits_prefill, _ = test_model(tokens[:, :prompt_len], start_pos=0, kv_caches=layer_caches)
    kv_cache_mgr.advance(prompt_len)

    # Assert prefill logits match uncached prefix
    assert torch.allclose(logits_prefill, logits_uncached[:, :prompt_len, :], atol=1e-4), (
        f"Prefill logits divergence: max diff = "
        f"{(logits_prefill - logits_uncached[:, :prompt_len, :]).abs().max().item():.2e}"
    )

    # Phase B: Incremental 1-token decode steps
    collected_step_logits = []
    for step in range(decode_steps):
        current_pos = kv_cache_mgr.cur_len
        token_step = tokens[:, current_pos : current_pos + 1]

        with torch.no_grad():
            logit_step, _ = test_model(token_step, start_pos=current_pos, kv_caches=layer_caches)
        kv_cache_mgr.advance(1)

        collected_step_logits.append(logit_step)

    # Concatenate all cached logits: (B, total_seq_len, vocab_size)
    all_cached_logits = torch.cat([logits_prefill] + collected_step_logits, dim=1)

    max_diff = (all_cached_logits - logits_uncached).abs().max().item()
    print(f"\nKV Cache vs Uncached Max Absolute Logit Difference: {max_diff:.2e}")

    assert max_diff < 1e-4, f"Cached logits diverged beyond atol=1e-4 (max diff = {max_diff:.2e})"
    assert torch.allclose(all_cached_logits, logits_uncached, atol=1e-4)


def test_kv_cache_teacher_forced_greedy_tokens(test_model: DecoderLLM):
    """
    Verifies that under teacher-forcing (identical historical prefix),
    cached decoding produces the exact same greedy next-token predictions as uncached decoding.
    """
    torch.manual_seed(2026)
    batch_size = 2
    prompt_len = 6
    decode_steps = 10
    total_seq_len = prompt_len + decode_steps

    tokens = torch.randint(0, test_model.vocab_size, (batch_size, total_seq_len))

    with torch.no_grad():
        logits_uncached, _ = test_model(tokens, start_pos=0, kv_caches=None)
        expected_next_tokens = torch.argmax(logits_uncached, dim=-1)

    kv_cache_mgr = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=test_model.max_seq_len,
        n_layers=test_model.depth,
        n_kv_heads=test_model.layers[0].attn.n_kv_heads,
        head_dim=test_model.layers[0].attn.head_dim,
        device=tokens.device,
        dtype=torch.float32,
    )
    layer_caches = [kv_cache_mgr.get_layer_cache(i) for i in range(test_model.depth)]

    # Prefill
    with torch.no_grad():
        logits_prefill, _ = test_model(tokens[:, :prompt_len], start_pos=0, kv_caches=layer_caches)
    kv_cache_mgr.advance(prompt_len)

    greedy_prefill = torch.argmax(logits_prefill, dim=-1)
    assert torch.equal(greedy_prefill, expected_next_tokens[:, :prompt_len])

    # Decode steps
    for step in range(decode_steps):
        curr_pos = kv_cache_mgr.cur_len
        token_step = tokens[:, curr_pos : curr_pos + 1]

        with torch.no_grad():
            logit_step, _ = test_model(token_step, start_pos=curr_pos, kv_caches=layer_caches)
        kv_cache_mgr.advance(1)

        greedy_step = torch.argmax(logit_step, dim=-1)
        expected_step = expected_next_tokens[:, curr_pos : curr_pos + 1]

        assert torch.equal(greedy_step, expected_step), (
            f"Greedy token mismatch at step {step} (position {curr_pos}): "
            f"cached {greedy_step} != uncached {expected_step}"
        )


def test_kv_cache_memory_and_reset(test_model: DecoderLLM):
    """
    Tests cache buffer dimensions, analytical memory calculation,
    and pointer reset functionality.
    """
    batch_size = 1
    max_seq_len = 256
    depth = 6
    n_kv_heads = 2
    head_dim = 32
    bytes_per_elem = 4  # float32

    cache = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_layers=depth,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # 1. Analytical byte calculation:
    # 2 buffers (K and V) * depth * (B * n_kv_heads * max_seq_len * head_dim) * 4 bytes
    expected_bytes = 2 * depth * (batch_size * n_kv_heads * max_seq_len * head_dim) * bytes_per_elem
    assert cache.total_bytes == expected_bytes
    assert cache.total_bytes == 786432  # Exactly 0.75 MB
    assert abs(cache.total_megabytes - 0.75) < 1e-4

    # 2. Advance pointer
    assert cache.cur_len == 0
    cache.advance(10)
    assert cache.cur_len == 10
    cache.advance(5)
    assert cache.cur_len == 15

    # 3. Reset
    cache.reset()
    assert cache.cur_len == 0
    k_buf, v_buf = cache.get_layer_cache(0)
    assert torch.all(k_buf == 0.0)
    assert torch.all(v_buf == 0.0)

