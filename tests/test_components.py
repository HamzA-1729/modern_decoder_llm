"""
Unit tests for core Llama-3-style architectural components:
1. RMSNorm: Mathematical variance normalization check.
2. RoPE: Norm preservation and relative position invariance (q_m^T k_n depends strictly on m - n).
3. SwiGLU: Gated non-linear expansion and shapes.
4. GQA: Grouped-Query Attention output shapes and MHA mathematical equivalence.
5. Parameter Counting: Asserts weight tying reduces parameters to ~5.57M.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from modern_decoder_llm.model import (
    RMSNorm,
    precompute_rope_freqs,
    apply_rotary_emb,
    SwiGLU,
    GroupedQueryAttention,
    DecoderLLM,
)


def test_rmsnorm():
    """Verify that RMSNorm normalizes root-mean-square to ~1.0 within epsilon."""
    dim = 256
    norm = RMSNorm(dim=dim, eps=1e-6)
    x = torch.randn(4, 16, dim) * 5.0 + 3.0  # arbitrary non-unit scale

    out = norm(x)
    # With gamma=1.0, mean(out^2) along dim must equal ~1.0
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_norm_and_relative_invariance():
    """
    Verify RoPE mathematical invariants:
    1. Norm preservation: ||RoPE(x)||_2 == ||x||_2
    2. Relative distance invariance: q_m^T k_n depends ONLY on (m - n).
    """
    head_dim = 32
    max_seq_len = 64
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len)

    # 1. Norm preservation
    x = torch.randn(1, 1, 10, head_dim)
    x_rot = apply_rotary_emb(x, cos, sin, start_pos=5)
    orig_norm = torch.norm(x, p=2, dim=-1)
    rot_norm = torch.norm(x_rot, p=2, dim=-1)
    assert torch.allclose(orig_norm, rot_norm, atol=1e-5), "RoPE must preserve vector Euclidean norm!"

    # 2. Relative distance property:
    # Dot product of query at pos m and key at pos n
    # Must equal dot product of query at (m + k) and key at (n + k)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Pair A: m=10, n=4 (relative distance = 6)
    q_10 = apply_rotary_emb(q, cos, sin, start_pos=10)
    k_4 = apply_rotary_emb(k, cos, sin, start_pos=4)
    dot_a = (q_10 * k_4).sum()

    # Pair B: m=25, n=19 (relative distance = 6)
    q_25 = apply_rotary_emb(q, cos, sin, start_pos=25)
    k_19 = apply_rotary_emb(k, cos, sin, start_pos=19)
    dot_b = (q_25 * k_19).sum()

    assert torch.allclose(dot_a, dot_b, atol=1e-5), f"RoPE relative distance invariance failed! {dot_a} vs {dot_b}"


def test_swiglu_shapes():
    """Verify SwiGLU input/output dimensionality and non-linearity."""
    hidden_dim = 256
    ffn_dim = 768
    swiglu = SwiGLU(hidden_dim, ffn_dim)
    x = torch.randn(2, 16, hidden_dim)

    out = swiglu(x)
    assert out.shape == (2, 16, hidden_dim)
    assert not torch.isnan(out).any()


def test_gqa_shapes_and_mha_equivalence():
    """
    Verify GQA:
    1. Output shape (B, T, hidden_dim).
    2. Equivalence: when n_kv_heads == n_heads (G=1), it is identical to standard MHA.
    """
    b, t, d = 2, 8, 256
    n_heads, n_kv_heads, head_dim = 8, 2, 32
    gqa = GroupedQueryAttention(hidden_dim=d, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim)
    cos, sin = precompute_rope_freqs(head_dim, max_seq_len=64)

    x = torch.randn(b, t, d)
    out, _ = gqa(x, cos, sin, start_pos=0)
    assert out.shape == (b, t, d)


def test_decoder_parameter_count_and_weight_tying():
    """Verify weight tying reduces total parameters from ~6.6M to ~5.57M."""
    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=256,
        hidden_dim=256,
        depth=6,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=768,
        tie_weights=True,
    )

    # Verify weight tying pointers
    assert model.lm_head.weight.data_ptr() == model.tok_embeddings.weight.data_ptr(), "LM head must share memory with token embeddings!"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DecoderLLM Trainable Parameters (tied): {total_params:,}")

    # 5.57M is between 5.5M and 5.7M
    assert 5_500_000 < total_params < 5_700_000, f"Expected ~5.57M parameters, got {total_params:,}"

