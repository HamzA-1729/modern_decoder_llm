"""
Strict Causality Verification Test:
Verifies that the attention causal mask strictly prevents future tokens
from leaking into past outputs. Changing a future token must NOT alter
the output logits of preceding tokens.
"""

import pytest
import torch
from modern_decoder_llm.model import DecoderLLM


def test_strict_causality_masking():
    """
    Test: Pass sequence [x_0, x_1, x_2, x_3].
    Perturb token x_3 -> x_3_new.
    Assert that output representations at positions 0, 1, 2 remain identical.
    """
    torch.manual_seed(42)

    model = DecoderLLM(
        vocab_size=4096,
        max_seq_len=64,
        hidden_dim=128,
        depth=3,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=384,
    )
    model.eval()

    # Original prompt of 6 tokens
    prompt_a = torch.tensor([[10, 25, 42, 99, 105, 300]], dtype=torch.long)

    # Perturbed prompt: identical prefix of 5 tokens, but token 6 is changed
    prompt_b = torch.tensor([[10, 25, 42, 99, 105, 3000]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(prompt_a)
        logits_b, _ = model(prompt_b)

    # Output shapes: (1, 6, vocab_size)
    # Positions 0, 1, 2, 3, 4 MUST be identical between prompt_a and prompt_b
    prefix_logits_a = logits_a[:, :5, :]
    prefix_logits_b = logits_b[:, :5, :]

    # We assert strict numerical equality (allclose with tight atol)
    diff = (prefix_logits_a - prefix_logits_b).abs().max().item()
    print(f"Max causal divergence on past prefix: {diff:.2e}")

    assert torch.allclose(prefix_logits_a, prefix_logits_b, atol=1e-6), (
        f"Causality violation detected! Future token change altered past prefix logits by {diff:.2e}"
    )

    # Position 5 (the perturbed token) MUST be different
    last_diff = (logits_a[:, 5, :] - logits_b[:, 5, :]).abs().max().item()
    assert last_diff > 1e-3, "Perturbed token position produced identical logits!"

