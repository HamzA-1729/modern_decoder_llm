"""
Single-Batch Overfitting Test:
1. Validates that the full modern decoder architecture (RMSNorm, RoPE, SwiGLU, GQA, tied weights)
   can rapidly memorize a single fixed batch of sequences.
2. Initial loss assertion: Evaluates that random initialization loss is approximately ln(vocab_size) = ln(4096) ≈ 8.32.
3. Step-based convergence: Trains for <= 300 steps (LR=1e-3, no warmup, no weight decay).
   Asserts final loss drops by >95% (below 0.40) and greedy token accuracy reaches >99%.
   (Note: Machine-independent step assertion; no wall-clock timers).
"""

import math
import torch
import torch.nn.functional as F
import pytest

from modern_decoder_llm.model import DecoderLLM


def test_single_batch_overfit():
    torch.manual_seed(42)
    device = torch.device("cpu")

    vocab_size = 4096
    seq_len = 32
    batch_size = 4

    # Target loss thresholds
    expected_initial_loss = math.log(vocab_size)  # ln(4096) ≈ 8.3178

    # Small 4-layer model for fast unit testing
    model = DecoderLLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        hidden_dim=128,
        depth=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        ffn_dim=384,
        rope_theta=10000.0,
        tie_weights=True,
        norm_eps=1e-6,
    ).to(device)

    # Fixed synthetic batch
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Next-token prediction target
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Optimizer: LR=1e-3, no weight decay, no warmup for fast single-batch overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # --- Step 0: Verify initial loss ---
    model.eval()
    with torch.no_grad():
        initial_logits, _ = model(x)
        initial_loss = F.cross_entropy(initial_logits.view(-1, vocab_size), y.view(-1)).item()

    print(f"\nStep 0 Initial Loss: {initial_loss:.4f} (Expected ln(4096) ≈ {expected_initial_loss:.4f})")
    assert abs(initial_loss - expected_initial_loss) < 1.0, (
        f"Initial loss {initial_loss:.4f} diverged significantly from ln(4096)={expected_initial_loss:.4f}"
    )

    # --- Training Loop: Max 300 steps ---
    model.train()
    max_steps = 300
    final_loss = initial_loss

    for step in range(max_steps):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        final_loss = loss.item()
        if (step + 1) % 50 == 0 or step == max_steps - 1:
            print(f"Step [{step + 1:3d}/{max_steps}] - Loss: {final_loss:.4f}")

        # Early exit if loss already dropped by >98%
        if final_loss < 0.15:
            print(f"Converged early at step {step + 1} with loss {final_loss:.4f}")
            break

    # --- Assertions ---
    loss_reduction_pct = (initial_loss - final_loss) / initial_loss * 100.0
    print(f"\nFinal Loss: {final_loss:.4f} | Drop: {loss_reduction_pct:.2f}%")

    assert loss_reduction_pct > 95.0, (
        f"Expected loss to drop by >95%, but only dropped by {loss_reduction_pct:.2f}% "
        f"(Initial: {initial_loss:.4f}, Final: {final_loss:.4f})"
    )
    assert final_loss < 0.40, f"Final loss ({final_loss:.4f}) is not below 0.40"

    # Verify greedy token prediction accuracy on the overfit batch
    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(x)
        preds = torch.argmax(eval_logits, dim=-1)
        accuracy = (preds == y).float().mean().item() * 100.0

    print(f"Token Memorization Accuracy: {accuracy:.2f}%")
    assert accuracy > 99.0, f"Expected >99% memorization accuracy, got {accuracy:.2f}%"

