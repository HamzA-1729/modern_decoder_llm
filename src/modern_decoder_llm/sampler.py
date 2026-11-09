"""
Autoregressive Generation & Sampling:
1. Temperature scaling & Top-p (Nucleus) sampling.
2. Fast generation with preallocated StaticKVCache.
3. Baseline generation without cache (for throughput benchmarking and correctness testing).
"""

from typing import Optional, List
import torch
import torch.nn.functional as F

from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.kv_cache import StaticKVCache


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Selects the next token ID from output logits using Temperature + Top-p (Nucleus) sampling.

    Args:
        logits: Tensor of shape (B, vocab_size)
        temperature: Temperature scaling factor (0.0 = greedy argmax)
        top_p: Nucleus cumulative probability threshold (1.0 = full distribution)

    Returns:
        next_tokens: LongTensor of shape (B, 1)
    """
    # 1. Greedy decoding (special case: temperature == 0.0)
    if temperature <= 1e-5:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature scaling
    logits = logits / temperature

    # 3. Compute probabilities
    probs = F.softmax(logits, dim=-1)

    # 4. Top-p (Nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Mask tokens whose cumulative probability exceeds top_p
        # Shift mask right by 1 so we always keep at least the top token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter the mask back to the original vocabulary indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        probs = probs.masked_fill(indices_to_remove, 0.0)
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # 5. Multinomial categorical sampling
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens


@torch.no_grad()
def generate_with_cache(
    model: DecoderLLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressive generation using the preallocated StaticKVCache.

    Phases:
    1. Prefill: Prompt tokens (T > 1) pass through the model in one forward pass,
       populating the KV cache up to prompt_len.
    2. Decode: Only the single newly sampled token (T = 1) is passed per step.
       The KV cache is updated at index cur_len and read up to cur_len + 1.
    """
    model.eval()
    batch_size, prompt_len = prompt_tokens.shape
    device = prompt_tokens.device
    total_len = prompt_len + max_new_tokens
    assert total_len <= model.max_seq_len, f"Total generation length ({total_len}) exceeds max_seq_len ({model.max_seq_len})"

    # 1. Allocate StaticKVCache once
    kv_cache_mgr = StaticKVCache(
        batch_size=batch_size,
        max_seq_len=model.max_seq_len,
        n_layers=model.depth,
        n_kv_heads=model.layers[0].attn.n_kv_heads,
        head_dim=model.layers[0].attn.head_dim,
        device=device,
        dtype=prompt_tokens.dtype if prompt_tokens.is_floating_point() else torch.float32,
    )

    layer_caches = [kv_cache_mgr.get_layer_cache(i) for i in range(model.depth)]

    # 2. Prefill Phase (process full prompt at start_pos = 0)
    logits, _ = model(prompt_tokens, start_pos=0, kv_caches=layer_caches)
    kv_cache_mgr.advance(prompt_len)

    # Sample first new token from the last logit of the prompt
    next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)

    generated_tokens = [next_token]

    # 3. Decode Phase (single-token generation T = 1)
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        current_pos = kv_cache_mgr.cur_len
        logits, _ = model(next_token, start_pos=current_pos, kv_caches=layer_caches)
        kv_cache_mgr.advance(1)

        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
        generated_tokens.append(next_token)

    # Concatenate prompt with all newly generated tokens
    output_tokens = torch.cat([prompt_tokens] + generated_tokens, dim=-1)
    return output_tokens


@torch.no_grad()
def generate_without_cache(
    model: DecoderLLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Baseline generation WITHOUT KV cache.
    Recomputes the full prefix through all projections and FFN layers at every single step.
    Used to benchmark throughput gains and verify cache correctness.
    """
    model.eval()
    current_tokens = prompt_tokens.clone()

    for _ in range(max_new_tokens):
        if current_tokens.shape[1] >= model.max_seq_len:
            break

        # Pass full sequence through model with no cache (start_pos = 0)
        logits, _ = model(current_tokens, start_pos=0, kv_caches=None)

        next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p)
        current_tokens = torch.cat([current_tokens, next_token], dim=-1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return current_tokens

