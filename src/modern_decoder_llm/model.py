"""
Llama-3-Style Decoder-Only Language Model from Scratch.

Core Building Blocks:
1. RMSNorm: Root Mean Square Layer Normalization (bias-free).
2. RoPE: Rotary Position Embeddings using the standard `rotate_half` layout.
3. SwiGLU: Swish-Gated Linear Unit feed-forward network.
4. GroupedQueryAttention (GQA): 8 query heads, 2 KV heads, causal mask with position offset.
5. TransformerBlock: Pre-RMSNorm residual stack.
6. DecoderLLM: Stack of blocks with tied token embeddings and LM head (5.57M parameters).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Formula:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * gamma

    Why RMSNorm:
        Omits mean-subtraction, saving floating-point reductions while
        providing comparable training stability to standard LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable gain parameter gamma, initialized to 1.0
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean square along the last dimension (dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stable variance calculation if mixed precision is used
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int = 256,
    theta_base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine frequency tensors for Rotary Position Embeddings (RoPE).

    Uses the standard Llama-3 half-split layout:
        dim is split into two halves of size head_dim / 2.
        Frequencies theta_i = theta_base^(-2i / head_dim) are computed for head_dim / 2,
        then duplicated to form the full head_dim vector.

    Returns:
        cos, sin: Tensors of shape (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE."

    # Compute theta frequencies for half the dimensions: i in [0, head_dim // 2 - 1]
    half_dim = head_dim // 2
    i = torch.arange(half_dim, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta_base ** (2.0 * i / head_dim))  # (half_dim,)

    # Outer product with position indices m in [0, max_seq_len - 1]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freqs)  # (max_seq_len, half_dim)

    # In Llama's rotate_half convention, frequencies are concatenated with themselves
    # so that both halves of the embedding vector rotate synchronously:
    angles = torch.cat([angles, angles], dim=-1)  # (max_seq_len, head_dim)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates the half-split dimension pairs:
        [-x[..., head_dim/2:], x[..., :head_dim/2]]
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to Query or Key tensors.

    Formula:
        x_rotated = x * cos(pos) + rotate_half(x) * sin(pos)

    Args:
        x: Tensor of shape (B, n_heads, seq_len, head_dim)
        cos: Precomputed cos table (max_seq_len, head_dim)
        sin: Precomputed sin table (max_seq_len, head_dim)
        start_pos: Starting position index (for KV cache decoding)

    Returns:
        x_rotated: Tensor with identical shape (B, n_heads, seq_len, head_dim)
    """
    seq_len = x.shape[2]
    # Slice frequency table for active position range [start_pos : start_pos + seq_len]
    cos_active = cos[start_pos : start_pos + seq_len]  # (seq_len, head_dim)
    sin_active = sin[start_pos : start_pos + seq_len]  # (seq_len, head_dim)

    # Reshape to (1, 1, seq_len, head_dim) for broadcasting over (B, n_heads, seq_len, head_dim)
    cos_active = cos_active.unsqueeze(0).unsqueeze(1)
    sin_active = sin_active.unsqueeze(0).unsqueeze(1)

    return (x * cos_active) + (rotate_half(x) * sin_active)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network.

    Formula:
        SwiGLU(x) = ( SiLU(x * W_gate) * (x * W_up) ) * W_down

    All linear layers are bias-free following Llama architecture.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int = 768):
        super().__init__()
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, hidden_dim) -> (B, T, ffn_dim) -> (B, T, hidden_dim)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with support for prefill and cached generation.

    Configuration:
        n_heads = 8 (Query heads)
        n_kv_heads = 2 (Key/Value heads)
        Group ratio G = 8 / 2 = 4 (Each KV head is shared across 4 query heads).

    RoPE is applied to Q and K BEFORE storing K into the cache, ensuring cached keys
    never require re-rotation. Un-repeated K and V are stored in the cache to achieve
    the true 4x memory compression.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        assert n_heads * head_dim == hidden_dim, f"n_heads * head_dim ({n_heads * head_dim}) must equal hidden_dim ({hidden_dim})"

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_groups = n_heads // n_kv_heads

        # Projections (all bias=False)
        self.q_proj = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor (B, T, hidden_dim)
            cos, sin: Precomputed RoPE tables
            start_pos: Position offset for RoPE and cache indexing
            kv_cache: Optional tuple (k_cache, v_cache) of shape (B, n_kv_heads, max_seq_len, head_dim)

        Returns:
            output: Attention output (B, T, hidden_dim)
            updated_cache: Updated (k_cache, v_cache) tuple
        """
        b_sz, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x).view(b_sz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b_sz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # Shapes: q: (B, n_heads, T, head_dim), k/v: (B, n_kv_heads, T, head_dim)

        # 2. Apply RoPE to Q and K (before caching K!)
        q = apply_rotary_emb(q, cos, sin, start_pos=start_pos)
        k = apply_rotary_emb(k, cos, sin, start_pos=start_pos)

        # 3. KV Cache update (stores un-repeated K/V for true 4x memory savings)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Write new K and V into static buffer at position [start_pos : start_pos + seq_len]
            k_cache[:, :, start_pos : start_pos + seq_len, :] = k
            v_cache[:, :, start_pos : start_pos + seq_len, :] = v

            # Attend over all tokens written so far: [0 : start_pos + seq_len]
            k_active = k_cache[:, :, : start_pos + seq_len, :]
            v_active = v_cache[:, :, : start_pos + seq_len, :]
        else:
            k_active, v_active = k, v

        total_kv_len = k_active.shape[2]

        # 4. Grouped-Query Attention: expand 2 KV heads to 8 Query heads via repeat_interleave
        if self.num_groups > 1:
            k_active = torch.repeat_interleave(k_active, repeats=self.num_groups, dim=1)
            v_active = torch.repeat_interleave(v_active, repeats=self.num_groups, dim=1)
        # Shapes now: q: (B, 8, seq_len, head_dim), k_active/v_active: (B, 8, total_kv_len, head_dim)

        # 5. Scaled Dot-Product Attention with causal masking
        # Calculate raw attention scores
        scores = torch.matmul(q, k_active.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask when seq_len > 1 (prefill phase)
        if seq_len > 1:
            # Query index i sits at absolute position start_pos + i
            q_pos = torch.arange(start_pos, start_pos + seq_len, device=x.device).unsqueeze(1)
            k_pos = torch.arange(0, total_kv_len, device=x.device).unsqueeze(0)
            # Mask position where key index > query index (future tokens)
            causal_mask = k_pos > q_pos
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        output = torch.matmul(attn_weights, v_active)  # (B, 8, seq_len, head_dim)

        # 6. Transpose back and project out
        output = output.transpose(1, 2).contiguous().view(b_sz, seq_len, self.hidden_dim)
        return self.o_proj(output), kv_cache


class TransformerBlock(nn.Module):
    """
    Standard Pre-RMSNorm Transformer Decoder Block:
        x = x + GQA( RMSNorm(x) )
        x = x + SwiGLU( RMSNorm(x) )
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        ffn_dim: int = 768,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.attn = GroupedQueryAttention(hidden_dim, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn = SwiGLU(hidden_dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention
        attn_out, updated_cache = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        x = x + attn_out

        # Pre-norm SwiGLU FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x, updated_cache


class DecoderLLM(nn.Module):
    """
    Llama-3-Style Decoder Language Model.

    Features:
    - Weight tying: Output LM head shares weights with token embeddings table.
    - Scaled initialization: residual projections scaled by 1 / sqrt(2 * depth).
    - Precomputed RoPE tables for context lengths up to max_seq_len.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 256,
        hidden_dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        ffn_dim: int = 768,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.tie_weights = tie_weights

        # 1. Token Embeddings table
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # 2. Precompute RoPE frequency tables
        cos, sin = precompute_rope_freqs(head_dim, max_seq_len, theta_base=rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        # 3. Stack of Transformer Decoder Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        ])

        # 4. Final RMSNorm
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)

        # 5. Output LM head projection
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying: share embedding table with output projection
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # 6. Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialization scheme:
        - Normal N(0, 0.02^2)
        - Scale o_proj and down_proj by 1 / sqrt(2 * depth)
        """
        scaled_std = 0.02 / math.sqrt(2.0 * self.depth)
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                if "o_proj" in name or "down_proj" in name:
                    nn.init.normal_(p, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass for training (prefill) or generation (cached decoding).

        Args:
            token_ids: (B, T) LongTensor of token IDs
            start_pos: Current position index
            kv_caches: Optional list of (k_cache, v_cache) tuples for each layer

        Returns:
            logits: (B, T, vocab_size) prediction logits
            updated_caches: Updated list of KV caches
        """
        b_sz, seq_len = token_ids.shape
        assert start_pos + seq_len <= self.max_seq_len, f"Sequence length exceeds max_seq_len ({self.max_seq_len})"

        # 1. Embed input tokens
        h = self.tok_embeddings(token_ids)  # (B, T, hidden_dim)

        # 2. Pass through Decoder Stack
        updated_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, updated_cache_i = layer(
                h,
                self.rope_cos,
                self.rope_sin,
                start_pos=start_pos,
                kv_cache=cache_i,
            )
            if kv_caches is not None:
                updated_caches.append(updated_cache_i)

        # 3. Final normalization and language model head
        h = self.norm(h)
        logits = self.lm_head(h)  # (B, T, vocab_size)

        return logits, updated_caches if kv_caches is not None else None

