"""
Preallocated Static KV Cache:
Manages persistent Key/Value buffers across Transformer decoder layers.
Eliminates prefix recomputation during autoregressive decoding.

Memory Architecture:
- Preallocated buffers per layer: (B, n_kv_heads, max_seq_len, head_dim)
- Tracks active position via an integer `cur_len` pointer.
- Slicing [:, :, :cur_len + T] produces a zero-copy memory view.
"""

from typing import List, Tuple, Optional
import torch


class StaticKVCache:
    """
    Manages static Key-Value tensor caches for all layers of DecoderLLM.

    Key Features:
    1. Preallocated: Zero memory allocations or torch.cat copying during token generation.
    2. Un-repeated storage: Stores raw n_kv_heads (2) instead of n_heads (8),
       achieving a 4x reduction in KV cache memory footprint.
    3. Global Pointer: Tracks `cur_len` across forward passes.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Global context length pointer
        self.cur_len: int = 0

        # Preallocate static (k, v) buffer pairs for each layer
        # Shape per buffer: (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(n_layers):
            k_buffer = torch.zeros(
                (batch_size, n_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            v_buffer = torch.zeros(
                (batch_size, n_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            self.caches.append((k_buffer, v_buffer))

    def get_layer_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the preallocated (k, v) buffer tuple for layer `layer_idx`."""
        return self.caches[layer_idx]

    def advance(self, num_tokens: int) -> None:
        """
        Advances the global context length pointer after all layers have finished
        their forward pass. (Advancing layer-by-layer would cause subsequent layers
        to see a shifted length).
        """
        self.cur_len += num_tokens
        assert self.cur_len <= self.max_seq_len, (
            f"KV Cache overflow: cur_len ({self.cur_len}) exceeds max_seq_len ({self.max_seq_len})"
        )

    def reset(self) -> None:
        """Reset the cache pointer for a new sequence generation."""
        self.cur_len = 0
        for k_buf, v_buf in self.caches:
            k_buf.zero_()
            v_buf.zero_()

    @property
    def total_bytes(self) -> int:
        """Returns total memory footprint of the KV cache in bytes."""
        total = 0
        for k_buf, v_buf in self.caches:
            total += k_buf.nelement() * k_buf.element_size()
            total += v_buf.nelement() * v_buf.element_size()
        return total

    @property
    def total_megabytes(self) -> float:
        """Returns total memory footprint in megabytes (MB)."""
        return self.total_bytes / (1024.0 * 1024.0)

