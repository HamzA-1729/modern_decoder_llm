"""
Unit Tests for Memory-Mapped Dataset & Tokenizer Pipeline:
1. Validates uint16 binary memmap reading and zero-copy slicing.
2. Validates causal alignment: y[t] == x[t + 1] (next token prediction target).
3. Validates end-to-end tokenization and binary generation.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from modern_decoder_llm.tokenizer import Tokenizer
from modern_decoder_llm.dataset import MemmapTokenDataset, build_pretokenized_binary


def test_memmap_dataset_causal_pairs():
    """
    Verifies that MemmapTokenDataset yields exact causal pairs:
    x = tokens[t : t + seq_len]
    y = tokens[t + 1 : t + seq_len + 1]
    """
    seq_len = 4
    # Create synthetic token sequence: 0, 1, 2, 3, 4, 5, 6, 7, 8
    raw_tokens = np.arange(13, dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test.bin")
        raw_tokens.tofile(bin_path)

        dataset = MemmapTokenDataset(bin_path=bin_path, seq_len=seq_len)
        # Expected samples: (13 - 1) // 4 = 3 samples
        assert len(dataset) == 3

        # Sample 0
        x0, y0 = dataset[0]
        assert x0.shape == (seq_len,)
        assert y0.shape == (seq_len,)
        assert x0.dtype == torch.long
        assert y0.dtype == torch.long

        assert torch.equal(x0, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(y0, torch.tensor([1, 2, 3, 4]))
        # Strict causal target condition: y[:-1] == x[1:]
        assert torch.equal(y0[:-1], x0[1:])

        # Sample 1
        x1, y1 = dataset[1]
        assert torch.equal(x1, torch.tensor([4, 5, 6, 7]))
        assert torch.equal(y1, torch.tensor([5, 6, 7, 8]))
        assert torch.equal(y1[:-1], x1[1:])


def test_build_pretokenized_binary():
    """Verifies building a binary dataset from text documents."""
    tokenizer = Tokenizer(vocab_size=4096)
    texts = [
        "First story about a happy cat.",
        "Second story about a playful dog.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "data.bin")
        total_tokens = build_pretokenized_binary(
            texts=texts,
            tokenizer=tokenizer,
            output_bin_path=bin_path,
            add_eos=True,
        )

        assert os.path.exists(bin_path)
        # Check file size matches total_tokens * 2 bytes (uint16)
        assert os.path.getsize(bin_path) == total_tokens * 2

        # Verify dataset loads the binary file without error
        dataset = MemmapTokenDataset(bin_path=bin_path, seq_len=8)
        assert len(dataset) > 0
        x, y = dataset[0]
        assert x.shape == (8,)
        assert y.shape == (8,)
        assert torch.equal(y[:-1], x[1:])

