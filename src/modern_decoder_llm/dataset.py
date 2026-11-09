"""
Memory-Mapped Dataset Pipeline for TinyStories Pretraining:
1. Pretokenization: Converts raw text stories into contiguous token streams with <|endoftext|> separators.
2. Binary Memory-Mapping: Stores tokens on disk as compact uint16 arrays (2 bytes per token for vocab < 65,536).
3. Zero-Copy Slicing: Uses np.memmap for instant, low-RAM window sampling during training and validation.
4. Causal Pair Generation: Returns (x, y) where x = tokens[i : i + seq_len] and y = tokens[i + 1 : i + seq_len + 1].
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from modern_decoder_llm.tokenizer import Tokenizer


class MemmapTokenDataset(Dataset):
    """
    PyTorch Dataset backed by a memory-mapped uint16 binary file.
    
    Memory Efficiency:
    - uint16 stores token IDs in [0, 65535]. For vocab_size=4096, this cuts
      storage by 75% compared to int64 and 50% compared to int32.
    - np.memmap(..., mode='r') streams segments from disk on-demand via the OS page cache.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int = 256,
    ):
        super().__init__()
        assert os.path.exists(bin_path), f"Binary dataset file not found: {bin_path}"
        self.bin_path = bin_path
        self.seq_len = seq_len

        # Memory-map the flat token array
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Number of non-overlapping or contiguous windows
        # Each sample requires (seq_len + 1) tokens: x is [:seq_len], y is [1:seq_len + 1]
        assert self.total_tokens > self.seq_len + 1, (
            f"Dataset has {self.total_tokens} tokens, but requires at least {self.seq_len + 2}"
        )
        self.num_samples = (self.total_tokens - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1

        # Read slice from memory-mapped disk buffer
        chunk = self.data[start_idx:end_idx].astype(np.int64)

        # Input x: tokens 0..seq_len - 1
        # Target y: tokens 1..seq_len (next token prediction)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])

        return x, y


def build_pretokenized_binary(
    texts: list[str],
    tokenizer: Tokenizer,
    output_bin_path: str,
    add_eos: bool = True,
) -> int:
    """
    Tokenizes a list of documents and writes them to a compact uint16 binary file.
    
    Args:
        texts: List of raw story strings.
        tokenizer: Initialized or trained Tokenizer instance.
        output_bin_path: Filepath where the .bin file will be written.
        add_eos: Whether to append <|endoftext|> between documents.
        
    Returns:
        total_tokens_written: Total number of tokens written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    eos_id = tokenizer.eos_id

    # Temporary list to accumulate token IDs before flushing to disk
    all_tokens: list[int] = []

    print(f"Tokenizing {len(texts):,} documents into {output_bin_path}...")
    for text in tqdm(texts, desc="Tokenizing"):
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        if add_eos:
            all_tokens.append(eos_id)

    token_array = np.array(all_tokens, dtype=np.uint16)
    with open(output_bin_path, "wb") as f:
        f.write(token_array.tobytes())

    total_tokens = len(token_array)
    file_size_mb = os.path.getsize(output_bin_path) / (1024 * 1024)
    print(
        f"Saved {total_tokens:,} tokens to {output_bin_path} "
        f"({file_size_mb:.2f} MB, uint16 format)"
    )
    return total_tokens


def prepare_tinystories_data(
    output_dir: str = "data",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    train_docs: int = 50000,
    val_docs: int = 2000,
    vocab_size: int = 4096,
) -> Tuple[str, str, Tokenizer]:
    """
    End-to-end pipeline:
    1. Loads or trains the custom Byte-Level BPE tokenizer.
    2. Downloads or streams TinyStories from HuggingFace datasets (with offline fallback).
    3. Writes train.bin and val.bin memory-mapped files.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")

    # Step 1: Initialize or load tokenizer
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        print(f"Tokenizer not found at {tokenizer_path}. Preparing corpus for tokenizer training...")
        # Attempt to load TinyStories from datasets
        train_stories = []
        try:
            from datasets import load_dataset
            print("Fetching TinyStories subset from Hugging Face...")
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            for i, item in enumerate(ds):
                if i >= train_docs:
                    break
                train_stories.append(item["text"])
        except Exception as e:
            print(f"Warning: Could not fetch TinyStories online ({e}). Using synthetic educational corpus.")
            train_stories = [
                "Once upon a time, there was a little bird named Timmy. Timmy loved to fly across the green forest.",
                "One sunny morning, a girl named Lily found a magical shiny pebble next to the sparkling river.",
                "The friendly bear gathered sweet honey and shared it with all his kind forest friends.",
                "A fast rabbit learned that being patient and gentle brought more joy than running in a hurry.",
            ] * 1000

        tokenizer = Tokenizer(vocab_size=vocab_size)
        print(f"Training Byte-Level BPE tokenizer on {len(train_stories):,} stories...")
        tokenizer.train(train_stories[:5000], target_vocab_size=vocab_size, min_freq=2)
        tokenizer.save(tokenizer_path)
        print(f"Saved trained tokenizer to {tokenizer_path}")

    # Step 2: Build train.bin and val.bin if they do not exist
    if not os.path.exists(train_bin_path):
        print(f"Building {train_bin_path}...")
        try:
            from datasets import load_dataset
            ds_train = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            raw_train = []
            for i, item in enumerate(ds_train):
                if i >= train_docs:
                    break
                raw_train.append(item["text"])
        except Exception:
            raw_train = [
                "Once upon a time, there was a little girl who lived in a cozy village.",
                "Every afternoon, she baked warm sweet bread and carried it to her grandmother.",
            ] * (train_docs // 2)

        build_pretokenized_binary(raw_train, tokenizer, train_bin_path)

    if not os.path.exists(val_bin_path):
        print(f"Building {val_bin_path}...")
        try:
            from datasets import load_dataset
            ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
            raw_val = []
            for i, item in enumerate(ds_val):
                if i >= val_docs:
                    break
                raw_val.append(item["text"])
        except Exception:
            raw_val = [
                "The little puppy hopped across the grass and caught the colorful red ball.",
                "His owner smiled warmly and patted him on the head happily.",
            ] * (val_docs // 2)

        build_pretokenized_binary(raw_val, tokenizer, val_bin_path)

    return train_bin_path, val_bin_path, tokenizer

