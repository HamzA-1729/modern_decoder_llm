"""
Unit Tests for Custom Byte-Level BPE Tokenizer:
1. Zero OOV (Out-Of-Vocabulary) property on arbitrary UTF-8 text and emoji.
2. Special token preservation (<|endoftext|>, <|pad|>).
3. BPE merge compression on repeated subwords.
4. Deterministic JSON serialization and deserialization round-trip.
"""

import os
import tempfile
import pytest
from modern_decoder_llm.tokenizer import Tokenizer


def test_base_vocab_and_zero_oov():
    """
    Even with zero learned merges, the 256 base byte vocabulary must guarantee
    lossless round-trip encoding/decoding of any Unicode string without OOV errors.
    """
    tokenizer = Tokenizer(vocab_size=4096)

    test_strings = [
        "Hello, World!",
        "Once upon a time, there was a little girl named Lily.",
        "Mathematics: \u03c0 \u2248 3.14159, e^{i\u03c0} + 1 = 0",
        "Emojis: 🚀🔥✨🤗🎉",
        "Non-Latin: \u0645\u0631\u062d\u0628\u0627 \u0628\u0643 (Arabic), \u3053\u3093\u306b\u3061\u306f (Japanese)",
    ]

    for s in test_strings:
        tokens = tokenizer.encode(s)
        # All tokens must fall within valid vocab bounds
        for tid in tokens:
            assert 0 <= tid < tokenizer.vocab_size

        decoded = tokenizer.decode(tokens)
        assert decoded == s, f"Round-trip failed for: {s!r} -> {decoded!r}"


def test_special_tokens():
    """Verify special tokens <|endoftext|> and <|pad|> map to designated IDs."""
    tokenizer = Tokenizer(vocab_size=4096)

    assert tokenizer.eos_id == 256
    assert tokenizer.pad_id == 257

    text = "Once upon a time.<|endoftext|>Then came tomorrow.<|pad|>"
    tokens = tokenizer.encode(text)

    # 256 and 257 must be present in token sequence
    assert 256 in tokens
    assert 257 in tokens

    decoded = tokenizer.decode(tokens)
    assert decoded == text


def test_bpe_training_and_compression():
    """Verify that training BPE merges frequent pairs and compresses sequences."""
    tokenizer = Tokenizer(vocab_size=512)

    corpus = [
        "Once upon a time, there was a little bird named Timmy.",
        "Timmy loved to fly high up in the blue sky every single day.",
        "One day, Timmy saw a shiny big red apple on a tall green tree.",
        "The little bird flew down to the apple and took a small bite.",
        "It was sweet and delicious, and Timmy was very happy.",
    ] * 20  # Repeat to establish high pair frequencies

    # Raw byte length before training
    raw_sample = corpus[0]
    raw_tokens = tokenizer.encode(raw_sample)
    raw_len = len(raw_tokens)

    # Train BPE merges
    tokenizer.train(corpus, target_vocab_size=320, min_freq=2)
    assert len(tokenizer.merges) > 0

    # Encoded sequence should now be compressed
    compressed_tokens = tokenizer.encode(raw_sample)
    compressed_len = len(compressed_tokens)

    assert compressed_len < raw_len, f"BPE should compress tokens: {compressed_len} vs {raw_len}"

    # Round-trip check on trained tokenizer
    decoded = tokenizer.decode(compressed_tokens)
    assert decoded == raw_sample


def test_tokenizer_save_and_load():
    """Verify tokenizer state can be saved to disk and loaded without loss."""
    tokenizer = Tokenizer(vocab_size=300)
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast blue bird sings a sweet gentle song.",
    ] * 10

    tokenizer.train(corpus, target_vocab_size=280, min_freq=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(save_path)
        assert os.path.exists(save_path)

        loaded_tok = Tokenizer.load(save_path)
        assert loaded_tok.vocab_size == tokenizer.vocab_size
        assert loaded_tok.merges == tokenizer.merges
        assert loaded_tok.special_tokens == tokenizer.special_tokens

        test_sentence = "The lazy dog and the quick blue bird."
        assert loaded_tok.encode(test_sentence) == tokenizer.encode(test_sentence)
        assert loaded_tok.decode(loaded_tok.encode(test_sentence)) == test_sentence

