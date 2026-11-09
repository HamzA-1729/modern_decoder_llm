"""
Modern Decoder LLM: Llama-3-style Decoder-Only Architecture from Scratch.
"""

from modern_decoder_llm.model import DecoderLLM
from modern_decoder_llm.kv_cache import StaticKVCache
from modern_decoder_llm.sampler import generate_with_cache, generate_without_cache, sample_next_token
from modern_decoder_llm.tokenizer import Tokenizer
from modern_decoder_llm.dataset import MemmapTokenDataset
from modern_decoder_llm.train import configure_optimizers, train

__all__ = [
    "DecoderLLM",
    "StaticKVCache",
    "Tokenizer",
    "MemmapTokenDataset",
    "configure_optimizers",
    "train",
    "generate_with_cache",
    "generate_without_cache",
    "sample_next_token",
]
