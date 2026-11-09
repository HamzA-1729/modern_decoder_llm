"""
Custom Byte-Level Byte Pair Encoding (BPE) Tokenizer:
1. Base Vocabulary: 256 individual raw bytes (0..255) -> Guarantees zero Out-Of-Vocabulary (OOV) tokens.
2. Special Tokens: <|endoftext|> (256) and <|pad|> (257).
3. Learned Merges: Byte pairs merged greedily up to target vocab_size (e.g., 4096).
4. Deterministic JSON serialization and deserialization.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union


# Regex pattern for splitting text into pre-token chunks (GPT-2/Llama style)
# Keeps contractions, alphanumeric clusters, punctuation, and whitespace separated.
BPE_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """
    Byte-Level BPE Tokenizer with custom vocabulary size (default: 4096).
    """

    DEFAULT_ENDOFTEXT = "<|endoftext|>"
    DEFAULT_PAD = "<|pad|>"

    def __init__(
        self,
        vocab_size: int = 4096,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.vocab_size = vocab_size

        # Special token mapping
        if special_tokens is None:
            self.special_tokens: Dict[str, int] = {
                self.DEFAULT_ENDOFTEXT: 256,
                self.DEFAULT_PAD: 257,
            }
        else:
            self.special_tokens = dict(special_tokens)

        self.inverse_special_tokens: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

        # Merges: (token_a, token_b) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Merge ranks: (token_a, token_b) -> priority rank (lower = merged earlier)
        self.merge_ranks: Dict[Tuple[int, int], int] = {}

        # Token ID to bytes mapping
        self.vocab: Dict[int, bytes] = {}
        self._build_base_vocab()

    def _build_base_vocab(self) -> None:
        """Initializes the base 256 byte vocabulary."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Add special tokens
        for name, idx in self.special_tokens.items():
            self.vocab[idx] = name.encode("utf-8")

    @property
    def pad_id(self) -> int:
        return self.special_tokens[self.DEFAULT_PAD]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[self.DEFAULT_ENDOFTEXT]

    @property
    def bos_id(self) -> int:
        # For Llama-style decoder, eos/endoftext often doubles as bos
        return self.eos_id

    def train(self, texts: List[str], target_vocab_size: Optional[int] = None, min_freq: int = 2) -> None:
        """
        Learns BPE merges from a corpus of raw strings.
        
        Args:
            texts: List of training sentences/documents.
            target_vocab_size: Total vocabulary target (defaults to self.vocab_size).
            min_freq: Minimum frequency threshold to consider a pair for merging.
        """
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size

        # 1. Reset vocabulary and merges
        self._build_base_vocab()
        self.merges.clear()
        self.merge_ranks.clear()

        # 2. Pre-tokenize and count word frequencies
        word_counts: Counter[Tuple[int, ...]] = Counter()
        for text in texts:
            for match in BPE_SPLIT_PATTERN.finditer(text):
                chunk = match.group()
                byte_tuple = tuple(chunk.encode("utf-8", errors="replace"))
                word_counts[byte_tuple] += 1

        # Words representation as a list of mutable token lists with their frequency
        words: List[List[int]] = [list(w) for w in word_counts.keys()]
        counts: List[int] = list(word_counts.values())

        # Build inverted index mapping pair -> set of word indices
        pair_to_words: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        for w_idx, (word, freq) in enumerate(zip(words, counts)):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_to_words[pair].add(w_idx)
                pair_counts[pair] += freq

        next_token_id = max(max(self.special_tokens.values()) + 1, 258)
        num_merges = target_vocab_size - next_token_id

        rank = 0
        for _ in range(num_merges):
            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair, best_freq = max(pair_counts.items(), key=lambda x: x[1])
            if best_freq < min_freq:
                break

            # Register merge
            self.merges[best_pair] = next_token_id
            self.merge_ranks[best_pair] = rank
            rank += 1

            # Update vocab table: byte representation of the new token
            token_a, token_b = best_pair
            self.vocab[next_token_id] = self.vocab[token_a] + self.vocab[token_b]

            # Invalidate best_pair in pair_counts
            del pair_counts[best_pair]

            # Affected words that contain best_pair
            affected_words = list(pair_to_words[best_pair])
            del pair_to_words[best_pair]

            for w_idx in affected_words:
                word = words[w_idx]
                freq = counts[w_idx]

                # Find all occurrences of best_pair in word and merge them
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        # Before removing old pairs, decrement their counts
                        if i > 0:
                            prev_pair = (word[i - 1], word[i])
                            pair_counts[prev_pair] -= freq
                            pair_to_words[prev_pair].discard(w_idx)
                            if pair_counts[prev_pair] <= 0:
                                pair_counts.pop(prev_pair, None)

                        if i + 2 < len(word):
                            next_pair = (word[i + 1], word[i + 2])
                            pair_counts[next_pair] -= freq
                            pair_to_words[next_pair].discard(w_idx)
                            if pair_counts[next_pair] <= 0:
                                pair_counts.pop(next_pair, None)

                        # Form new token
                        new_word.append(next_token_id)

                        # Increment new pairs created with adjacent neighbors
                        if new_word and len(new_word) >= 2:
                            new_left_pair = (new_word[-2], new_word[-1])
                            pair_counts[new_left_pair] += freq
                            pair_to_words[new_left_pair].add(w_idx)

                        i += 2
                    else:
                        new_word.append(word[i])
                        if len(new_word) >= 2 and i > 0 and word[i - 1] == next_token_id:
                            new_right_pair = (new_word[-2], new_word[-1])
                            pair_counts[new_right_pair] += freq
                            pair_to_words[new_right_pair].add(w_idx)
                        i += 1

                words[w_idx] = new_word

            next_token_id += 1

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Encodes a single pre-tokenized chunk using learned merges."""
        tokens = list(chunk_bytes)
        if len(tokens) <= 1:
            return tokens

        while len(tokens) >= 2:
            # Find the adjacent pair in tokens that has the lowest merge rank
            min_rank = float("inf")
            best_pair = None
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair, None)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_pair is None:
                # No more possible merges for this chunk
                break

            # Apply this merge to all matching adjacent pairs in the token list
            merged_token = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encodes a string into a list of integer token IDs.
        
        Guarantees:
        - Exact byte preservation (zero OOV).
        - Special tokens (<|endoftext|>, <|pad|>) are preserved verbatim when permitted.
        """
        if allowed_special is None:
            allowed_special = set(self.special_tokens.keys())

        # If special tokens are present, split around them using a regex
        if allowed_special:
            escaped_specials = [re.escape(s) for s in allowed_special if s in self.special_tokens]
            if escaped_specials:
                special_pattern = re.compile(f"({'|'.join(escaped_specials)})")
                segments = special_pattern.split(text)
            else:
                segments = [text]
        else:
            segments = [text]

        encoded_tokens: List[int] = []
        for segment in segments:
            if not segment:
                continue

            if segment in self.special_tokens and segment in allowed_special:
                encoded_tokens.append(self.special_tokens[segment])
            else:
                for match in BPE_SPLIT_PATTERN.finditer(segment):
                    chunk = match.group()
                    encoded_tokens.extend(self._encode_chunk(chunk.encode("utf-8", errors="replace")))

        return encoded_tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of integer token IDs back into a Unicode string.
        Byte-fallback handles arbitrary bytes safely via replace errors.
        """
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
            else:
                # Fallback for unexpected IDs: ignore or render unknown marker
                byte_chunks.append(b"")

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        """Serializes tokenizer configuration and merges to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [
                {"pair": list(pair), "id": tid, "rank": self.merge_ranks[pair]}
                for pair, tid in self.merges.items()
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Tokenizer":
        """Loads a saved tokenizer from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data.get("special_tokens"),
        )
        tokenizer._build_base_vocab()

        for item in data["merges"]:
            pair = (int(item["pair"][0]), int(item["pair"][1]))
            tid = int(item["id"])
            rank = int(item["rank"])
            tokenizer.merges[pair] = tid
            tokenizer.merge_ranks[pair] = rank

            # Rebuild vocab byte representation
            token_a, token_b = pair
            tokenizer.vocab[tid] = tokenizer.vocab[token_a] + tokenizer.vocab[token_b]

        return tokenizer

