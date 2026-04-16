"""Near-duplicate detection for generated conversations.

Uses MinHash with n-gram shingling for fast approximate Jaccard similarity.
Pure Python — no external dependencies. Thread-safe for concurrent generation.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from typing import List

from .types import ConversationWithContext

# Large Mersenne prime for hash mixing
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _conversation_to_text(conversation: ConversationWithContext) -> str:
    """Extract all message content from a conversation into a single string."""
    return " ".join(
        entry.content for entry in conversation.conversations if entry.content
    )


def _ngram_shingles(text: str, n: int = 3) -> set[str]:
    """Extract word-level n-gram shingles from text.

    Word-level (not character-level) to be robust against minor
    punctuation/whitespace differences while catching semantic overlap.
    """
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _hash_shingle(shingle: str) -> int:
    """Deterministic 32-bit hash of a shingle string."""
    return struct.unpack("<I", hashlib.md5(shingle.encode("utf-8")).digest()[:4])[0]


class MinHashSignature:
    """A MinHash signature for approximate Jaccard similarity estimation."""

    __slots__ = ("_hashes",)

    def __init__(self, hashes: tuple[int, ...]):
        self._hashes = hashes

    def jaccard(self, other: MinHashSignature) -> float:
        """Estimate Jaccard similarity between two signatures."""
        if len(self._hashes) != len(other._hashes):
            raise ValueError("Signatures must have the same number of permutations")
        if not self._hashes:
            return 0.0
        matches = sum(a == b for a, b in zip(self._hashes, other._hashes))
        return matches / len(self._hashes)


class DuplicateDetector:
    """Detects near-duplicate conversations using MinHash signatures.

    Parameters:
        threshold: Jaccard similarity threshold above which a conversation
            is considered a duplicate. Default 0.7 (70% similar).
        num_perm: Number of hash permutations for MinHash. More = higher
            accuracy but slower. 128 is a good balance.
        shingle_size: Word n-gram size for shingling. 3 works well for
            conversational text.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        num_perm: int = 128,
        shingle_size: int = 3,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._signatures: List[MinHashSignature] = []
        self._lock = threading.Lock()

        # Pre-generate hash function coefficients: h(x) = (a*x + b) % prime
        import random

        rng = random.Random(42)  # deterministic for reproducibility
        self._a = tuple(rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm))
        self._b = tuple(rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm))

    def _compute_signature(self, text: str) -> MinHashSignature:
        """Compute MinHash signature for a text string."""
        shingles = _ngram_shingles(text, self.shingle_size)
        if not shingles:
            return MinHashSignature(tuple(_MAX_HASH for _ in range(self.num_perm)))

        shingle_hashes = [_hash_shingle(s) for s in shingles]

        min_hashes = []
        for i in range(self.num_perm):
            a, b = self._a[i], self._b[i]
            min_val = min((a * h + b) % _MERSENNE_PRIME for h in shingle_hashes)
            min_hashes.append(min_val)

        return MinHashSignature(tuple(min_hashes))

    def is_duplicate(self, conversation: ConversationWithContext) -> bool:
        """Check if a conversation is a near-duplicate of any previously seen one.

        Thread-safe: multiple workers can call this concurrently.

        Returns:
            True if the conversation is too similar to an existing one.
        """
        text = _conversation_to_text(conversation)
        sig = self._compute_signature(text)

        with self._lock:
            for existing in self._signatures:
                similarity = sig.jaccard(existing)
                if similarity >= self.threshold:
                    return True
            # Not a duplicate — register it
            self._signatures.append(sig)
            return False

    @property
    def seen_count(self) -> int:
        """Number of unique conversations registered."""
        with self._lock:
            return len(self._signatures)

    def reset(self) -> None:
        """Clear all stored signatures."""
        with self._lock:
            self._signatures.clear()
