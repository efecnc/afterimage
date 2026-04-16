"""Tests for near-duplicate detection."""

import pytest

from afterimage.dedup import (
    DuplicateDetector,
    MinHashSignature,
    _conversation_to_text,
    _ngram_shingles,
)
from afterimage.types import ConversationEntry, ConversationWithContext, Role


def _make_conversation(messages: list[str]) -> ConversationWithContext:
    """Helper to build a ConversationWithContext from alternating user/assistant messages."""
    entries = []
    for i, msg in enumerate(messages):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        entries.append(ConversationEntry(role=role, content=msg))
    return ConversationWithContext(conversations=entries)


class TestNgramShingles:
    def test_basic(self):
        shingles = _ngram_shingles("the quick brown fox jumps", n=3)
        assert "the quick brown" in shingles
        assert "quick brown fox" in shingles
        assert "brown fox jumps" in shingles
        assert len(shingles) == 3

    def test_short_text(self):
        shingles = _ngram_shingles("hello world", n=3)
        assert shingles == {"hello world"}

    def test_empty(self):
        assert _ngram_shingles("", n=3) == set()

    def test_case_insensitive(self):
        s1 = _ngram_shingles("The Quick Brown", n=3)
        s2 = _ngram_shingles("the quick brown", n=3)
        assert s1 == s2


class TestConversationToText:
    def test_extracts_all_content(self):
        conv = _make_conversation(["Hello there", "Hi, how can I help?"])
        text = _conversation_to_text(conv)
        assert "Hello there" in text
        assert "Hi, how can I help?" in text


class TestMinHashSignature:
    def test_identical_texts_high_similarity(self):
        detector = DuplicateDetector(threshold=0.9, num_perm=256)
        sig1 = detector._compute_signature("the quick brown fox jumps over the lazy dog")
        sig2 = detector._compute_signature("the quick brown fox jumps over the lazy dog")
        assert sig1.jaccard(sig2) == 1.0

    def test_different_texts_low_similarity(self):
        detector = DuplicateDetector(threshold=0.9, num_perm=256)
        sig1 = detector._compute_signature(
            "the quick brown fox jumps over the lazy dog"
        )
        sig2 = detector._compute_signature(
            "quantum mechanics describes the behavior of subatomic particles"
        )
        assert sig1.jaccard(sig2) < 0.3

    def test_similar_texts_medium_similarity(self):
        detector = DuplicateDetector(threshold=0.9, num_perm=256)
        sig1 = detector._compute_signature(
            "how do I train a language model on custom data"
        )
        sig2 = detector._compute_signature(
            "how do I train a language model on my own data"
        )
        similarity = sig1.jaccard(sig2)
        assert 0.3 < similarity < 0.95

    def test_mismatched_lengths_raise(self):
        s1 = MinHashSignature((1, 2, 3))
        s2 = MinHashSignature((1, 2))
        with pytest.raises(ValueError):
            s1.jaccard(s2)


class TestDuplicateDetector:
    def test_first_conversation_never_duplicate(self):
        detector = DuplicateDetector(threshold=0.7)
        conv = _make_conversation(["What is Python?", "Python is a programming language."])
        assert not detector.is_duplicate(conv)
        assert detector.seen_count == 1

    def test_identical_conversation_is_duplicate(self):
        detector = DuplicateDetector(threshold=0.7)
        conv = _make_conversation(["What is Python?", "Python is a programming language."])
        assert not detector.is_duplicate(conv)
        assert detector.is_duplicate(conv)
        assert detector.seen_count == 1  # duplicate was not registered

    def test_very_different_conversation_not_duplicate(self):
        detector = DuplicateDetector(threshold=0.7)
        conv1 = _make_conversation([
            "What is Python?",
            "Python is a high-level programming language created by Guido van Rossum.",
        ])
        conv2 = _make_conversation([
            "How does photosynthesis work?",
            "Plants convert sunlight into chemical energy using chlorophyll.",
        ])
        assert not detector.is_duplicate(conv1)
        assert not detector.is_duplicate(conv2)
        assert detector.seen_count == 2

    def test_near_duplicate_detected(self):
        # Jaccard similarity ~0.45 for these texts with word trigrams,
        # so threshold 0.4 should catch the overlap.
        detector = DuplicateDetector(threshold=0.4, num_perm=256)
        conv1 = _make_conversation([
            "How do I fine-tune a large language model on my custom dataset?",
            "To fine-tune a large language model, you need to prepare your dataset, choose a base model, and train it.",
        ])
        conv2 = _make_conversation([
            "How do I fine-tune a large language model on my own dataset?",
            "To fine-tune a large language model, you should prepare your data, select a base model, and train it.",
        ])
        assert not detector.is_duplicate(conv1)
        assert detector.is_duplicate(conv2)

    def test_threshold_boundary(self):
        """Very high threshold should let similar conversations pass."""
        detector = DuplicateDetector(threshold=0.99)
        conv1 = _make_conversation([
            "What is machine learning?",
            "Machine learning is a subset of artificial intelligence.",
        ])
        conv2 = _make_conversation([
            "What is machine learning?",
            "Machine learning is a branch of artificial intelligence.",
        ])
        assert not detector.is_duplicate(conv1)
        assert not detector.is_duplicate(conv2)  # threshold too high to trigger

    def test_reset_clears_state(self):
        detector = DuplicateDetector(threshold=0.7)
        conv = _make_conversation(["Hello", "Hi there"])
        detector.is_duplicate(conv)
        assert detector.seen_count == 1
        detector.reset()
        assert detector.seen_count == 0
        assert not detector.is_duplicate(conv)  # no longer a duplicate

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            DuplicateDetector(threshold=0.0)
        with pytest.raises(ValueError):
            DuplicateDetector(threshold=1.5)

    def test_thread_safety(self):
        """Multiple threads can call is_duplicate without corruption."""
        import threading

        detector = DuplicateDetector(threshold=0.99)  # high threshold = no dedup
        conversations = [
            _make_conversation([f"Question {i}", f"Answer {i}"])
            for i in range(50)
        ]

        def worker(convs):
            for conv in convs:
                detector.is_duplicate(conv)

        threads = [
            threading.Thread(target=worker, args=(conversations[i::5],))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert detector.seen_count == 50


class TestQualityGateWithDedup:
    """Integration tests for QualityGate + DuplicateDetector."""

    @pytest.mark.asyncio
    async def test_dedup_rejects_duplicate(self):
        from afterimage.quality_gate import QualityGate

        detector = DuplicateDetector(threshold=0.7)
        gate = QualityGate(evaluator=None, dedup=detector)

        conv = _make_conversation(["What is AI?", "AI is artificial intelligence."])
        result1 = await gate.evaluate(conv)
        assert result1.accepted
        assert not result1.rejected_as_duplicate

        result2 = await gate.evaluate(conv)
        assert not result2.accepted
        assert result2.rejected_as_duplicate

    @pytest.mark.asyncio
    async def test_no_dedup_accepts_all(self):
        from afterimage.quality_gate import QualityGate

        gate = QualityGate(evaluator=None, dedup=None)
        conv = _make_conversation(["Hello", "Hi"])

        result1 = await gate.evaluate(conv)
        result2 = await gate.evaluate(conv)
        assert result1.accepted
        assert result2.accepted
