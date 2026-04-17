"""In-process embedding dedup gate.

Wraps any :class:`~afterimage.storage.BaseStorage` so that near-duplicate
conversations (cosine similarity on the final assistant turn above
``threshold`` against a rolling window of prior accepted rows) are dropped
before they hit storage. Reuses an existing
:class:`~afterimage.providers.embedding_providers.EmbeddingProvider` so no
extra API budget is allocated beyond one embedding per write.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Iterable

import numpy as np

from .providers.embedding_providers import EmbeddingProvider
from .storage import BaseStorage
from .types import Role

logger = logging.getLogger(__name__)


def _final_assistant_text(conversation: Any) -> str | None:
    entries = getattr(conversation, "conversations", None) or []
    texts = [
        getattr(e, "content", None) for e in entries if getattr(e, "role", None) == Role.ASSISTANT
    ]
    texts = [t for t in texts if isinstance(t, str) and t]
    return texts[-1] if texts else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class DedupStorage(BaseStorage):
    """Storage wrapper that drops near-duplicate conversations.

    Args:
        inner: Underlying storage to receive accepted rows.
        embedding_provider: Shared embedding model (reuse the judge's).
        threshold: Cosine similarity above which a row is considered a dup.
        window_size: How many recent embeddings to keep in memory.
    """

    def __init__(
        self,
        inner: BaseStorage,
        embedding_provider: EmbeddingProvider,
        threshold: float = 0.92,
        window_size: int = 500,
    ) -> None:
        self._inner = inner
        self._embed = embedding_provider
        self._threshold = threshold
        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._lock = asyncio.Lock()
        self.dropped = 0
        self.accepted = 0

    async def _embed_text(self, text: str) -> np.ndarray:
        vecs = await self._embed.embed([text])
        arr = np.asarray(vecs[0], dtype=np.float32).reshape(-1)
        return arr

    async def _is_duplicate(self, vec: np.ndarray) -> bool:
        if not self._window:
            return False
        sims = [_cosine(vec, w) for w in self._window]
        return max(sims) >= self._threshold

    async def _filter(self, conversations: Iterable[Any]) -> list[Any]:
        kept: list[Any] = []
        async with self._lock:
            for conv in conversations:
                text = _final_assistant_text(conv)
                if not text:
                    kept.append(conv)
                    continue
                vec = await self._embed_text(text)
                if await self._is_duplicate(vec):
                    self.dropped += 1
                    logger.debug("dedup dropped near-duplicate row")
                    continue
                self._window.append(vec)
                self.accepted += 1
                kept.append(conv)
        return kept

    async def asave_conversations(self, conversations: Iterable[Any]) -> None:
        kept = await self._filter(conversations)
        if kept:
            if hasattr(self._inner, "asave_conversations"):
                await self._inner.asave_conversations(kept)
            else:
                await asyncio.to_thread(self._inner.save_conversations, kept)

    def save_conversations(self, conversations: Iterable[Any]) -> None:
        loop = asyncio.new_event_loop()
        try:
            kept = loop.run_until_complete(self._filter(conversations))
        finally:
            loop.close()
        if kept:
            self._inner.save_conversations(kept)

    def load_conversations(self):
        return self._inner.load_conversations()

    def load_documents(self):
        return self._inner.load_documents() if hasattr(self._inner, "load_documents") else []

    async def asave_documents(self, documents):
        if hasattr(self._inner, "asave_documents"):
            await self._inner.asave_documents(documents)

    def save_documents(self, documents):
        if hasattr(self._inner, "save_documents"):
            self._inner.save_documents(documents)
