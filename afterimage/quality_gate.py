"""Quality gate for conversation generation.

Wraps ConversationJudge and implements accept/reject/retry logic for the
auto_improve workflow. Extracts evaluation retry logic from
ConversationGenerator.generate_single().

Also integrates near-duplicate detection via MinHash to reject conversations
that are too similar to previously accepted ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .dedup import DuplicateDetector
from .types import (
    ConversationWithContext,
    EvaluatedConversationWithContext,
    GradeSchema,
)

if TYPE_CHECKING:
    from .evaluator import ConversationJudge

logger = logging.getLogger(__name__)

# Grades that trigger a retry
_RETRY_GRADES = frozenset(
    {
        GradeSchema.NOT_ACCEPTABLE,
        GradeSchema.BAD,
        GradeSchema.NEEDS_IMPROVEMENT,
    }
)


@dataclass
class QualityResult:
    """Result of a quality gate evaluation.

    Attributes:
        conversation_row: The evaluated conversation (with evaluation if judge was used).
        accepted: Whether the conversation passed quality checks.
        rejected_as_duplicate: Whether the conversation was rejected for being a near-duplicate.
    """

    conversation_row: ConversationWithContext | EvaluatedConversationWithContext
    accepted: bool
    rejected_as_duplicate: bool = False


class QualityGate:
    """Evaluates conversations and decides whether to accept or retry.

    When no evaluator is configured (auto_improve=False), all conversations
    are accepted immediately. When an evaluator is present, conversations are
    judged and only accepted if they meet the grade threshold.

    When dedup is enabled, conversations that are near-duplicates of previously
    accepted ones are rejected before quality evaluation (saving LLM judge calls).

    Attributes:
        evaluator: Optional ConversationJudge instance for quality evaluation.
        dedup: Optional DuplicateDetector for near-duplicate rejection.
    """

    def __init__(
        self,
        evaluator: Optional[ConversationJudge] = None,
        dedup: Optional[DuplicateDetector] = None,
    ):
        self._evaluator = evaluator
        self._dedup = dedup

    @property
    def evaluator(self) -> Optional[ConversationJudge]:
        return self._evaluator

    @property
    def dedup(self) -> Optional[DuplicateDetector]:
        return self._dedup

    @property
    def is_enabled(self) -> bool:
        """Whether quality gating is active (evaluation or dedup)."""
        return self._evaluator is not None or self._dedup is not None

    async def evaluate(
        self, conversation_row: ConversationWithContext
    ) -> QualityResult:
        """Evaluate a conversation row.

        Checks near-duplicate status first (cheap), then quality evaluation
        (expensive LLM calls) only if the conversation is unique.

        Args:
            conversation_row: The conversation to evaluate.

        Returns:
            QualityResult with the evaluated row and acceptance status.
        """
        # --- Dedup check (fast, before expensive LLM evaluation) ---
        if self._dedup is not None:
            if self._dedup.is_duplicate(conversation_row):
                logger.info(
                    "Conversation rejected as near-duplicate "
                    "(seen %d unique so far)",
                    self._dedup.seen_count,
                )
                return QualityResult(
                    conversation_row=conversation_row,
                    accepted=False,
                    rejected_as_duplicate=True,
                )

        # --- Quality evaluation ---
        if self._evaluator is None:
            return QualityResult(conversation_row=conversation_row, accepted=True)

        evaluated = await self._evaluator.aevaluate_row(conversation_row)
        accepted = evaluated.evaluation.overall_grade not in _RETRY_GRADES
        return QualityResult(conversation_row=evaluated, accepted=accepted)

    @staticmethod
    def should_retry(result: QualityResult) -> bool:
        """Whether the conversation should be regenerated.

        Args:
            result: A previous QualityResult.

        Returns:
            True if the conversation should be retried.
        """
        return not result.accepted
