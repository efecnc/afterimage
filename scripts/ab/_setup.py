"""Shared helpers for A/B tooling. Currently: persona pre-generation.

``config_to_generator.build_conversation_run`` wires ``PersonaInstructionGeneratorCallback``
when ``personas.enabled: true`` but never populates the persona pool. The callback
then falls back to a default ``"A curious user"`` for every dialog. This helper
runs the missing ``PersonaGenerator`` pass so the baseline actually exercises the
persona layer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def populate_personas_if_enabled(run: Any, cfg: Any) -> bool:
    """Populate the callback's document provider with personas when enabled.

    Returns True when personas were generated, False when skipped (not enabled,
    wrong callback type, or provider missing).
    """
    if not getattr(getattr(cfg, "personas", None), "enabled", False):
        return False

    callback = getattr(run.generator, "instruction_generator_callback", None)
    provider = getattr(callback, "provider", None)
    if provider is None:
        logger.warning("persona pre-gen skipped: callback has no provider")
        return False

    from afterimage.persona_generator import PersonaGenerator
    from afterimage.config import resolve_api_key

    api_key = resolve_api_key(cfg)
    if api_key is None and cfg.model.provider == "local":
        api_key = "not-needed"

    persona_gen = PersonaGenerator(
        api_key=api_key,
        model_name=cfg.model.model_name,
        model_provider_name=cfg.model.provider,
    )
    logger.info("pre-generating personas for %d docs", len(provider.get_all()))
    await persona_gen.generate_from_documents(provider)

    total = sum(
        sum(len(e.descriptions) for e in d.personas) for d in provider.get_all()
    )
    logger.info("pre-generated %d persona descriptions across provider", total)
    return True


def install_dedup_gate(run: Any, threshold: float, window_size: int = 500) -> Any | None:
    """Wrap the generator's storage with :class:`DedupStorage`.

    Returns the installed :class:`DedupStorage` so callers can read
    ``.accepted`` / ``.dropped`` after the run, or ``None`` when no embedding
    provider is available (dedup is silently skipped).
    """
    from afterimage.dedup import DedupStorage

    gen = run.generator
    evaluator = getattr(gen, "evaluator", None) or getattr(gen, "_evaluator", None)
    embedder = None
    if evaluator is not None:
        embedder = (
            getattr(evaluator, "embedding_provider", None)
            or getattr(evaluator, "_embedding_provider", None)
            or getattr(evaluator, "_embedding", None)
        )

    if embedder is None:
        logger.warning(
            "dedup gate skipped: no embedding provider on evaluator "
            "(enable auto_improve or supply embedding_provider to the generator)"
        )
        return None

    wrapped = DedupStorage(
        inner=gen.storage,
        embedding_provider=embedder,
        threshold=threshold,
        window_size=window_size,
    )
    gen.storage = wrapped
    logger.info(
        "dedup gate installed: threshold=%.3f window_size=%d",
        threshold,
        window_size,
    )
    return wrapped
