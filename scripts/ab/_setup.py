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
