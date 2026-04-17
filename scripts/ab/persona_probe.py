"""Persona-pool-only probe: generates personas at given depth, measures diversity.

Runs :class:`~afterimage.persona_generator.PersonaGenerator` on the demo corpus
(ab_baseline.yaml) at a configurable tree depth and reports:

- total persona descriptions
- exact-string unique count
- mean and max pairwise embedding cosine similarity across the pool

Used to isolate fix 4's effect on persona cardinality without running the full
conversation pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from itertools import combinations
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


async def _run(args: argparse.Namespace) -> int:
    from afterimage.config import load_config, resolve_api_key
    from afterimage.config_to_generator import _build_document_provider
    from afterimage.persona_generator import PersonaGenerator

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    if cfg.documents is None:
        logger.error("config has no documents block; cannot probe personas")
        return 2

    provider = _build_document_provider(cfg)
    api_key = resolve_api_key(cfg)

    persona_gen = PersonaGenerator(
        api_key=api_key,
        model_name=cfg.model.model_name,
        model_provider_name=cfg.model.provider,
    )
    logger.info(
        "generating personas: docs=%d depth=%d",
        len(provider.get_all()),
        args.depth,
    )
    await persona_gen.generate_from_documents(provider, n_iterations=args.depth)

    all_descriptions: list[str] = []
    for doc in provider.get_all():
        for entry in doc.personas:
            for d in entry.descriptions:
                all_descriptions.append(d)

    unique = len(set(all_descriptions))
    total = len(all_descriptions)

    # Embedding pairwise cosine
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeds = model.encode(all_descriptions, normalize_embeddings=True)
    sims = []
    for i, j in combinations(range(len(embeds)), 2):
        sims.append(float(np.dot(embeds[i], embeds[j])))
    mean_sim = float(np.mean(sims)) if sims else 0.0
    max_sim = float(np.max(sims)) if sims else 0.0
    # Count highly-similar pairs (> 0.85) as proxy for mode collapse
    near_dup_pairs = sum(1 for s in sims if s > 0.85)

    report = {
        "depth": args.depth,
        "total_personas": total,
        "exact_unique": unique,
        "cardinality_ratio": unique / max(total, 1),
        "pairwise_cosine_mean": mean_sim,
        "pairwise_cosine_max": max_sim,
        "near_dup_pair_count_gt_0_85": near_dup_pairs,
        "near_dup_pair_rate": near_dup_pairs / max(len(sims), 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("report=%s", json.dumps(report, indent=2))

    # Also persist the raw persona list so we can inspect.
    (out_path.parent / "personas.jsonl").write_text(
        "\n".join(json.dumps({"text": d}) for d in all_descriptions) + "\n"
    )
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe persona-pool diversity only.")
    p.add_argument("--config", required=True)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
