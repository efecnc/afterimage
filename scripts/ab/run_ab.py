"""A/B runner: execute baseline and variant configs, capture timing and outputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Result of a single generator run."""

    label: str
    config_path: str
    config_hash: str
    output_path: str
    elapsed_s: float
    rows: int
    total_tokens: Optional[int]
    error: Optional[str]


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


async def _run_one(
    label: str,
    config_path: Path,
    seed: int,
    n: int,
    out_dir: Path,
) -> RunOutcome:
    """Run one generator end-to-end, writing JSONL to ``out_dir/{label}.jsonl``."""
    from afterimage.config import load_config
    from afterimage.config_to_generator import build_conversation_run
    from afterimage.storage import JSONLStorage

    _seed_everything(seed)

    cfg = load_config(str(config_path))
    cfg.generation.num_dialogs = n
    out_path = out_dir / f"{label}.jsonl"
    if out_path.exists():
        out_path.unlink()

    logger.info("label=%s config=%s n=%d seed=%d", label, config_path, n, seed)

    run = build_conversation_run(cfg)
    run.generator.storage = JSONLStorage(conversations_path=str(out_path))

    from ._setup import populate_personas_if_enabled
    await populate_personas_if_enabled(run, cfg)

    # Rebuild stopping criteria for the requested N (override YAML num_dialogs).
    from afterimage.callbacks import FixedNumberStoppingCallback

    stopping_criteria = [
        c
        for c in run.stopping_criteria
        if not isinstance(c, FixedNumberStoppingCallback)
    ]
    stopping_criteria.append(FixedNumberStoppingCallback(n=n))

    start = time.time()
    error: Optional[str] = None
    try:
        await run.generator.generate(
            num_dialogs=None,
            max_turns=cfg.generation.max_turns,
            max_concurrency=cfg.generation.max_concurrency,
            stopping_criteria=stopping_criteria,
            num_requested=n,
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error("run %s failed: %s", label, error)
        traceback.print_exc()

    elapsed = time.time() - start
    rows = _count_jsonl_rows(out_path)

    total_tokens: Optional[int] = None
    monitor = getattr(run.generator, "monitor", None)
    if monitor is not None and hasattr(monitor, "get_total_token_usage"):
        try:
            total_tokens = int(monitor.get_total_token_usage().total_tokens)
        except Exception:
            total_tokens = None

    return RunOutcome(
        label=label,
        config_path=str(config_path),
        config_hash=_config_hash(config_path),
        output_path=str(out_path),
        elapsed_s=elapsed,
        rows=rows,
        total_tokens=total_tokens,
        error=error,
    )


async def _main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = Path(args.baseline).resolve()
    variant_cfg = Path(args.variant).resolve()

    for p in (baseline_cfg, variant_cfg):
        if not p.is_file():
            logger.error("config not found: %s", p)
            return 2

    # Preserve a copy of each config alongside the outputs for later reference.
    shutil.copy2(baseline_cfg, out_dir / "baseline_config.yaml")
    shutil.copy2(variant_cfg, out_dir / "variant_config.yaml")

    baseline = await _run_one("baseline", baseline_cfg, args.seed, args.n, out_dir)
    variant = await _run_one("variant", variant_cfg, args.seed, args.n, out_dir)

    meta = {
        "seed": args.seed,
        "n": args.n,
        "baseline_elapsed_s": baseline.elapsed_s,
        "variant_elapsed_s": variant.elapsed_s,
        "baseline_rows": baseline.rows,
        "variant_rows": variant.rows,
        "baseline_total_tokens": baseline.total_tokens,
        "variant_total_tokens": variant.total_tokens,
        "baseline_config_hash": baseline.config_hash,
        "variant_config_hash": variant.config_hash,
        "baseline_error": baseline.error,
        "variant_error": variant.error,
        "baseline_output": baseline.output_path,
        "variant_output": variant.output_path,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("wrote %s", out_dir / "run_meta.json")

    # Exit 0 if at least one run produced rows, else 1.
    return 0 if (baseline.rows or variant.rows) else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline and variant generators under identical seeds.",
    )
    p.add_argument("--baseline", required=True, help="Path to baseline YAML config.")
    p.add_argument("--variant", required=True, help="Path to variant YAML config.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
