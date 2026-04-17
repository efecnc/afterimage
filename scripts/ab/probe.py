"""Instrumented probe run: counts retrieval misses, silent failures, grades, retries."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProbeCounters:
    """Runtime counters populated by the instrumentation context manager."""

    retriever_calls: int = 0
    retriever_failures: int = 0
    structured_silent_failures: int = 0
    grade_counts: Counter = field(default_factory=Counter)
    retries_per_dialog: list[int] = field(default_factory=list)
    personas: list[str] = field(default_factory=list)


@contextlib.contextmanager
def instrument(counters: ProbeCounters) -> Iterator[ProbeCounters]:
    """Monkey-patch generator hotspots to record probe counters; revert on exit."""
    from afterimage import structured_generator as sg_module
    from afterimage.conversation_generator import ConversationGenerator
    from afterimage.quality_gate import QualityGate
    from afterimage.retrievers import QdrantRetriever
    from afterimage.structured_generator import StructuredGenerator

    from .metrics import RETRIEVAL_FAIL_SENTINEL

    fail_sentinel = RETRIEVAL_FAIL_SENTINEL

    # 1) Retriever miss counter
    original_search = QdrantRetriever._search_with_vector

    def patched_search(self, query_vector):
        counters.retriever_calls += 1
        result = original_search(self, query_vector)
        if isinstance(result, str) and fail_sentinel in result:
            counters.retriever_failures += 1
        return result

    QdrantRetriever._search_with_vector = patched_search

    # 2) StructuredGenerator silent except/continue counter, observed via
    # GenerationMonitor.track_generation (the library logs success=False with
    # metadata.operation == "structured_generation" inside the silent except).
    _ = StructuredGenerator  # referenced for import validation
    monitor_class = sg_module.GenerationMonitor
    original_track = monitor_class.track_generation

    def patched_track(self, duration, success, **kwargs):
        if (
            not success
            and isinstance(kwargs.get("metadata"), dict)
            and kwargs["metadata"].get("operation") == "structured_generation"
        ):
            counters.structured_silent_failures += 1
        return original_track(self, duration, success, **kwargs)

    monitor_class.track_generation = patched_track

    # 3) QualityGate grade recorder
    original_evaluate = QualityGate.evaluate

    async def patched_evaluate(self, conversation_row):
        result = await original_evaluate(self, conversation_row)
        ev = getattr(result.conversation_row, "evaluation", None)
        grade = getattr(ev, "overall_grade", None)
        if grade is not None:
            counters.grade_counts[str(grade.value if hasattr(grade, "value") else grade).lower()] += 1
        return result

    QualityGate.evaluate = patched_evaluate

    # 4) ConversationGenerator retry counter
    original_cg_single = ConversationGenerator.generate_single

    async def patched_cg_single(self, *args, **kwargs):
        async for row in _count_retries(self, counters, original_cg_single, *args, **kwargs):
            yield row

    ConversationGenerator.generate_single = patched_cg_single

    try:
        yield counters
    finally:
        QdrantRetriever._search_with_vector = original_search
        monitor_class.track_generation = original_track
        QualityGate.evaluate = original_evaluate
        ConversationGenerator.generate_single = original_cg_single


async def _count_retries(generator, counters, original, *args, **kwargs):
    """Wrap generate_single to observe per-dialog retry count via grade_counts delta."""
    before = sum(counters.grade_counts.values())
    async for row in original(generator, *args, **kwargs):
        after = sum(counters.grade_counts.values())
        # Each retried dialog fires ``evaluate`` once per attempt (final accepted attempt included).
        attempts = max(0, after - before)
        retries = max(0, attempts - 1) if attempts else 0
        counters.retries_per_dialog.append(retries)
        before = after
        persona = getattr(row, "persona", None)
        if isinstance(persona, str) and persona:
            counters.personas.append(persona)
        yield row


def _format_grade_table(counters: ProbeCounters) -> str:
    total = sum(counters.grade_counts.values())
    if total == 0:
        return "_no grade data collected_"
    bands = ["perfect", "good", "needs_improvement", "bad", "not_acceptable"]
    rows = ["| Grade | Count | Fraction |", "|---|---|---|"]
    for b in bands:
        c = counters.grade_counts.get(b, 0)
        rows.append(f"| {b} | {c} | {c / total:.3f} |")
    passing = counters.grade_counts.get("perfect", 0) + counters.grade_counts.get(
        "good", 0
    )
    rows.append(f"| **>= GOOD** | {passing} | {passing / total:.3f} |")
    return "\n".join(rows)


def _format_retry_table(counters: ProbeCounters) -> str:
    retries = counters.retries_per_dialog
    if not retries:
        return "_no dialogs observed_"
    bins = Counter()
    for r in retries:
        bins["3+" if r >= 3 else str(r)] += 1
    rows = ["| Retries | Dialogs |", "|---|---|"]
    for key in ["0", "1", "2", "3+"]:
        rows.append(f"| {key} | {bins.get(key, 0)} |")
    rows.append(f"| max | {max(retries)} |")
    rows.append(f"| mean | {np.mean(retries):.2f} |")
    return "\n".join(rows)


def _format_retrieval_table(counters: ProbeCounters) -> str:
    if counters.retriever_calls == 0:
        return "_no retriever calls observed_"
    frac = counters.retriever_failures / counters.retriever_calls
    return (
        "| Total calls | Failures | Failure rate |\n"
        "|---|---|---|\n"
        f"| {counters.retriever_calls} | {counters.retriever_failures} | {frac:.3f} |"
    )


def _format_structured_table(counters: ProbeCounters) -> str:
    return (
        "| Silent structured-gen failures |\n"
        "|---|\n"
        f"| {counters.structured_silent_failures} |"
    )


def _format_persona_table(counters: ProbeCounters, n: int) -> str:
    unique = len(set(counters.personas))
    total = len(counters.personas)
    return (
        "| N requested | Personas observed | Unique |\n"
        "|---|---|---|\n"
        f"| {n} | {total} | {unique} |"
    )


def write_report(path: Path, counters: ProbeCounters, n: int, elapsed: float) -> None:
    body = [
        "# AfterImage Probe Report",
        "",
        f"- dialogs requested: {n}",
        f"- elapsed: {elapsed:.1f}s",
        "",
        "## Judge grade distribution",
        _format_grade_table(counters),
        "",
        "## Retry distribution",
        _format_retry_table(counters),
        "",
        "## Retriever fail rate",
        _format_retrieval_table(counters),
        "",
        "## Structured-gen silent-failure rate",
        _format_structured_table(counters),
        "",
        "## Persona cardinality",
        _format_persona_table(counters, n),
        "",
    ]
    path.write_text("\n".join(body))


async def _run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from afterimage.config import load_config
    from afterimage.config_to_generator import build_conversation_run
    from afterimage.storage import JSONLStorage
    from afterimage.callbacks import FixedNumberStoppingCallback

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    cfg.generation.num_dialogs = args.n

    run = build_conversation_run(cfg)
    out_jsonl = out_dir / "probe_output.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()
    run.generator.storage = JSONLStorage(conversations_path=str(out_jsonl))

    from ._setup import populate_personas_if_enabled, apply_retry_policy
    await populate_personas_if_enabled(run, cfg)
    apply_retry_policy(
        run,
        max_retries=args.max_retries,
        inject_judge_feedback=args.inject_judge_feedback,
    )

    stopping_criteria = [
        c
        for c in run.stopping_criteria
        if not isinstance(c, FixedNumberStoppingCallback)
    ]
    stopping_criteria.append(FixedNumberStoppingCallback(n=args.n))

    counters = ProbeCounters()
    start = time.time()
    try:
        with instrument(counters):
            await run.generator.generate(
                num_dialogs=None,
                max_turns=cfg.generation.max_turns,
                max_concurrency=cfg.generation.max_concurrency,
                stopping_criteria=stopping_criteria,
                num_requested=args.n,
            )
    except Exception as e:
        logger.error("probe run failed: %s", e)
        traceback.print_exc()
    elapsed = time.time() - start

    report_path = out_dir / "probe_report.md"
    write_report(report_path, counters, args.n, elapsed)
    logger.info("wrote %s", report_path)

    # Persist raw counters too.
    raw = {
        "retriever_calls": counters.retriever_calls,
        "retriever_failures": counters.retriever_failures,
        "structured_silent_failures": counters.structured_silent_failures,
        "grade_counts": dict(counters.grade_counts),
        "retries_per_dialog": counters.retries_per_dialog,
        "persona_count": len(counters.personas),
        "persona_unique": len(set(counters.personas)),
        "elapsed_s": elapsed,
    }
    (out_dir / "probe_counters.json").write_text(json.dumps(raw, indent=2) + "\n")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an instrumented generator probe and emit probe_report.md",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--max-retries",
        type=int,
        default=None,
        dest="max_retries",
        help="Cap retries per dialog (default: unlimited).",
    )
    p.add_argument(
        "--inject-judge-feedback",
        action="store_true",
        dest="inject_judge_feedback",
        help="Inject lowest-scoring judge feedback into respondent prompt on retry.",
    )
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
