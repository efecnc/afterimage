"""Compare baseline vs variant JSONL on every metric; emit markdown report."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

from .metrics import (
    ALL_METRICS,
    MetricResult,
    load_rows,
    metric_direction,
    paired_delta_ci,
)

logger = logging.getLogger(__name__)

_PAIRED_KINDS = {
    "retriever_fail_rate": "retriever_fail",
    "length_entropy": "length",
}


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.4f}"


def _delta_arrow(base: MetricResult, var: MetricResult) -> tuple[float, str]:
    if math.isnan(base.value) or math.isnan(var.value):
        return float("nan"), ""
    delta = var.value - base.value
    direction = metric_direction(base.name)
    if direction == "up":
        arrow = "good" if delta > 0 else "worse" if delta < 0 else ""
    elif direction == "down":
        arrow = "good" if delta < 0 else "worse" if delta > 0 else ""
    else:
        arrow = ""
    return delta, arrow


def _paired_ci_for(
    name: str,
    base_rows: Sequence[dict],
    var_rows: Sequence[dict],
) -> tuple[float, float] | None:
    kind = _PAIRED_KINDS.get(name)
    if kind is None:
        return None
    result = paired_delta_ci(base_rows, var_rows, kind)
    if result is None:
        return None
    _, lo, hi = result
    return lo, hi


def _build_report(
    baseline_path: Path,
    variant_path: Path,
    base_rows: Sequence[dict],
    var_rows: Sequence[dict],
    base_results: list[MetricResult],
    var_results: list[MetricResult],
) -> str:
    lines = [
        "# A/B Comparison",
        "",
        f"- baseline: `{baseline_path}` (n={len(base_rows)})",
        f"- variant: `{variant_path}` (n={len(var_rows)})",
        "",
        "| Metric | Baseline [CI] | Variant [CI] | Δ | Δ CI | Direction |",
        "|---|---|---|---|---|---|",
    ]
    skipped: list[tuple[str, str]] = []
    for base, var in zip(base_results, var_results):
        base_cell = f"{_fmt(base.value)} [{_fmt(base.ci_low)}, {_fmt(base.ci_high)}]"
        var_cell = f"{_fmt(var.value)} [{_fmt(var.ci_low)}, {_fmt(var.ci_high)}]"
        delta, arrow = _delta_arrow(base, var)
        paired = _paired_ci_for(base.name, base_rows, var_rows)
        delta_cell = _fmt(delta)
        if paired is not None:
            delta_ci_cell = f"[{_fmt(paired[0])}, {_fmt(paired[1])}]"
        else:
            delta_ci_cell = "-"
        lines.append(
            f"| `{base.name}` | {base_cell} | {var_cell} | {delta_cell} | {delta_ci_cell} | {arrow} |"
        )
        if base.notes or var.notes:
            if base.notes:
                skipped.append((f"{base.name} (baseline)", base.notes))
            if var.notes:
                skipped.append((f"{var.name} (variant)", var.notes))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Direction = good` flags a metric where the variant moved the right way.")
    lines.append("- Paired Δ CI is reported only for row-aligned per-row metrics.")
    if skipped:
        lines.append("- Skipped or annotated metrics:")
        for name, note in skipped:
            lines.append(f"  - `{name}`: {note}")
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline and variant JSONL outputs.")
    p.add_argument("baseline", help="Baseline JSONL path")
    p.add_argument("variant", help="Variant JSONL path")
    p.add_argument("--out", default="report.md")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_path = Path(args.baseline).resolve()
    var_path = Path(args.variant).resolve()
    out_path = Path(args.out).resolve()

    base_rows = load_rows(base_path)
    var_rows = load_rows(var_path)
    logger.info("loaded %d baseline rows / %d variant rows", len(base_rows), len(var_rows))

    base_results: list[MetricResult] = []
    var_results: list[MetricResult] = []
    for _, fn in ALL_METRICS:
        base_results.append(fn(base_rows))
        var_results.append(fn(var_rows))

    report = _build_report(
        base_path, var_path, base_rows, var_rows, base_results, var_results
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
