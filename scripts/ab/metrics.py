"""A/B quality metrics operating on AfterImage JSONL rows."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# The literal retrieval-miss sentinel emitted by afterimage.retrievers.
RETRIEVAL_FAIL_SENTINEL = "No relevant context found."

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_BOOTSTRAP_RESAMPLES = 1000
_DEFAULT_RNG_SEED = 12345

# Grades at or above GOOD count as a pass.
_PASSING_GRADES = frozenset({"perfect", "good"})
_ALL_GRADES = ("perfect", "good", "needs_improvement", "bad", "not_acceptable")


@dataclass
class MetricResult:
    """Scalar metric with 95% CI."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file of conversation rows."""
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line in %s", p)
    return rows


def _assistant_turns(row: dict[str, Any]) -> list[str]:
    turns = row.get("conversations") or []
    return [
        t.get("content", "")
        for t in turns
        if isinstance(t, dict) and t.get("role") == "assistant" and t.get("content")
    ]


def _final_assistant(row: dict[str, Any]) -> str:
    t = _assistant_turns(row)
    return t[-1] if t else ""


def _all_assistant_text(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Concatenate all assistant turns per row (one string per row)."""
    return [" ".join(_assistant_turns(r)) for r in rows]


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float],
    resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _DEFAULT_RNG_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for ``stat`` over ``values``."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        v = float(stat(arr))
        return (v, v)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(resamples, n))
    samples = arr[indices]
    stats = np.array([stat(row) for row in samples], dtype=float)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def _proportion_ci(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    bits = np.zeros(n, dtype=float)
    bits[:successes] = 1.0
    return _bootstrap_ci(bits, stat=np.mean)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def self_bleu(rows: Sequence[dict[str, Any]], n: int = 4) -> MetricResult:
    """Mean self-BLEU-n across rows. Lower is more diverse."""
    texts = [t for t in _all_assistant_text(rows) if t.strip()]
    if len(texts) < 2:
        return MetricResult(
            name=f"self_bleu_{n}",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(texts),
            notes="not enough rows to compare",
        )

    token_lists = [_tokenize(t) for t in texts]
    ref_counts = [_ngram_counts(toks, n) for toks in token_lists]

    per_row: list[float] = []
    for i, cand in enumerate(token_lists):
        cand_counts = _ngram_counts(cand, n)
        if not cand_counts:
            per_row.append(0.0)
            continue
        merged: Counter = Counter()
        for j, other in enumerate(ref_counts):
            if j == i:
                continue
            for ng, c in other.items():
                if c > merged[ng]:
                    merged[ng] = c
        overlap = sum(min(c, merged.get(ng, 0)) for ng, c in cand_counts.items())
        total = sum(cand_counts.values())
        per_row.append(overlap / total if total else 0.0)

    mean = float(np.mean(per_row))
    lo, hi = _bootstrap_ci(per_row, np.mean)
    return MetricResult(
        name=f"self_bleu_{n}",
        value=mean,
        ci_low=lo,
        ci_high=hi,
        n=len(texts),
    )


def distinct_n(rows: Sequence[dict[str, Any]], n: int = 2) -> MetricResult:
    """Distinct-n ratio over assistant-turn tokens, mean over rows."""
    per_row: list[float] = []
    for text in _all_assistant_text(rows):
        toks = _tokenize(text)
        if len(toks) < n:
            continue
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        if not grams:
            continue
        per_row.append(len(set(grams)) / len(grams))
    if not per_row:
        return MetricResult(
            name=f"distinct_{n}",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows had enough tokens",
        )
    lo, hi = _bootstrap_ci(per_row, np.mean)
    return MetricResult(
        name=f"distinct_{n}",
        value=float(np.mean(per_row)),
        ci_low=lo,
        ci_high=hi,
        n=len(per_row),
    )


def length_entropy(rows: Sequence[dict[str, Any]], bins: int = 10) -> MetricResult:
    """Normalized Shannon entropy over binned final-turn char lengths."""
    lengths = [len(_final_assistant(r)) for r in rows if _final_assistant(r)]
    if len(lengths) < 2:
        return MetricResult(
            name="length_entropy",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(lengths),
            notes="not enough rows",
        )
    lo_edge, hi_edge = min(lengths), max(lengths)
    if hi_edge == lo_edge:
        return MetricResult(
            name="length_entropy",
            value=0.0,
            ci_low=0.0,
            ci_high=0.0,
            n=len(lengths),
            notes="all lengths identical",
        )

    def _entropy_of(sample: np.ndarray) -> float:
        hist, _ = np.histogram(sample, bins=bins, range=(lo_edge, hi_edge))
        total = hist.sum()
        if total == 0:
            return 0.0
        p = hist[hist > 0] / total
        h = -float(np.sum(p * np.log(p)))
        return h / math.log(bins) if bins > 1 else 0.0

    value = _entropy_of(np.asarray(lengths, dtype=float))
    lo, hi = _bootstrap_ci(lengths, _entropy_of)
    return MetricResult(
        name="length_entropy",
        value=value,
        ci_low=lo,
        ci_high=hi,
        n=len(lengths),
    )


# ---------------------------------------------------------------------------
# Embedding metrics
# ---------------------------------------------------------------------------


def _try_embed(texts: Sequence[str], model_name: str) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed; skipping embedding metric")
        return None
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.warning("failed to load %s: %s", model_name, e)
        return None
    vecs = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs


def dedup_rate(
    rows: Sequence[dict[str, Any]],
    threshold: float = 0.92,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> MetricResult:
    """Fraction of rows whose nearest neighbor (cosine, off-diagonal) exceeds ``threshold``."""
    texts = [t for t in (_final_assistant(r) for r in rows) if t]
    if len(texts) < 2:
        return MetricResult(
            name="dedup_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(texts),
            notes="not enough rows",
        )
    vecs = _try_embed(texts, model_name)
    if vecs is None:
        return MetricResult(
            name="dedup_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(texts),
            notes="skipped: sentence-transformers unavailable",
        )

    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    nn = sim.max(axis=1)
    hits = (nn > threshold).astype(float)
    value = float(hits.mean())
    lo, hi = _bootstrap_ci(hits, np.mean)
    return MetricResult(
        name="dedup_rate",
        value=value,
        ci_low=lo,
        ci_high=hi,
        n=len(texts),
        extra={"threshold": threshold},
    )


def nli_entailment_rate(
    rows: Sequence[dict[str, Any]],
    model: str = "cross-encoder/nli-deberta-v3-small",
) -> MetricResult:
    """Fraction of rows where (context, final_assistant_turn) entails with P>0.5."""
    pairs: list[tuple[str, str]] = []
    for r in rows:
        ctx = r.get("instruction_context") or r.get("response_context")
        ans = _final_assistant(r)
        if ctx and ans:
            pairs.append((ctx, ans))
    if not pairs:
        return MetricResult(
            name="nli_entailment_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows with context",
        )
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return MetricResult(
            name="nli_entailment_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(pairs),
            notes="skipped: sentence-transformers unavailable",
        )
    try:
        ce = CrossEncoder(model)
    except Exception as e:
        return MetricResult(
            name="nli_entailment_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(pairs),
            notes=f"skipped: model unavailable ({e})",
        )

    try:
        raw = ce.predict(pairs, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        return MetricResult(
            name="nli_entailment_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(pairs),
            notes=f"skipped: inference failed ({e})",
        )

    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 2 and raw.shape[1] >= 3:
        # Typical NLI label order: contradiction, entailment, neutral.
        labels = [lbl.lower() for lbl in getattr(ce, "default_label", []) or []]
        if "entailment" in labels:
            ent_idx = labels.index("entailment")
        else:
            ent_idx = 1
        ent_probs = raw[:, ent_idx]
    else:
        ent_probs = raw.reshape(-1)
    hits = (ent_probs > 0.5).astype(float)
    lo, hi = _bootstrap_ci(hits, np.mean)
    return MetricResult(
        name="nli_entailment_rate",
        value=float(hits.mean()),
        ci_low=lo,
        ci_high=hi,
        n=len(pairs),
    )


# ---------------------------------------------------------------------------
# Judge / metadata metrics
# ---------------------------------------------------------------------------


def judge_grade_dist(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Fraction of rows passing >= GOOD; full distribution in ``notes``/``extra``."""
    grades: list[str] = []
    for r in rows:
        ev = r.get("evaluation")
        if isinstance(ev, dict):
            g = ev.get("overall_grade")
            if isinstance(g, str):
                grades.append(g.lower())
    if not grades:
        return MetricResult(
            name="judge_grade_pass_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows had evaluation.overall_grade",
        )
    dist = {g: 0 for g in _ALL_GRADES}
    for g in grades:
        dist[g] = dist.get(g, 0) + 1
    passes = sum(1 for g in grades if g in _PASSING_GRADES)
    value = passes / len(grades)
    lo, hi = _proportion_ci(passes, len(grades))
    return MetricResult(
        name="judge_grade_pass_rate",
        value=value,
        ci_low=lo,
        ci_high=hi,
        n=len(grades),
        notes=json.dumps(dist),
        extra={"distribution": dist},
    )


def persona_cardinality(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Distinct persona strings / N."""
    personas = []
    for r in rows:
        meta = r.get("metadata") or {}
        p = meta.get("persona_name") or r.get("persona")
        if isinstance(p, str) and p:
            personas.append(p)
    total = len(rows)
    if total == 0:
        return MetricResult(
            name="persona_cardinality",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows",
        )
    unique = len(set(personas))
    value = unique / total
    return MetricResult(
        name="persona_cardinality",
        value=value,
        ci_low=value,
        ci_high=value,
        n=total,
        notes=f"unique={unique}",
        extra={"unique": unique, "with_persona": len(personas)},
    )


def distinct_contexts(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Distinct ``metadata.context_ids`` tuples / N."""
    total = len(rows)
    if total == 0:
        return MetricResult(
            name="distinct_contexts",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows",
        )
    keys: list[tuple[str, ...]] = []
    for r in rows:
        meta = r.get("metadata") or {}
        ids = meta.get("context_ids")
        if isinstance(ids, list) and ids:
            keys.append(tuple(str(x) for x in ids))
        else:
            cid = meta.get("context_id")
            keys.append((str(cid),) if cid else ())
    unique = len({k for k in keys if k})
    value = unique / total
    return MetricResult(
        name="distinct_contexts",
        value=value,
        ci_low=value,
        ci_high=value,
        n=total,
        notes=f"unique={unique}",
        extra={"unique": unique},
    )


def retriever_fail_rate(rows: Sequence[dict[str, Any]]) -> MetricResult:
    """Fraction of rows whose injected context contains the retrieval-miss sentinel."""
    total = len(rows)
    if total == 0:
        return MetricResult(
            name="retriever_fail_rate",
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            notes="no rows",
        )
    hits_list: list[float] = []
    for r in rows:
        ctx = r.get("instruction_context") or r.get("response_context") or ""
        hits_list.append(
            1.0 if isinstance(ctx, str) and RETRIEVAL_FAIL_SENTINEL in ctx else 0.0
        )
    lo, hi = _bootstrap_ci(hits_list, np.mean)
    return MetricResult(
        name="retriever_fail_rate",
        value=float(np.mean(hits_list)),
        ci_low=lo,
        ci_high=hi,
        n=total,
    )


# ---------------------------------------------------------------------------
# Paired-bootstrap on the delta (variant - baseline)
# ---------------------------------------------------------------------------


def _per_row_values(
    rows: Sequence[dict[str, Any]],
    kind: str,
) -> np.ndarray | None:
    if kind == "retriever_fail":
        out = []
        for r in rows:
            ctx = r.get("instruction_context") or r.get("response_context") or ""
            out.append(
                1.0
                if isinstance(ctx, str) and RETRIEVAL_FAIL_SENTINEL in ctx
                else 0.0
            )
        return np.asarray(out, dtype=float)
    if kind == "length":
        out = [len(_final_assistant(r)) for r in rows]
        return np.asarray(out, dtype=float)
    return None


def paired_delta_ci(
    base: Sequence[dict[str, Any]],
    var: Sequence[dict[str, Any]],
    kind: str,
    seed: int = _DEFAULT_RNG_SEED,
    resamples: int = _BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float] | None:
    """Return (delta, ci_low, ci_high) for a row-aligned per-row metric or None."""
    b = _per_row_values(base, kind)
    v = _per_row_values(var, kind)
    if b is None or v is None:
        return None
    m = min(len(b), len(v))
    if m == 0:
        return None
    b = b[:m]
    v = v[:m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(resamples, m))
    deltas = v[idx].mean(axis=1) - b[idx].mean(axis=1)
    delta = float(v.mean() - b.mean())
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return delta, lo, hi


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


ALL_METRICS: list[tuple[str, Callable[[Sequence[dict[str, Any]]], MetricResult]]] = [
    ("self_bleu", lambda rs: self_bleu(rs, n=4)),
    ("distinct_2", lambda rs: distinct_n(rs, n=2)),
    ("length_entropy", length_entropy),
    ("dedup_rate", dedup_rate),
    ("nli_entailment_rate", nli_entailment_rate),
    ("judge_grade_pass_rate", judge_grade_dist),
    ("persona_cardinality", persona_cardinality),
    ("distinct_contexts", distinct_contexts),
    ("retriever_fail_rate", retriever_fail_rate),
]


def run_all(rows: Sequence[dict[str, Any]]) -> list[MetricResult]:
    """Evaluate every metric; failures return NaN results with notes."""
    results: list[MetricResult] = []
    for name, fn in ALL_METRICS:
        try:
            results.append(fn(rows))
        except Exception as e:
            logger.exception("metric %s crashed", name)
            results.append(
                MetricResult(
                    name=name,
                    value=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    n=len(rows),
                    notes=f"error: {type(e).__name__}: {e}",
                )
            )
    return results


def metric_direction(name: str) -> str:
    """Direction of improvement for delta flagging ('up', 'down', or 'none')."""
    if name in {"distinct_2", "length_entropy", "persona_cardinality", "distinct_contexts", "judge_grade_pass_rate", "nli_entailment_rate"}:
        return "up"
    if name in {"self_bleu_4", "dedup_rate", "retriever_fail_rate"}:
        return "down"
    return "none"


__all__ = [
    "MetricResult",
    "RETRIEVAL_FAIL_SENTINEL",
    "ALL_METRICS",
    "load_rows",
    "self_bleu",
    "distinct_n",
    "length_entropy",
    "dedup_rate",
    "nli_entailment_rate",
    "judge_grade_dist",
    "persona_cardinality",
    "distinct_contexts",
    "retriever_fail_rate",
    "run_all",
    "paired_delta_ci",
    "metric_direction",
]
