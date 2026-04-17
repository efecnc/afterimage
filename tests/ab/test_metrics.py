"""Fixture-only tests for scripts/ab/metrics.py (no network, no API calls)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from scripts.ab import metrics


def _row(
    assistant_contents: list[str],
    *,
    context: str | None = None,
    persona: str | None = None,
    context_ids: list[str] | None = None,
    grade: str | None = None,
) -> dict[str, Any]:
    convos: list[dict[str, Any]] = []
    for a in assistant_contents:
        convos.append({"role": "user", "content": "placeholder"})
        convos.append({"role": "assistant", "content": a})
    row: dict[str, Any] = {
        "conversations": convos,
        "instruction_context": context,
        "response_context": None,
        "persona": persona,
        "metadata": {
            "persona_name": persona,
            "context_ids": context_ids or [],
        },
    }
    if grade:
        row["evaluation"] = {"overall_grade": grade}
    return row


# ---------------------------------------------------------------------------
# self_bleu
# ---------------------------------------------------------------------------


def test_self_bleu_identical_corpus_is_high() -> None:
    txt = "the quick brown fox jumps over the lazy dog every single day now"
    rows = [_row([txt]) for _ in range(5)]
    r = metrics.self_bleu(rows, n=4)
    assert r.value > 0.9
    assert r.n == 5


def test_self_bleu_diverse_corpus_is_low() -> None:
    rows = [
        _row(["alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo"]),
        _row(["sun moon star wind rain cloud thunder lightning rainbow horizon dawn"]),
        _row(["apple banana cherry durian elderberry fig grape honeydew imbe jujube"]),
        _row(["pierre marie jean claude bernard francois michel henri louis andre"]),
        _row(["jupiter mars venus saturn neptune mercury pluto earth uranus ceres"]),
    ]
    r = metrics.self_bleu(rows, n=4)
    assert r.value < 0.05


def test_self_bleu_too_few_rows_returns_nan() -> None:
    r = metrics.self_bleu([_row(["hello"])], n=4)
    assert math.isnan(r.value)
    assert "not enough" in r.notes


# ---------------------------------------------------------------------------
# distinct_n
# ---------------------------------------------------------------------------


def test_distinct_n_on_repetition_is_low() -> None:
    rep = "foo bar " * 20
    rows = [_row([rep]) for _ in range(3)]
    r = metrics.distinct_n(rows, n=2)
    assert r.value < 0.1


def test_distinct_n_on_unique_tokens_is_one() -> None:
    uniq = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike"
    rows = [_row([uniq])]
    r = metrics.distinct_n(rows, n=2)
    assert r.value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# length_entropy
# ---------------------------------------------------------------------------


def test_length_entropy_collapsed_is_low() -> None:
    rows = [_row(["same reply"]) for _ in range(12)]
    r = metrics.length_entropy(rows, bins=10)
    assert r.value == 0.0 or r.value < 0.05


def test_length_entropy_uniform_is_high() -> None:
    lengths = [10, 30, 55, 80, 105, 130, 160, 190, 220, 260]
    rows = [_row(["x" * L]) for L in lengths]
    r = metrics.length_entropy(rows, bins=10)
    assert r.value > 0.8


# ---------------------------------------------------------------------------
# dedup_rate
# ---------------------------------------------------------------------------


def test_dedup_rate_on_duplicates() -> None:
    pytest.importorskip("sentence_transformers")
    dup = "Kubernetes rolling updates use maxSurge and maxUnavailable to bound pod churn."
    rows = [_row([dup]) for _ in range(4)] + [
        _row(["Pour-over coffee relies on grind size and pour technique."]),
        _row(["DPO minimizes a log-sigmoid margin between chosen and rejected."]),
    ]
    r = metrics.dedup_rate(rows, threshold=0.92)
    # The four duplicates should all be flagged.
    assert r.value >= 4 / 6 - 1e-6


def test_dedup_rate_skips_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_try_embed", lambda *a, **k: None)
    rows = [_row(["hi"]), _row(["bye"])]
    r = metrics.dedup_rate(rows)
    assert math.isnan(r.value)
    assert "unavailable" in r.notes


# ---------------------------------------------------------------------------
# judge_grade_dist
# ---------------------------------------------------------------------------


def test_judge_grade_dist_basic() -> None:
    rows = (
        [_row(["x"], grade="perfect") for _ in range(3)]
        + [_row(["x"], grade="good") for _ in range(2)]
        + [_row(["x"], grade="needs_improvement")]
        + [_row(["x"], grade="bad")]
    )
    r = metrics.judge_grade_dist(rows)
    assert r.value == pytest.approx(5 / 7)
    dist = r.extra["distribution"]
    assert dist["perfect"] == 3
    assert dist["bad"] == 1


def test_judge_grade_dist_empty() -> None:
    rows = [_row(["x"])]
    r = metrics.judge_grade_dist(rows)
    assert math.isnan(r.value)


# ---------------------------------------------------------------------------
# persona_cardinality
# ---------------------------------------------------------------------------


def test_persona_cardinality() -> None:
    rows = [
        _row(["x"], persona="A"),
        _row(["x"], persona="B"),
        _row(["x"], persona="A"),
        _row(["x"], persona="C"),
    ]
    r = metrics.persona_cardinality(rows)
    assert r.value == pytest.approx(3 / 4)
    assert r.extra["unique"] == 3


# ---------------------------------------------------------------------------
# distinct_contexts
# ---------------------------------------------------------------------------


def test_distinct_contexts_counts_unique_tuples() -> None:
    rows = [
        _row(["x"], context_ids=["a", "b"]),
        _row(["x"], context_ids=["a", "b"]),
        _row(["x"], context_ids=["a"]),
        _row(["x"], context_ids=["c"]),
    ]
    r = metrics.distinct_contexts(rows)
    assert r.extra["unique"] == 3
    assert r.value == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# retriever_fail_rate
# ---------------------------------------------------------------------------


def test_retriever_fail_rate_counts_sentinel() -> None:
    rows = [
        _row(["x"], context="Some good context with details."),
        _row(["x"], context=metrics.RETRIEVAL_FAIL_SENTINEL),
        _row(["x"], context="More grounded context."),
        _row(["x"], context=f"head {metrics.RETRIEVAL_FAIL_SENTINEL} tail"),
    ]
    r = metrics.retriever_fail_rate(rows)
    assert r.value == pytest.approx(0.5)
    assert r.n == 4


# ---------------------------------------------------------------------------
# Bootstrap CI sanity
# ---------------------------------------------------------------------------


def test_bootstrap_ci_does_not_cross_zero_on_clear_signal() -> None:
    hits_all = [1.0] * 30
    hits_none = [0.0] * 30
    rows = [_row(["x"], context=metrics.RETRIEVAL_FAIL_SENTINEL)] * 30
    r_high = metrics.retriever_fail_rate(rows)
    assert r_high.ci_low > 0.5
    # Reference arrays (hits_all/hits_none) keep the symmetry explicit.
    assert all(h == 1.0 for h in hits_all)
    assert all(h == 0.0 for h in hits_none)


def test_paired_delta_ci_retriever_fail() -> None:
    base = [_row(["x"], context=metrics.RETRIEVAL_FAIL_SENTINEL) for _ in range(20)]
    var = [_row(["x"], context="solid context") for _ in range(20)]
    result = metrics.paired_delta_ci(base, var, kind="retriever_fail")
    assert result is not None
    delta, lo, hi = result
    assert delta == pytest.approx(-1.0)
    assert hi <= 0.0


def test_paired_delta_ci_length() -> None:
    base = [_row(["abc"]) for _ in range(10)]
    var = [_row(["abcdefg"]) for _ in range(10)]
    result = metrics.paired_delta_ci(base, var, kind="length")
    assert result is not None
    delta, _lo, _hi = result
    assert delta > 0


def test_paired_delta_ci_unknown_kind_returns_none() -> None:
    rows = [_row(["x"]) for _ in range(3)]
    assert metrics.paired_delta_ci(rows, rows, kind="nope") is None


def test_load_rows_roundtrip(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        '{"conversations": [{"role": "assistant", "content": "a"}]}\n'
        "\n"
        "bad line not json\n"
        '{"conversations": [{"role": "assistant", "content": "b"}]}\n'
    )
    rows = metrics.load_rows(path)
    assert len(rows) == 2


def test_load_rows_missing_path(tmp_path) -> None:
    assert metrics.load_rows(tmp_path / "nope.jsonl") == []


def test_metric_direction_buckets() -> None:
    assert metrics.metric_direction("distinct_2") == "up"
    assert metrics.metric_direction("self_bleu_4") == "down"
    assert metrics.metric_direction("unknown_metric") == "none"


def test_retriever_fail_rate_empty_rows() -> None:
    r = metrics.retriever_fail_rate([])
    assert math.isnan(r.value)
    assert r.n == 0


def test_distinct_contexts_empty_rows() -> None:
    r = metrics.distinct_contexts([])
    assert math.isnan(r.value)


def test_persona_cardinality_empty_rows() -> None:
    r = metrics.persona_cardinality([])
    assert math.isnan(r.value)


def test_nli_without_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the import to behave as if sentence_transformers is missing.
    import builtins

    original = builtins.__import__

    def fake(name, *a, **kw):
        if name == "sentence_transformers":
            raise ImportError("forced")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    rows = [_row(["reply"], context="a context")]
    r = metrics.nli_entailment_rate(rows)
    assert math.isnan(r.value)
    assert "unavailable" in r.notes


def test_nli_without_context_rows_short_circuits() -> None:
    rows = [_row(["reply with no context"])]
    r = metrics.nli_entailment_rate(rows)
    assert math.isnan(r.value)
    assert r.n == 0


def test_run_all_tolerates_missing_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_try_embed", lambda *a, **k: None)
    rows = [
        _row(
            ["some answer"],
            context="small context",
            persona=f"p{i}",
            context_ids=[f"c{i}"],
            grade="good" if i % 2 == 0 else "bad",
        )
        for i in range(5)
    ]
    results = metrics.run_all(rows)
    names = {r.name for r in results}
    # Ensure we ran every metric even with optional deps unavailable.
    assert "dedup_rate" in names
    assert "nli_entailment_rate" in names
    assert "self_bleu_4" in names
