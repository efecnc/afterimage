# AfterImage A/B harness

Small scripts for comparing two AfterImage configs on the same seed.

## Scripts

### `run_ab.py` — run baseline and variant back-to-back

```bash
python -m scripts.ab.run_ab \
  --baseline examples/configs/ab_baseline.yaml \
  --variant examples/configs/ab_variant.yaml \
  --seed 42 --n 200 \
  --out ab_runs/exp1/
```

Writes `baseline.jsonl`, `variant.jsonl`, and `run_meta.json`.

### `probe.py` — single instrumented run

```bash
python -m scripts.ab.probe \
  --config examples/configs/ab_baseline.yaml \
  --n 50 \
  --out ab_runs/probe/
```

Monkey-patches `QdrantRetriever`, `ConversationGenerator.generate_single`,
`QualityGate.evaluate`, and `GenerationMonitor.track_generation` to count
retrieval misses, retries, judge grades, and silent structured-gen failures.
Emits `probe_report.md` and `probe_counters.json`.

### `compare.py` — run every metric, emit markdown

```bash
python -m scripts.ab.compare \
  ab_runs/exp1/baseline.jsonl \
  ab_runs/exp1/variant.jsonl \
  --out ab_runs/exp1/report.md
```

## Metrics

| Metric | Meaning | Good direction |
|---|---|---|
| `self_bleu_4` | Mean 4-gram overlap across assistant turns. | lower |
| `distinct_2` | Unique-bigram ratio per row. | higher |
| `length_entropy` | Shannon entropy over binned reply lengths (normalized 0..1). | higher |
| `dedup_rate` | Fraction of rows whose nearest-neighbor cosine > 0.92 on MiniLM. | lower |
| `nli_entailment_rate` | Fraction of rows where (context, reply) entails (P > 0.5). | higher |
| `judge_grade_pass_rate` | Fraction of rows with `evaluation.overall_grade ∈ {perfect, good}`. | higher |
| `persona_cardinality` | Distinct `metadata.persona_name` strings / N. | higher |
| `distinct_contexts` | Distinct `metadata.context_ids` tuples / N. | higher |
| `retriever_fail_rate` | Fraction of rows whose injected context is the "No relevant context found." sentinel. | lower |

## What "good" looks like

A healthy variant moves diversity metrics (`distinct_2`, `length_entropy`,
`persona_cardinality`, `distinct_contexts`) **up** and pathologies
(`self_bleu_4`, `dedup_rate`, `retriever_fail_rate`) **down**, with
non-crossing 95% CIs. Grade pass rate and entailment should also rise.

## Notes

- Embedding/NLI metrics require `sentence-transformers`. If unavailable they
  return NaN with a note instead of crashing.
- First NLI run downloads the cross-encoder model (~300 MB).
- All runs set `random.seed` and `np.random.seed` from `--seed` before
  building the generator.
