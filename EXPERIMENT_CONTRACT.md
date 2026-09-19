# Experiment Contract

This repository contains CLEAR and three baselines:

| Method | Purpose |
|---|---|
| `direct` | Answer from the question and options only. |
| `bm25` | Retrieve local BM25 evidence, then answer. |
| `medcpt` | Retrieve local MedCPT evidence, then answer. |
| `clear` | Run CLEAR across parametric, local, and dynamic evidence sources. |

## Input Schema

MCQ runners consume one JSON object per line:

```json
{
  "realidx": "0",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_idx": "A"
}
```

The loader normalizes rows into:

```json
{
  "dataset": "medqa",
  "split": "test_full",
  "id": "0",
  "question": "...",
  "options": {"A": "..."},
  "gold": "A",
  "gold_options": ["A"],
  "task_type": "mcq"
}
```

## Output Schema

Each run writes:

```text
runs/<timestamp>_<method>_<dataset>_<split>_<model>/
  config.json
  results.jsonl
  summary.json
```

Each result row includes:

```json
{
  "id": "0",
  "pred": "A",
  "gold": "A",
  "correct": true,
  "raw_response": "...",
  "trace": {"method": "direct", "model": "o3-mini"}
}
```

CLEAR rows also include candidate predictions, online evidence, source quality
fields, and verifier output.

## Metrics

Accuracy is `correct / num_samples`. Failed samples are retained in
`results.jsonl` and counted as incorrect.
