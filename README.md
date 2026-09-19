# CLEAR: Cross-Source Evidence Adjudication for Large Language Models in Medicine

This repository provides the implementation for
["CLEAR: Cross-Source Evidence Adjudication for Large Language Models in
Medicine"](https://arxiv.org/abs/2609.16301v1).

CLEAR combines parametric model answers, local retrieval evidence, online
search evidence, and verifier-based answer selection.

## Included Methods

| Directory | Method |
|---|---|
| `methods/direct/` | Direct answering baseline |
| `methods/BM25/` | BM25 retrieval baseline |
| `methods/MedCPT/` | MedCPT retrieval baseline |
| `methods/clear/` | CLEAR |

## Repository Scope

Benchmark data, model outputs, retrieval caches, logs, API keys, and service
account files are not tracked by this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your model and search credentials before running API-backed
commands.

## Data

See `data/README.md` for the expected JSONL schema and local layout.

## Example Commands

Offline smoke test:

```bash
python scripts/smoke_test.py
```

Direct baseline:

```bash
python methods/direct/run_direct.py \
  --data-path data/examples/mcq_example.jsonl \
  --dataset demo \
  --split example \
  --provider openai \
  --model gpt-4o-mini
```

BM25 baseline:

```bash
python methods/BM25/run_bm25.py \
  --data-path data/MCQ/medqa/test.jsonl \
  --dataset medqa \
  --split test_full \
  --provider openai \
  --model gpt-4o-mini \
  --db-dir /path/to/local/medrag_corpora
```

CLEAR:

```bash
python methods/clear/run_clear.py \
  --data-path data/MCQ/medqa/test.jsonl \
  --dataset medqa \
  --split test_full \
  --provider openai \
  --model gpt-4o-mini \
  --search-provider azure \
  --search-model gpt-4o \
  --cache-dir runs/cache/online
```

Qwen can be used through an OpenAI-compatible local server. See
`scripts/qwen_openai_server.py`.

## Outputs

Each run writes `config.json`, `results.jsonl`, and `summary.json` under `runs/`.
These outputs are ignored by git.

## Notes

Retrieval baselines require a local corpus/index. Benchmark data and retrieval
corpora should be obtained from their original sources.

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{wang2026clear,
  title         = {CLEAR: Cross-Source Evidence Adjudication for Large Language Models in Medicine},
  author        = {Wang, Shuai and Zhao, Yize and Chen, Qingyu},
  year          = {2026},
  eprint        = {2609.16301},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2609.16301},
  url           = {https://arxiv.org/abs/2609.16301}
}
```

Machine-readable citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

This study is supported by the National Institutes of Health National Library
of Medicine under Award Number R01LM014604.
