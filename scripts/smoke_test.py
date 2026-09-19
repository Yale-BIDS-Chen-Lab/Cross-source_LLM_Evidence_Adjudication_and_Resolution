#!/usr/bin/env python3
"""Offline smoke test for CLEAR.

The test avoids model APIs, online search, and retrieval indexes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summary import build_summary
from methods.clear.source_quality import summarize_source_quality
from methods.clear.structured import normalize_evidence
from methods.common.data import load_jsonl, normalize_sample, sample_is_correct


def main() -> int:
    example_path = PROJECT_ROOT / "data" / "examples" / "mcq_example.jsonl"
    rows = load_jsonl(example_path)
    if len(rows) != 1:
        raise RuntimeError(f"expected one example row in {example_path}, found {len(rows)}")

    sample = normalize_sample(rows[0], dataset="demo", split="example", row_index=0)
    pred = sample["gold"]
    if not sample_is_correct(sample, pred):
        raise RuntimeError("example answer normalization failed")

    evidence = normalize_evidence(
        {
            "evidence_summary": "Synthetic evidence for smoke testing.",
            "key_facts": ["This row is not benchmark data."],
            "sources": [
                {
                    "title": "Synthetic source",
                    "url": "https://example.org/synthetic",
                    "claim": "Used only for smoke testing.",
                    "relevance": "high",
                }
            ],
            "search_confidence": "medium",
        },
        max_sources=6,
    )
    quality = summarize_source_quality(evidence["sources"])

    summary = build_summary(
        [{"id": sample["id"], "pred": pred, "gold": sample["gold"], "correct": True, "trace": {}}],
        method="smoke",
        dataset="demo",
        split="example",
        model="offline",
        run_dir=Path("runs") / "smoke",
        extra={"source_quality": quality},
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
