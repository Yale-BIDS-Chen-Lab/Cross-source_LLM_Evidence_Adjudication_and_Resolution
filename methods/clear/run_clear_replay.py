#!/usr/bin/env python3
"""Replay CLEAR verification over existing Direct/BM25/online runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.summary import build_summary
from methods.common.data import (
    canonical_split_from_path,
    gold_display,
    infer_dataset_name,
    load_jsonl,
    normalize_sample,
    sample_is_correct,
)
from methods.common.io import make_run_dir, write_json, write_jsonl
from methods.clear.verifier import run_verifier


def load_result_objects(path: Path) -> List[Dict[str, Any]]:
    """Load JSON result objects, tolerating old files with raw Unicode separators."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, dict):
            rows.append(obj)
        idx = next_idx
    return rows


def by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}


def candidate_from_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {"pred": None, "raw_response": "", "trace": {"error": "missing candidate row"}}
    return {
        "pred": row.get("pred"),
        "raw_response": row.get("raw_response", ""),
        "trace": row.get("trace") or {},
    }


def row_correct(row: Optional[Dict[str, Any]]) -> bool:
    return bool(row and row.get("correct"))


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path).resolve()
    dataset = infer_dataset_name(data_path, args.dataset)
    split = args.split or canonical_split_from_path(data_path)
    output_root = Path(args.output_root).resolve()

    raw_rows = load_jsonl(data_path, limit=args.limit)
    samples = [
        normalize_sample(row, dataset=dataset, split=split, row_index=i)
        for i, row in enumerate(raw_rows)
    ]

    direct_by_id = by_id(load_result_objects(Path(args.direct_results).resolve()))
    bm25_by_id = by_id(load_result_objects(Path(args.bm25_results).resolve()))
    online_by_id = by_id(load_result_objects(Path(args.online_results).resolve()))

    run_dir = Path(args.run_dir).resolve() if args.run_dir else make_run_dir(
        output_root,
        method="clear_replay",
        dataset=dataset,
        split=split,
        model=args.model,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    write_json(
        config_path,
        {
            **vars(args),
            "data_path": str(data_path),
            "dataset": dataset,
            "split": split,
            "output_root": str(output_root),
            "direct_results": str(Path(args.direct_results).resolve()),
            "bm25_results": str(Path(args.bm25_results).resolve()),
            "online_results": str(Path(args.online_results).resolve()),
        },
    )

    results: List[Dict[str, Any]] = load_result_objects(results_path) if args.run_dir and results_path.exists() else []
    completed_ids = {str(row.get("id", "")).strip() for row in results if str(row.get("id", "")).strip()}
    if completed_ids:
        print(f"[resume] loaded {len(completed_ids)} existing replay results from {results_path}", flush=True)
    decision_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for row in results:
        verifier = row.get("verifier") or {}
        decision_counts[str(verifier.get("decision", ""))] += 1
        mode_counts[str(verifier.get("mode", ""))] += 1

    for idx, sample in enumerate(samples, start=1):
        if sample["id"] in completed_ids:
            continue
        direct_row = direct_by_id.get(sample["id"])
        bm25_row = bm25_by_id.get(sample["id"])
        online_row = online_by_id.get(sample["id"])
        online_evidence = (online_row or {}).get("evidence") or {}
        local_docs = ((bm25_row or {}).get("trace") or {}).get("retrieved_docs") or []

        verifier = run_verifier(
            sample,
            direct_candidate=candidate_from_row(direct_row),
            bm25_candidate=candidate_from_row(bm25_row),
            online_candidate=candidate_from_row(online_row),
            online_evidence=online_evidence,
            local_docs=local_docs,
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            azure_suffix=args.azure_suffix,
            mode=args.verifier_mode,
            max_local_docs=args.max_local_docs,
            enable_second_search=args.enable_second_search,
            second_search_provider=args.second_search_provider,
            second_search_model=args.second_search_model,
            second_search_azure_suffix=args.second_search_azure_suffix,
            second_search_timeout_s=args.second_search_timeout_s,
            second_search_max_retries=args.second_search_max_retries,
            second_search_max_sources=args.second_search_max_sources,
            second_search_context_size=args.second_search_context_size,
            second_search_cache_dir=Path(args.second_search_cache_dir).resolve(),
            reuse_second_search_cache=args.reuse_second_search_cache,
            second_search_sleep_s=args.sleep_s,
            second_search_quality_threshold=args.second_search_quality_threshold,
            enable_online_challenge_audit=args.enable_online_challenge_audit,
        )
        pred = verifier.get("final_pred")
        correct = sample_is_correct(sample, pred)
        decision_counts[str(verifier.get("decision", ""))] += 1
        mode_counts[str(verifier.get("mode", ""))] += 1

        result = {
            "id": sample["id"],
            "pred": pred,
            "gold": gold_display(sample),
            "correct": correct,
            "raw_response": verifier.get("raw_response", ""),
            "candidates": {
                "direct": {
                    "pred": (direct_row or {}).get("pred"),
                    "correct": row_correct(direct_row),
                    "trace": (direct_row or {}).get("trace") or {},
                },
                "bm25": {
                    "pred": (bm25_row or {}).get("pred"),
                    "correct": row_correct(bm25_row),
                    "trace": (bm25_row or {}).get("trace") or {},
                },
                "online": {
                    "pred": (online_row or {}).get("pred"),
                    "correct": row_correct(online_row),
                    "trace": (online_row or {}).get("trace") or {},
                },
            },
            "evidence": online_evidence,
            "verifier": verifier,
            "trace": {
                "method": "clear_replay",
                "model": args.model,
                "retriever": "direct+bm25+online_search",
                "verifier_mode": args.verifier_mode,
            },
        }
        if len(sample.get("gold_options") or []) > 1:
            result["gold_options"] = sample["gold_options"]
        if verifier.get("error"):
            result["trace"]["verifier_error"] = verifier["error"]
        results.append(result)
        completed_ids.add(sample["id"])
        write_jsonl(results_path, results)
        print(
            f"[{idx}/{len(samples)}] id={sample['id']} pred={pred} gold={gold_display(sample)} "
            f"correct={correct} decision={verifier.get('decision')}",
            flush=True,
        )
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    direct_correct = sum(row_correct(direct_by_id.get(sample["id"])) for sample in samples)
    bm25_correct = sum(row_correct(bm25_by_id.get(sample["id"])) for sample in samples)
    online_correct = sum(row_correct(online_by_id.get(sample["id"])) for sample in samples)
    second_search_count = sum(1 for row in results if (row.get("verifier") or {}).get("targeted_second_search"))
    second_search_allowed = sum(
        1
        for row in results
        if ((row.get("verifier") or {}).get("targeted_second_search") or {}).get("allow_override")
    )
    guard_applied = sum(1 for row in results if (row.get("verifier") or {}).get("guard_applied"))
    online_challenge_count = sum(1 for row in results if (row.get("verifier") or {}).get("online_challenge_audit"))
    online_challenge_allowed = sum(
        1
        for row in results
        if ((row.get("verifier") or {}).get("online_challenge_audit") or {}).get("allow_override")
    )
    online_challenge_applied = sum(1 for row in results if (row.get("verifier") or {}).get("online_challenge_applied"))
    summary = build_summary(
        results,
        method="clear_replay",
        dataset=dataset,
        split=split,
        model=args.model,
        run_dir=run_dir,
        retriever="direct+bm25+online_search",
        extra={
            "verifier_mode": args.verifier_mode,
            "direct_correct": direct_correct,
            "bm25_correct": bm25_correct,
            "online_correct": online_correct,
            "direct_accuracy": direct_correct / len(samples) if samples else 0.0,
            "bm25_accuracy": bm25_correct / len(samples) if samples else 0.0,
            "online_accuracy": online_correct / len(samples) if samples else 0.0,
            "decision_counts": dict(decision_counts),
            "verifier_mode_counts": dict(mode_counts),
            "enable_second_search": bool(args.enable_second_search),
            "second_search_count": second_search_count,
            "second_search_allowed": second_search_allowed,
            "guard_applied": guard_applied,
            "enable_online_challenge_audit": bool(args.enable_online_challenge_audit),
            "online_challenge_count": online_challenge_count,
            "online_challenge_allowed": online_challenge_allowed,
            "online_challenge_applied": online_challenge_applied,
        },
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay the CLEAR verifier over existing candidate runs.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--direct-results", required=True)
    parser.add_argument("--bm25-results", required=True)
    parser.add_argument("--online-results", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--run-dir", default="", help="Resume/write results in a fixed run directory.")
    parser.add_argument("--provider", choices=["auto", "azure", "openai", "qwen"], default="auto")
    parser.add_argument("--model", default="o3-mini", help="Verifier model/deployment name.")
    parser.add_argument("--azure-suffix", default=None)
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--verifier-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--max-local-docs", type=int, default=6)
    parser.add_argument("--enable-second-search", action="store_true")
    parser.add_argument("--second-search-provider", choices=["auto", "azure", "openai", "tavily"], default="auto")
    parser.add_argument(
        "--second-search-model",
        default=os.getenv("AZURE_WEBSEARCH_DEPLOYMENT") or os.getenv("AZURE_WEBSEARCH_MODEL") or "gpt-4o",
    )
    parser.add_argument("--second-search-azure-suffix", default="")
    parser.add_argument("--second-search-timeout-s", type=int, default=120)
    parser.add_argument("--second-search-context-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--second-search-max-sources", type=int, default=6)
    parser.add_argument("--second-search-max-retries", type=int, default=1)
    parser.add_argument(
        "--second-search-cache-dir",
        default=str(PROJECT_ROOT / "temp" / "clear_second_search_cache"),
    )
    parser.add_argument("--reuse-second-search-cache", action="store_true")
    parser.add_argument("--second-search-quality-threshold", type=float, default=0.8)
    parser.add_argument("--enable-online-challenge-audit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    try:
        run(build_arg_parser().parse_args())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
