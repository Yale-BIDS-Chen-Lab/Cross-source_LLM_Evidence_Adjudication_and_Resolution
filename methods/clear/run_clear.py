#!/usr/bin/env python3
"""CLEAR: Direct + BM25 + online evidence + verifier."""

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
from methods.BM25.retriever import (
    BM25Retriever,
    DEFAULT_BM25_CORPORA,
    DEFAULT_DB_DIR,
    parse_corpus_list,
    summarize_retrieved_docs,
)
from methods.common.data import (
    canonical_split_from_path,
    gold_display,
    infer_dataset_name,
    load_jsonl,
    normalize_sample,
    sample_is_correct,
)
from methods.common.io import make_run_dir, write_json, write_jsonl
from methods.common.llm import call_chat_model
from methods.common.parser import parse_answer
from methods.common.prompt import build_direct_prompt, build_retrieval_prompt
from methods.clear.agent import search_evidence
from methods.clear.prompts import ANSWER_PROMPT_VERSION, SEARCH_PROMPT_VERSION, build_answer_messages
from methods.clear.structured import evidence_has_content
from methods.clear.verifier import run_verifier


def _empty_candidate(error: str = "") -> Dict[str, Any]:
    trace: Dict[str, Any] = {}
    if error:
        trace["error"] = error
    return {"pred": None, "raw_response": "", "trace": trace}


def _run_direct_candidate(sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    try:
        raw_response = call_chat_model(
            build_direct_prompt(sample),
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            azure_suffix=args.azure_suffix,
        )
        pred = parse_answer(raw_response, sample["options"])
        if pred is None:
            raise RuntimeError("could not parse answer option")
        return {
            "pred": pred,
            "raw_response": raw_response,
            "trace": {"method": "direct", "model": args.model, "retriever": "none"},
        }
    except Exception as exc:
        return _empty_candidate(str(exc))


def _run_bm25_candidate(
    sample: Dict[str, Any],
    *,
    args: argparse.Namespace,
    retriever: BM25Retriever,
    corpora: List[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    snippets: List[Dict[str, Any]] = []
    try:
        snippets, _ = retriever.retrieve(sample["question"], k=args.top_k)
        raw_response = call_chat_model(
            build_retrieval_prompt(sample, snippets),
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            azure_suffix=args.azure_suffix,
        )
        pred = parse_answer(raw_response, sample["options"])
        if pred is None:
            raise RuntimeError("could not parse answer option")
        return (
            {
                "pred": pred,
                "raw_response": raw_response,
                "trace": {
                    "method": "bm25",
                    "model": args.model,
                    "retriever": "BM25",
                    "corpus": ",".join(corpora),
                    "top_k": int(args.top_k),
                    "retrieved_docs": summarize_retrieved_docs(snippets),
                },
            },
            snippets,
        )
    except Exception as exc:
        return _empty_candidate(str(exc)), snippets


def _run_online_candidate(sample: Dict[str, Any], args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    evidence: Dict[str, Any] = {}
    search_trace: Dict[str, Any] = {}
    try:
        evidence, search_trace = search_evidence(
            sample,
            search_provider=args.search_provider,
            search_model=args.search_model,
            search_azure_suffix=args.search_azure_suffix,
            timeout_s=args.search_timeout_s,
            max_retries=args.max_search_retries,
            max_sources=args.max_sources,
            search_context_size=args.search_context_size,
            cache_dir=Path(args.cache_dir).resolve(),
            reuse_cache=args.reuse_search_cache,
            sleep_s=args.sleep_s,
        )
        if not evidence_has_content(evidence):
            raise RuntimeError("online search returned no usable evidence")
        raw_response = call_chat_model(
            build_answer_messages(sample, evidence),
            provider=args.provider,
            model=args.model,
            timeout_s=args.timeout_s,
            azure_suffix=args.azure_suffix,
        )
        pred = parse_answer(raw_response, sample["options"])
        if pred is None:
            raise RuntimeError("could not parse answer option")
        return (
            {
                "pred": pred,
                "raw_response": raw_response,
                "trace": {
                    "method": "online_search",
                    "model": args.model,
                    "retriever": "online_search",
                    "search_provider": args.search_provider,
                    "search_model": args.search_model,
                    "search_context_size": args.search_context_size,
                    "max_sources": args.max_sources,
                    **search_trace,
                },
            },
            evidence,
        )
    except Exception as exc:
        candidate = _empty_candidate(str(exc))
        candidate["trace"].update(
            {
                "method": "online_search",
                "model": args.model,
                "retriever": "online_search",
                "search_provider": args.search_provider,
                "search_model": args.search_model,
                **search_trace,
            }
        )
        return candidate, evidence


def _candidate_correct(sample: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    return sample_is_correct(sample, candidate.get("pred"))


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path).resolve()
    dataset = infer_dataset_name(data_path, args.dataset)
    split = args.split or canonical_split_from_path(data_path)
    output_root = Path(args.output_root).resolve()
    corpora = parse_corpus_list(args.corpus)
    verifier_model = args.verifier_model or args.model

    raw_rows = load_jsonl(data_path, limit=args.limit)
    samples = [
        normalize_sample(row, dataset=dataset, split=split, row_index=i)
        for i, row in enumerate(raw_rows)
    ]
    retriever = BM25Retriever(db_dir=args.db_dir, corpora=corpora)

    run_dir = make_run_dir(output_root, method="clear", dataset=dataset, split=split, model=args.model)
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
            "db_dir": str(Path(args.db_dir).resolve()),
            "corpus": ",".join(corpora),
            "verifier_model_resolved": verifier_model,
            "search_prompt_version": SEARCH_PROMPT_VERSION,
            "answer_prompt_version": ANSWER_PROMPT_VERSION,
        },
    )

    results: List[Dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()

    for idx, sample in enumerate(samples, start=1):
        direct = _run_direct_candidate(sample, args)
        bm25, local_docs = _run_bm25_candidate(sample, args=args, retriever=retriever, corpora=corpora)
        online, evidence = _run_online_candidate(sample, args)

        verifier = run_verifier(
            sample,
            direct_candidate=direct,
            bm25_candidate=bm25,
            online_candidate=online,
            online_evidence=evidence,
            local_docs=local_docs,
            provider=args.verifier_provider,
            model=verifier_model,
            timeout_s=args.verifier_timeout_s,
            azure_suffix=args.verifier_azure_suffix,
            mode=args.verifier_mode,
            max_local_docs=args.max_verifier_local_docs,
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
                "direct": {**direct, "correct": _candidate_correct(sample, direct)},
                "bm25": {**bm25, "correct": _candidate_correct(sample, bm25)},
                "online": {**online, "correct": _candidate_correct(sample, online)},
            },
            "evidence": evidence,
            "verifier": verifier,
            "trace": {
                "method": "clear",
                "model": args.model,
                "verifier_model": verifier_model,
                "retriever": "direct+bm25+online_search",
                "verifier_mode": args.verifier_mode,
            },
        }
        if len(sample.get("gold_options") or []) > 1:
            result["gold_options"] = sample["gold_options"]
        if verifier.get("error"):
            result["trace"]["verifier_error"] = verifier["error"]
        results.append(result)
        write_jsonl(results_path, results)
        print(
            f"[{idx}/{len(samples)}] id={sample['id']} pred={pred} gold={gold_display(sample)} "
            f"correct={correct} direct={direct.get('pred')} bm25={bm25.get('pred')} "
            f"online={online.get('pred')} decision={verifier.get('decision')}",
            flush=True,
        )
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    direct_correct = sum(bool(row["candidates"]["direct"]["correct"]) for row in results)
    bm25_correct = sum(bool(row["candidates"]["bm25"]["correct"]) for row in results)
    online_correct = sum(bool(row["candidates"]["online"]["correct"]) for row in results)
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
        method="clear",
        dataset=dataset,
        split=split,
        model=args.model,
        run_dir=run_dir,
        retriever="direct+bm25+online_search",
        extra={
            "verifier_model": verifier_model,
            "verifier_mode": args.verifier_mode,
            "direct_correct": direct_correct,
            "bm25_correct": bm25_correct,
            "online_correct": online_correct,
            "direct_accuracy": direct_correct / len(results) if results else 0.0,
            "bm25_accuracy": bm25_correct / len(results) if results else 0.0,
            "online_accuracy": online_correct / len(results) if results else 0.0,
            "search_model": args.search_model,
            "search_provider": args.search_provider,
            "search_context_size": args.search_context_size,
            "max_sources": args.max_sources,
            "corpus": ",".join(corpora),
            "top_k": int(args.top_k),
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
    parser = argparse.ArgumentParser(description="Run the CLEAR medical QA pipeline.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--provider", choices=["auto", "azure", "openai", "qwen"], default="auto")
    parser.add_argument("--model", default="o3-mini", help="Candidate answer model/deployment name.")
    parser.add_argument("--azure-suffix", default=None)
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR))
    parser.add_argument(
        "--corpus",
        default=",".join(parse_corpus_list(None, default=DEFAULT_BM25_CORPORA)),
        help="Comma-separated corpus list. Default: pubmed,textbooks,statpearls,wikipedia",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--search-provider", choices=["auto", "azure", "openai", "tavily"], default="auto")
    parser.add_argument(
        "--search-model",
        default=os.getenv("AZURE_WEBSEARCH_DEPLOYMENT") or os.getenv("AZURE_WEBSEARCH_MODEL") or "gpt-4o",
    )
    parser.add_argument("--search-azure-suffix", default="")
    parser.add_argument("--search-timeout-s", type=int, default=120)
    parser.add_argument("--search-context-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--max-search-retries", type=int, default=2)
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "temp" / "clear_search_cache"))
    parser.add_argument("--reuse-search-cache", action="store_true")
    parser.add_argument("--verifier-provider", choices=["auto", "azure", "openai", "qwen"], default="auto")
    parser.add_argument("--verifier-model", default="", help="Verifier model/deployment; defaults to --model.")
    parser.add_argument("--verifier-azure-suffix", default=None)
    parser.add_argument("--verifier-timeout-s", type=int, default=90)
    parser.add_argument("--verifier-mode", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--max-verifier-local-docs", type=int, default=6)
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
    parser.add_argument("--second-search-cache-dir", default=str(PROJECT_ROOT / "temp" / "clear_second_search_cache"))
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
