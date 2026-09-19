#!/usr/bin/env python3
"""CLEAR runner for free-text MedRBench and HealthBench."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.common.io import make_run_dir, write_json
from methods.common.llm import call_chat_model
from methods.clear.agent import _call_chat_search, _resolve_search_provider
from methods.clear.prompts import format_evidence
from methods.clear.source_quality import summarize_source_quality
from methods.clear.structured import normalize_evidence, parse_json_object

from methods.common.benchmarks import healthbench as hb
from methods.common.benchmarks import medrbench as mrb


SEARCH_PROMPT_VERSION = "free_text_clear_search_v1"
ONLINE_ANSWER_PROMPT_VERSION = "free_text_clear_online_answer_v1"
REVIEWER_PROMPT_VERSION = "free_text_clear_reviewer_v1"


def _resolve_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (PROJECT_ROOT / path_obj).resolve()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _by_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}


def _cache_key(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _candidate_text(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return ""
    return _normalize_text(str(row.get("pred") or row.get("answer_text") or row.get("raw_response") or ""))


def _build_search_messages(*, benchmark: str, task: str, row_id: str, prompt_text: str, max_sources: int) -> List[Dict[str, str]]:
    user = (
        "Use web search to find reliable medical evidence for this free-text medical benchmark item.\n"
        "Prefer clinical guidelines, official medical references, review articles, major journals, and specialty societies.\n"
        "Focus on evidence that helps produce or verify a concise final answer. Do not rely only on memory.\n\n"
        f"Benchmark: {benchmark}\n"
        f"Task/subset: {task}\n"
        f"ID: {row_id}\n\n"
        f"Question or case:\n{prompt_text}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "evidence_summary": "short synthesis of the best evidence",\n'
        '  "key_facts": ["fact relevant to the final answer", "..."],\n'
        '  "sources": [\n'
        '    {"title": "source title", "url": "https://...", "claim": "supported claim", "relevance": "why it matters"}\n'
        "  ],\n"
        '  "search_confidence": "high|medium|low"\n'
        "}\n\n"
        f"Use at most {max_sources} sources."
    )
    return [
        {
            "role": "system",
            "content": "You are a medical research assistant. Search the web and return valid JSON only.",
        },
        {"role": "user", "content": user},
    ]


def _search_evidence(
    *,
    benchmark: str,
    task: str,
    row_id: str,
    prompt_text: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_dir = _resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_provider = _resolve_search_provider(args.search_provider)
    payload = {
        "prompt_version": SEARCH_PROMPT_VERSION,
        "benchmark": benchmark,
        "task": task,
        "id": row_id,
        "prompt_text": prompt_text,
        "search_provider": resolved_provider,
        "search_model": args.search_model,
        "search_context_size": args.search_context_size,
        "max_sources": args.max_sources,
    }
    path = cache_dir / f"{_cache_key(payload)}.json"
    if args.reuse_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["evidence"], {
            "search_cache_hit": True,
            "search_cache_path": str(path),
            "search_provider": resolved_provider,
            "raw_search_response": cached.get("raw_search_response", ""),
            "search_annotations": cached.get("annotations", []),
        }

    messages = _build_search_messages(
        benchmark=benchmark,
        task=task,
        row_id=row_id,
        prompt_text=prompt_text,
        max_sources=args.max_sources,
    )
    last_error = ""
    for attempt in range(max(1, args.max_search_retries + 1)):
        try:
            raw_response, annotations = _call_chat_search(
                messages=messages,
                provider=resolved_provider,
                model=args.search_model,
                timeout_s=args.search_timeout_s,
                search_context_size=args.search_context_size,
                azure_suffix=args.search_azure_suffix,
            )
            evidence = normalize_evidence(parse_json_object(raw_response), max_sources=args.max_sources)
            record = {"payload": payload, "evidence": evidence, "raw_search_response": raw_response, "annotations": annotations}
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return evidence, {
                "search_cache_hit": False,
                "search_cache_path": str(path),
                "search_provider": resolved_provider,
                "raw_search_response": raw_response,
                "search_annotations": annotations,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt >= args.max_search_retries:
                break
            time.sleep(max(1.0, args.sleep_s))
    raise RuntimeError(f"online search failed: {last_error}")


def _online_messages(base_messages: Sequence[Dict[str, str]], evidence: Dict[str, Any], *, answer_instruction: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a careful clinical reasoning assistant. Use the online evidence only when relevant and reliable. "
                + answer_instruction
            ),
        },
        {"role": "user", "content": f"Online evidence:\n{format_evidence(evidence)}"},
        *list(base_messages),
    ]


def _review_messages(
    *,
    benchmark: str,
    task: str,
    prompt_text: str,
    evidence: Dict[str, Any],
    candidates: Dict[str, str],
    answer_instruction: str,
) -> List[Dict[str, str]]:
    quality = summarize_source_quality(list(evidence.get("sources") or []))
    candidate_text = "\n\n".join(
        f"{name} candidate:\n{text or '[missing]'}"
        for name, text in candidates.items()
    )
    user = (
        "Synthesize the final answer for this free-text medical benchmark item.\n"
        "Compare direct, BM25, MedCPT, and online-search candidates. Prefer a concise answer that is medically correct.\n"
        "Use online evidence when it is aligned and high-quality; ignore irrelevant or weak sources. "
        "Do not mention the candidate names in the final answer.\n\n"
        f"Benchmark: {benchmark}\n"
        f"Task/subset: {task}\n\n"
        f"Question or case:\n{prompt_text}\n\n"
        f"Candidates:\n{candidate_text}\n\n"
        f"Online evidence:\n{format_evidence(evidence)}\n\n"
        f"Source quality summary:\n{json.dumps(quality, ensure_ascii=False)[:3000]}\n\n"
        f"{answer_instruction}"
    )
    return [
        {"role": "system", "content": "You are a conservative medical answer reviewer."},
        {"role": "user", "content": user},
    ]


def _summarize_free_text_results(rows: Sequence[Dict[str, Any]], *, method: str, dataset: str, split: str, model: str, run_dir: Path) -> Dict[str, Any]:
    if dataset == "healthbench":
        scored = [float(row["score"]) for row in rows if row.get("score") is not None]
        metrics: Dict[str, List[float]] = {}
        for row in rows:
            for key, value in (row.get("metrics") or {}).items():
                try:
                    metrics.setdefault(str(key), []).append(float(value))
                except Exception:
                    pass
        metric_means = {key: sum(values) / len(values) for key, values in metrics.items() if values}
        return {
            "method": method,
            "dataset": dataset,
            "split": split,
            "num_samples": len(rows),
            "failed_samples": sum(1 for row in rows if row.get("score") is None),
            "score": (sum(scored) / len(scored)) if scored else 0.0,
            "overall_score": metric_means.get("overall_score", 0.0),
            "metrics": metric_means,
            "model": model,
            "run_dir": str(run_dir),
        }
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "method": method,
        "dataset": dataset,
        "split": split,
        "num_samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "failed_samples": sum(1 for row in rows if (row.get("trace") or {}).get("error")),
        "model": model,
        "run_dir": str(run_dir),
    }


def _run_medrbench_task(
    *,
    task: str,
    case_path: Path,
    case_limit: int,
    baseline_dirs: Dict[str, Path],
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    case_map = mrb._load_case_map(case_path)
    selected = mrb._select_cases(case_map, case_limit)
    baselines = {
        name: _by_id(_load_jsonl(path / f"{task}_results.jsonl"))
        for name, path in baseline_dirs.items()
    }
    results_path = run_dir / f"{task}_results.jsonl"
    summary_path = run_dir / f"{task}_summary.json"
    if not args.run_dir or not results_path.exists():
        results_path.write_text("", encoding="utf-8")
    results = _load_jsonl(results_path)
    completed_ids = {str(row.get("id", "")).strip() for row in results if str(row.get("id", "")).strip()}

    for idx, (case_id, case_row) in enumerate(selected, start=1):
        if case_id in completed_ids:
            continue
        case_summary, analysis, gold = mrb._case_parts(task, case_row)
        prompt_text = case_summary if task == "diagnose" else f"{case_summary}\n\n{analysis}"
        pred = ""
        raw_response = ""
        online_raw = ""
        evidence: Dict[str, Any] = {}
        search_trace: Dict[str, Any] = {}
        judge: Dict[str, Any] = {}
        error = ""
        candidates = {name: _candidate_text(rows.get(case_id)) for name, rows in baselines.items()}
        try:
            try:
                evidence, search_trace = _search_evidence(
                    benchmark="medrbench",
                    task=task,
                    row_id=case_id,
                    prompt_text=prompt_text,
                    args=args,
                )
            except Exception as exc:
                evidence = {
                    "evidence_summary": "",
                    "key_facts": [],
                    "sources": [],
                    "search_confidence": "low",
                }
                search_trace = {"search_error": str(exc)}
            answer_instruction = "Return only the final diagnosis phrase." if task == "diagnose" else "Return only the final treatment plan."
            if evidence.get("sources") or evidence.get("evidence_summary"):
                online_raw = call_chat_model(
                    _online_messages(mrb._base_messages(task, case_id, case_row), evidence, answer_instruction=answer_instruction),
                    provider=args.provider,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    azure_suffix=args.azure_suffix,
                )
                candidates["online"] = mrb._extract_answer(online_raw)
            raw_response = call_chat_model(
                _review_messages(
                    benchmark="medrbench",
                    task=task,
                    prompt_text=prompt_text,
                    evidence=evidence,
                    candidates=candidates,
                    answer_instruction=answer_instruction,
                ),
                provider=args.provider,
                model=args.model,
                timeout_s=args.timeout_s,
                azure_suffix=args.azure_suffix,
            )
            pred = mrb._extract_answer(raw_response)
            if not pred:
                raise RuntimeError("reviewer returned empty answer")
            judge = mrb._judge_prediction(
                task=task,
                case_summary=case_summary,
                gold=gold,
                pred=pred,
                judge_model=args.judge_model,
                judge_timeout_s=args.judge_timeout_s,
                judge_azure_suffix=args.judge_azure_suffix,
            )
        except Exception as exc:
            error = str(exc)

        result = {
            "id": case_id,
            "pred": pred or None,
            "gold": gold,
            "correct": bool(judge.get("correct")) if judge else False,
            "raw_response": raw_response,
            "candidates": candidates,
            "online_raw_response": online_raw,
            "evidence": evidence,
            "trace": {
                "method": "clear_free_text",
                "task": task,
                "model": args.model,
                "judge_model": args.judge_model,
                "prompt_versions": {
                    "search": SEARCH_PROMPT_VERSION,
                    "online_answer": ONLINE_ANSWER_PROMPT_VERSION,
                    "reviewer": REVIEWER_PROMPT_VERSION,
                },
                **search_trace,
            },
        }
        if judge:
            result["trace"]["judge_explanation"] = judge.get("explanation", "")
            result["trace"]["judge_raw_response"] = judge.get("raw_response", "")
        if error:
            result["trace"]["error"] = error
        results.append(result)
        completed_ids.add(case_id)
        _append_jsonl(results_path, result)
        print(f"[medrbench {task} {idx}/{len(selected)}] id={case_id} correct={result['correct']}", flush=True)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    summary = _summarize_free_text_results(
        results,
        method="clear_free_text",
        dataset=f"medrbench_{task}",
        split="test",
        model=args.model,
        run_dir=run_dir,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def run_medrbench(args: argparse.Namespace) -> None:
    output_root = _resolve_path(args.output_root)
    run_dir = _resolve_path(args.run_dir) if args.run_dir else make_run_dir(
        output_root,
        method="clear_free_text",
        dataset="medrbench",
        split=args.task,
        model=args.model,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_dirs = {
        "direct": _resolve_path(args.direct_run_dir),
        "bm25": _resolve_path(args.bm25_run_dir),
        "medcpt": _resolve_path(args.medcpt_run_dir),
    }
    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["baseline_dirs"] = {key: str(path) for key, path in baseline_dirs.items()}
    write_json(run_dir / "config.json", config)

    task_summaries: Dict[str, Dict[str, Any]] = {}
    if args.task in {"diagnose", "both"}:
        task_summaries["diagnose"] = _run_medrbench_task(
            task="diagnose",
            case_path=_resolve_path(args.diag_path),
            case_limit=args.diag_cases,
            baseline_dirs=baseline_dirs,
            run_dir=run_dir,
            args=args,
        )
    if args.task in {"treatment", "both"}:
        task_summaries["treatment"] = _run_medrbench_task(
            task="treatment",
            case_path=_resolve_path(args.treat_path),
            case_limit=args.treat_cases,
            baseline_dirs=baseline_dirs,
            run_dir=run_dir,
            args=args,
        )
    n = sum(int(item.get("num_samples", 0)) for item in task_summaries.values())
    correct = sum(int(item.get("correct", 0)) for item in task_summaries.values())
    failed = sum(int(item.get("failed_samples", 0)) for item in task_summaries.values())
    summary = {
        "method": "clear_free_text",
        "dataset": "medrbench",
        "split": args.task,
        "num_samples": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "failed_samples": failed,
        "model": args.model,
        "judge_model": args.judge_model,
        "run_dir": str(run_dir),
        "tasks": task_summaries,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def run_healthbench(args: argparse.Namespace) -> None:
    data_path = _resolve_path(args.data_path)
    subset = hb._resolve_subset_name(data_path, args.subset)
    output_root = _resolve_path(args.output_root)
    run_dir = _resolve_path(args.run_dir) if args.run_dir else make_run_dir(
        output_root,
        method="clear_free_text",
        dataset="healthbench",
        split=subset,
        model=args.model,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    if not args.run_dir or not results_path.exists():
        results_path.write_text("", encoding="utf-8")
    rows = hb._load_jsonl(data_path, limit=args.limit)
    baselines = {
        "direct": _by_id(_load_jsonl(_resolve_path(args.direct_run_dir) / "results.jsonl")),
        "bm25": _by_id(_load_jsonl(_resolve_path(args.bm25_run_dir) / "results.jsonl")),
        "medcpt": _by_id(_load_jsonl(_resolve_path(args.medcpt_run_dir) / "results.jsonl")),
    }
    config = vars(args).copy()
    config.update({"run_dir": str(run_dir), "data_path": str(data_path), "subset": subset})
    write_json(run_dir / "config.json", config)

    results = _load_jsonl(results_path)
    completed_ids = {str(row.get("id", "")).strip() for row in results if str(row.get("id", "")).strip()}
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("prompt_id", f"row_{idx - 1}"))
        if row_id in completed_ids:
            continue
        prompt_messages = hb._normalize_messages(row.get("prompt") or [])
        prompt_text = hb._latest_user_text(prompt_messages)
        rubrics = [hb.RubricItem.from_dict(item) for item in row.get("rubrics", [])]
        example_tags = [str(tag).strip() for tag in row.get("example_tags", []) if str(tag).strip()]
        answer_text = ""
        raw_response = ""
        online_raw = ""
        evidence: Dict[str, Any] = {}
        search_trace: Dict[str, Any] = {}
        metrics: Dict[str, float] = {}
        rubric_rows: List[Dict[str, Any]] = []
        score = None
        error = ""
        candidates = {name: _candidate_text(rows_by_id.get(row_id)) for name, rows_by_id in baselines.items()}
        try:
            try:
                evidence, search_trace = _search_evidence(
                    benchmark="healthbench",
                    task=subset,
                    row_id=row_id,
                    prompt_text=prompt_text,
                    args=args,
                )
            except Exception as exc:
                evidence = {
                    "evidence_summary": "",
                    "key_facts": [],
                    "sources": [],
                    "search_confidence": "low",
                }
                search_trace = {"search_error": str(exc)}
            answer_instruction = "Answer the user directly and completely."
            if evidence.get("sources") or evidence.get("evidence_summary"):
                online_raw = call_chat_model(
                    _online_messages(prompt_messages, evidence, answer_instruction=answer_instruction),
                    provider=args.provider,
                    model=args.model,
                    timeout_s=args.timeout_s,
                    azure_suffix=args.azure_suffix,
                )
                candidates["online"] = _normalize_text(online_raw)
            raw_response = call_chat_model(
                _review_messages(
                    benchmark="healthbench",
                    task=subset,
                    prompt_text=prompt_text,
                    evidence=evidence,
                    candidates=candidates,
                    answer_instruction=answer_instruction,
                ),
                provider=args.provider,
                model=args.model,
                timeout_s=args.timeout_s,
                azure_suffix=args.azure_suffix,
            )
            answer_text = _normalize_text(raw_response)
            if not answer_text:
                raise RuntimeError("reviewer returned empty answer")
            grades = hb._judge_rubrics(
                prompt_messages=prompt_messages,
                response_text=answer_text,
                rubrics=rubrics,
                judge_model=args.judge_model,
                judge_timeout_s=args.judge_timeout_s,
                judge_azure_suffix=args.judge_azure_suffix,
            )
            score, metrics, rubric_rows = hb._score_sample(
                example_tags=example_tags,
                rubrics=rubrics,
                grades=grades,
            )
        except Exception as exc:
            error = str(exc)

        result = {
            "id": row_id,
            "score": score,
            "metrics": metrics,
            "rubric_items": rubric_rows,
            "answer_text": answer_text,
            "raw_response": raw_response,
            "candidates": candidates,
            "online_raw_response": online_raw,
            "evidence": evidence,
            "trace": {
                "method": "clear_free_text",
                "model": args.model,
                "judge_model": args.judge_model,
                "prompt_versions": {
                    "search": SEARCH_PROMPT_VERSION,
                    "online_answer": ONLINE_ANSWER_PROMPT_VERSION,
                    "reviewer": REVIEWER_PROMPT_VERSION,
                },
                **search_trace,
            },
        }
        if error:
            result["trace"]["error"] = error
        results.append(result)
        completed_ids.add(row_id)
        _append_jsonl(results_path, result)
        status = "ERROR" if score is None else f"score={score:.4f}"
        print(f"[healthbench {idx}/{len(rows)}] id={row_id} {status}", flush=True)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    summary = _summarize_free_text_results(
        results,
        method="clear_free_text",
        dataset="healthbench",
        split=subset,
        model=args.model,
        run_dir=run_dir,
    )
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "metrics.json", summary.get("metrics", {}))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["medrbench", "healthbench"], required=True)
    parser.add_argument("--task", choices=["diagnose", "treatment", "both"], default="both")
    parser.add_argument("--diag-path", default=str(mrb.DEFAULT_DIAG_PATH))
    parser.add_argument("--treat-path", default=str(mrb.DEFAULT_TREAT_PATH))
    parser.add_argument("--diag-cases", type=int, default=5)
    parser.add_argument("--treat-cases", type=int, default=5)
    parser.add_argument("--data-path", default=str(hb.DEFAULT_SUBSET_PATHS["hard"]))
    parser.add_argument("--subset", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--direct-run-dir", required=True)
    parser.add_argument("--bm25-run-dir", required=True)
    parser.add_argument("--medcpt-run-dir", required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--provider", choices=["auto", "azure", "openai", "qwen"], default="azure")
    parser.add_argument("--model", default="o3-mini")
    parser.add_argument("--azure-suffix", default="_2")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-azure-suffix", default="")
    parser.add_argument("--judge-timeout-s", type=int, default=120)
    parser.add_argument("--search-provider", choices=["auto", "azure", "openai"], default="azure")
    parser.add_argument("--search-model", default="gpt-4o")
    parser.add_argument("--search-azure-suffix", default="")
    parser.add_argument("--search-timeout-s", type=int, default=120)
    parser.add_argument("--search-context-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--max-search-retries", type=int, default=1)
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "temp" / "free_text_clear_cache"))
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--sleep-s", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        if args.benchmark == "medrbench":
            run_medrbench(args)
        else:
            run_healthbench(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
