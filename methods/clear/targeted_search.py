"""Targeted second-search checks for disputed CLEAR answers.

In this patch, the search model only gathers source-backed evidence. The
current verifier model (o3/Qwen) performs the adjudication over that evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from methods.common.llm import call_chat_model
from methods.common.prompt import format_options
from methods.clear.agent import _call_chat_search, _resolve_search_provider
from methods.clear.prompts import dumps_for_cache
from methods.clear.source_quality import summarize_source_quality
from methods.clear.structured import parse_json_object


SECOND_SEARCH_PROMPT_VERSION = "clear_second_search_v2_patch_split_judge"
VALID_SUPPORT_SCOPE = {"exact_option", "background_only", "related_task", "unclear"}


def _cache_path(cache_dir: Path, payload: Dict[str, Any]) -> Path:
    key = hashlib.sha256(dumps_for_cache(payload).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _clean_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _normalize_support_rows(raw_rows: Any, *, max_sources: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not isinstance(raw_rows, list):
        return rows
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        row = {
            "title": str(item.get("title", item.get("source", ""))).strip(),
            "url": str(item.get("url", "")).strip(),
            "claim": str(item.get("claim", "")).strip(),
            "option_mapping": str(item.get("option_mapping", "")).strip(),
            "quality": str(item.get("quality", "")).strip().lower(),
        }
        if any(row.values()):
            rows.append(row)
        if len(rows) >= max_sources:
            break
    return rows


def _dedupe_sources(sources: List[Dict[str, Any]], *, max_sources: int) -> List[Dict[str, str]]:
    seen = set()
    rows: List[Dict[str, str]] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", item.get("source", ""))).strip()
        claim = str(item.get("claim", item.get("snippet", ""))).strip()
        key = (url, title, claim)
        if not any(key) or key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "url": url, "claim": claim})
        if len(rows) >= max_sources:
            break
    return rows


def build_targeted_search_messages(
    sample: Dict[str, Any],
    *,
    default_pred: str,
    challenger_pred: str,
    max_sources: int,
) -> List[Dict[str, str]]:
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    default_text = sample["options"].get(default_pred, "")
    challenger_text = sample["options"].get(challenger_pred, "")
    user = (
        "Use web search to gather evidence for a disputed medical multiple-choice answer. "
        "This is evidence collection only: do not adjudicate the winner and do not recommend override.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options ({labels}):\n{format_options(sample['options'])}\n\n"
        f"Default answer to preserve unless clearly wrong: {default_pred}. {default_text}\n"
        f"Challenger answer requesting override: {challenger_pred}. {challenger_text}\n\n"
        "Search specifically for source claims that distinguish these two answers. Collect claims for both "
        "the challenger and the default when available. Keep claims close to what the source says.\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        f'  "default_answer": "{default_pred}",\n'
        f'  "challenger_answer": "{challenger_pred}",\n'
        '  "task_type": "diagnosis|next_test|treatment|prognosis|guideline_or_regulation|other",\n'
        '  "search_queries": ["query used"],\n'
        '  "evidence": [\n'
        '    {"title": "source title", "url": "https://...", "claim": "source-supported claim", '
        '"relevance": "challenger|default|both|background|unclear", "quality": "high|medium|low"}\n'
        "  ],\n"
        '  "notes": "brief description of evidence coverage and limitations"\n'
        "}\n\n"
        f"Use at most {max_sources} total sources. Do not include final answer labels, override decisions, "
        "or claims not grounded in sources."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a skeptical medical evidence adjudicator. Search the web and return strict, "
                "source-backed JSON evidence only. Do not decide which answer is correct."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_second_search_adjudication_messages(
    sample: Dict[str, Any],
    *,
    default_pred: str,
    challenger_pred: str,
    evidence_payload: Dict[str, Any],
    max_sources: int,
) -> List[Dict[str, str]]:
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    default_text = sample["options"].get(default_pred, "")
    challenger_text = sample["options"].get(challenger_pred, "")
    user = (
        "Adjudicate a disputed medical multiple-choice answer using only the supplied second-search evidence. "
        "Do not use external knowledge beyond interpreting the listed source claims.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options ({labels}):\n{format_options(sample['options'])}\n\n"
        f"Default answer to preserve unless clearly wrong: {default_pred}. {default_text}\n"
        f"Challenger answer requesting override: {challenger_pred}. {challenger_text}\n\n"
        f"Second-search evidence JSON:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        f'  "default_answer": "{default_pred}",\n'
        f'  "challenger_answer": "{challenger_pred}",\n'
        '  "task_type": "diagnosis|next_test|treatment|prognosis|guideline_or_regulation|other",\n'
        '  "challenger_support": [\n'
        '    {"title": "source title", "url": "https://...", "claim": "source-supported claim", '
        '"option_mapping": "why this supports the challenger over default", "quality": "high|medium|low"}\n'
        "  ],\n"
        '  "default_support": [\n'
        '    {"title": "source title", "url": "https://...", "claim": "source-supported claim", '
        '"option_mapping": "why this supports the default over challenger", "quality": "high|medium|low"}\n'
        "  ],\n"
        '  "supports_challenger": false,\n'
        '  "refutes_default": false,\n'
        '  "default_not_best": false,\n'
        '  "question_aligned": false,\n'
        '  "unresolved_conflict": true,\n'
        '  "support_scope": "exact_option|background_only|related_task|unclear",\n'
        '  "recommend_override": false,\n'
        '  "reason": "one concise sentence explaining the adjudication",\n'
        '  "sources": [\n'
        '    {"title": "source title", "url": "https://...", "claim": "source-supported claim"}\n'
        "  ]\n"
        "}\n\n"
        f"Use at most {max_sources} total sources. Recommend override only if the supplied evidence directly "
        "supports the challenger, makes the default wrong or less appropriate as the single best answer, and "
        "maps to the exact question and answer options."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative medical QA reviewer. Judge only from the provided second-search "
                "evidence; preserve the default unless the evidence is exact and option-level."
            ),
        },
        {"role": "user", "content": user},
    ]


def second_search_cache_payload(
    sample: Dict[str, Any],
    *,
    search_provider: str,
    search_model: str,
    search_context_size: str,
    max_sources: int,
    default_pred: str,
    challenger_pred: str,
    adjudicator_provider: str,
    adjudicator_model: str,
) -> Dict[str, Any]:
    return {
        "prompt_version": SECOND_SEARCH_PROMPT_VERSION,
        "search_provider": search_provider,
        "search_model": search_model,
        "adjudicator_provider": adjudicator_provider,
        "adjudicator_model": adjudicator_model,
        "search_context_size": search_context_size,
        "max_sources": max_sources,
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "id": sample.get("id"),
        "question": sample.get("question"),
        "options": sample.get("options"),
        "default_pred": default_pred,
        "challenger_pred": challenger_pred,
    }


def normalize_second_search_evidence_payload(
    payload: Dict[str, Any],
    *,
    default_pred: str,
    challenger_pred: str,
    max_sources: int,
    annotations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    evidence = _normalize_support_rows(payload.get("evidence"), max_sources=max_sources)
    raw_sources: List[Dict[str, Any]] = list(evidence)
    if annotations:
        raw_sources.extend(annotations)
    sources = _dedupe_sources(raw_sources, max_sources=max_sources)
    return {
        "default_answer": str(payload.get("default_answer", default_pred)).strip().upper() or default_pred,
        "challenger_answer": str(payload.get("challenger_answer", challenger_pred)).strip().upper() or challenger_pred,
        "task_type": str(payload.get("task_type", "other")).strip().lower() or "other",
        "search_queries": payload.get("search_queries", []) if isinstance(payload.get("search_queries"), list) else [],
        "evidence": evidence,
        "sources": sources,
        "notes": str(payload.get("notes", "")).strip(),
        "source_quality": summarize_source_quality(sources),
    }


def normalize_second_search_payload(
    payload: Dict[str, Any],
    *,
    default_pred: str,
    challenger_pred: str,
    max_sources: int,
    annotations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    challenger_support = _normalize_support_rows(payload.get("challenger_support"), max_sources=max_sources)
    default_support = _normalize_support_rows(payload.get("default_support"), max_sources=max_sources)
    top_sources = _normalize_support_rows(payload.get("sources"), max_sources=max_sources)
    raw_sources: List[Dict[str, Any]] = []
    raw_sources.extend(challenger_support)
    raw_sources.extend(default_support)
    raw_sources.extend(top_sources)
    if annotations:
        raw_sources.extend(annotations)
    sources = _dedupe_sources(raw_sources, max_sources=max_sources)

    support_scope = str(payload.get("support_scope", "unclear")).strip().lower()
    if support_scope not in VALID_SUPPORT_SCOPE:
        support_scope = "unclear"

    quality_report = summarize_source_quality(sources)
    return {
        "default_answer": str(payload.get("default_answer", default_pred)).strip().upper() or default_pred,
        "challenger_answer": str(payload.get("challenger_answer", challenger_pred)).strip().upper() or challenger_pred,
        "task_type": str(payload.get("task_type", "other")).strip().lower() or "other",
        "challenger_support": challenger_support,
        "default_support": default_support,
        "supports_challenger": _clean_bool(payload.get("supports_challenger", False)),
        "refutes_default": _clean_bool(payload.get("refutes_default", False)),
        "default_not_best": _clean_bool(
            payload.get("default_not_best", payload.get("refutes_default", False))
        ),
        "question_aligned": _clean_bool(payload.get("question_aligned", False)),
        "unresolved_conflict": _clean_bool(payload.get("unresolved_conflict", True)),
        "support_scope": support_scope,
        "recommend_override": _clean_bool(payload.get("recommend_override", False)),
        "reason": str(payload.get("reason", "")).strip(),
        "sources": sources,
        "source_quality": quality_report,
    }


def allow_second_search_override(payload: Dict[str, Any]) -> bool:
    source_quality = payload.get("source_quality") or {}
    return bool(
        payload.get("recommend_override")
        and payload.get("supports_challenger")
        and payload.get("default_not_best")
        and payload.get("question_aligned")
        and not payload.get("unresolved_conflict")
        and payload.get("support_scope") == "exact_option"
        and int(source_quality.get("high_quality_count", 0) or 0) >= 1
    )


def run_targeted_second_search(
    sample: Dict[str, Any],
    *,
    default_pred: str,
    challenger_pred: str,
    search_provider: str,
    search_model: str,
    search_azure_suffix: Optional[str],
    timeout_s: int,
    max_retries: int,
    max_sources: int,
    search_context_size: str,
    cache_dir: Path,
    reuse_cache: bool,
    sleep_s: float,
    adjudicator_provider: str,
    adjudicator_model: str,
    adjudicator_azure_suffix: Optional[str],
    adjudicator_timeout_s: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_provider = _resolve_search_provider(search_provider)
    cache_payload = second_search_cache_payload(
        sample,
        search_provider=resolved_provider,
        search_model=search_model,
        search_context_size=search_context_size,
        max_sources=max_sources,
        default_pred=default_pred,
        challenger_pred=challenger_pred,
        adjudicator_provider=adjudicator_provider,
        adjudicator_model=adjudicator_model,
    )
    path = _cache_path(cache_dir, cache_payload)

    if reuse_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["second_search"], {
            "second_search_cache_hit": True,
            "second_search_cache_path": str(path),
            "second_search_provider": resolved_provider,
            "second_search_adjudicator_provider": adjudicator_provider,
            "second_search_adjudicator_model": adjudicator_model,
            "raw_second_search_response": cached.get("raw_second_search_response", ""),
            "raw_second_search_adjudication_response": cached.get("raw_second_search_adjudication_response", ""),
            "second_search_annotations": cached.get("annotations", []),
        }

    messages = build_targeted_search_messages(
        sample,
        default_pred=default_pred,
        challenger_pred=challenger_pred,
        max_sources=max_sources,
    )
    last_error = ""
    for attempt in range(max(1, max_retries + 1)):
        try:
            raw_response, annotations = _call_chat_search(
                messages=messages,
                provider=resolved_provider,
                model=search_model,
                timeout_s=timeout_s,
                search_context_size=search_context_size,
                azure_suffix=search_azure_suffix,
            )
            evidence_payload = normalize_second_search_evidence_payload(
                parse_json_object(raw_response),
                default_pred=default_pred,
                challenger_pred=challenger_pred,
                max_sources=max_sources,
                annotations=annotations,
            )
            raw_adjudication = call_chat_model(
                build_second_search_adjudication_messages(
                    sample,
                    default_pred=default_pred,
                    challenger_pred=challenger_pred,
                    evidence_payload=evidence_payload,
                    max_sources=max_sources,
                ),
                provider=adjudicator_provider,
                model=adjudicator_model,
                timeout_s=adjudicator_timeout_s,
                azure_suffix=adjudicator_azure_suffix,
            )
            payload = normalize_second_search_payload(
                parse_json_object(raw_adjudication),
                default_pred=default_pred,
                challenger_pred=challenger_pred,
                max_sources=max_sources,
                annotations=evidence_payload.get("sources") or annotations,
            )
            payload["search_evidence"] = evidence_payload
            payload["adjudicator_model"] = adjudicator_model
            cache_record = {
                "cache_payload": cache_payload,
                "second_search": payload,
                "raw_second_search_response": raw_response,
                "raw_second_search_adjudication_response": raw_adjudication,
                "annotations": annotations,
            }
            path.write_text(json.dumps(cache_record, indent=2, ensure_ascii=False), encoding="utf-8")
            return payload, {
                "second_search_cache_hit": False,
                "second_search_cache_path": str(path),
                "second_search_provider": resolved_provider,
                "second_search_adjudicator_provider": adjudicator_provider,
                "second_search_adjudicator_model": adjudicator_model,
                "raw_second_search_response": raw_response,
                "raw_second_search_adjudication_response": raw_adjudication,
                "second_search_annotations": annotations,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(max(1.0, sleep_s))

    raise RuntimeError(f"targeted second search failed: {last_error}")
