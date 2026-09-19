"""Prompts for CLEAR dynamic evidence search and answer generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from methods.common.prompt import format_options


SEARCH_PROMPT_VERSION = "clear_search_v1"
ANSWER_PROMPT_VERSION = "clear_answer_v1"


def build_search_messages(sample: Dict[str, Any], *, max_sources: int) -> List[Dict[str, str]]:
    """Build the GPT-4o-search evidence gathering prompt."""
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    user = (
        "Use web search to find reliable medical evidence for the following multiple-choice question.\n"
        "Focus on evidence that distinguishes the answer options. Prefer clinical guidelines, textbooks, "
        "review articles, official medical references, and authoritative medical organizations.\n"
        "Do not rely only on memory. Do not include irrelevant search results.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options ({labels}):\n{format_options(sample['options'])}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "evidence_summary": "short synthesis of the best evidence",\n'
        '  "key_facts": ["fact that helps choose among options", "..."],\n'
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
            "content": (
                "You are a medical research assistant. Search the web and produce concise, source-grounded evidence. "
                "Return valid JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]


def format_evidence(evidence: Dict[str, Any]) -> str:
    """Render evidence compactly for the answer model."""
    rows: List[str] = []
    summary = str(evidence.get("evidence_summary", "")).strip()
    if summary:
        rows.append(f"Evidence summary:\n{summary}")

    facts = [str(item).strip() for item in evidence.get("key_facts", []) if str(item).strip()]
    if facts:
        rows.append("Key facts:\n" + "\n".join(f"- {item}" for item in facts))

    sources = evidence.get("sources", [])
    if isinstance(sources, list) and sources:
        rendered_sources: List[str] = []
        for idx, item in enumerate(sources, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip() or "(untitled)"
            claim = str(item.get("claim", "")).strip()
            relevance = str(item.get("relevance", "")).strip()
            url = str(item.get("url", "")).strip()
            body = f"[{idx}] {title}"
            if claim:
                body += f"\nClaim: {claim}"
            if relevance:
                body += f"\nRelevance: {relevance}"
            if url:
                body += f"\nURL: {url}"
            rendered_sources.append(body)
        if rendered_sources:
            rows.append("Sources:\n" + "\n\n".join(rendered_sources))

    if not rows:
        return "No usable online evidence was found."
    return "\n\n".join(rows)


def build_answer_messages(sample: Dict[str, Any], evidence: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build final answer prompt for the target answer model."""
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    user = (
        "Answer the following medical multiple-choice question using the online evidence when it is relevant.\n"
        "If the evidence is incomplete or conflicts with the question, use careful medical reasoning.\n"
        f"Return only one option letter from: {labels}.\n\n"
        f"Online evidence:\n{format_evidence(evidence)}\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        "Final answer:"
    )
    return [
        {
            "role": "system",
            "content": "You are a careful medical QA assistant. Choose exactly one option.",
        },
        {"role": "user", "content": user},
    ]


def evidence_cache_payload(
    sample: Dict[str, Any],
    *,
    search_provider: str,
    search_model: str,
    search_context_size: str,
    max_sources: int,
) -> Dict[str, Any]:
    """Return the stable payload used to key the evidence cache."""
    return {
        "prompt_version": SEARCH_PROMPT_VERSION,
        "search_provider": search_provider,
        "search_model": search_model,
        "search_context_size": search_context_size,
        "max_sources": max_sources,
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "id": sample.get("id"),
        "question": sample.get("question"),
        "options": sample.get("options"),
    }


def dumps_for_cache(payload: Dict[str, Any]) -> str:
    """Canonical JSON serialization for cache hashing."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
