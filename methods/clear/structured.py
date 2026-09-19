"""Structured evidence helpers for the deep research method."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def parse_json_object(text: str) -> Dict[str, Any]:
    """Extract one JSON object from model text."""
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```json\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def normalize_sources(raw_sources: Any, *, max_sources: int) -> List[Dict[str, str]]:
    """Normalize evidence source rows to a stable schema."""
    rows: List[Dict[str, str]] = []
    if not isinstance(raw_sources, list):
        return rows

    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        claim = str(item.get("claim", "")).strip()
        relevance = str(item.get("relevance", "")).strip()
        if not any([title, url, claim, relevance]):
            continue
        rows.append(
            {
                "title": title,
                "url": url,
                "claim": claim,
                "relevance": relevance,
            }
        )
        if len(rows) >= max_sources:
            break
    return rows


def normalize_evidence(payload: Dict[str, Any], *, max_sources: int) -> Dict[str, Any]:
    """Normalize search output into the evidence object stored in traces."""
    key_facts = payload.get("key_facts", [])
    if not isinstance(key_facts, list):
        key_facts = []
    facts = [str(item).strip() for item in key_facts if str(item).strip()]

    confidence = str(payload.get("search_confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "evidence_summary": str(payload.get("evidence_summary", "")).strip(),
        "key_facts": facts,
        "sources": normalize_sources(payload.get("sources", []), max_sources=max_sources),
        "search_confidence": confidence,
    }


def evidence_has_content(evidence: Dict[str, Any]) -> bool:
    """Return True if the evidence has enough content for answer synthesis."""
    return bool(
        str(evidence.get("evidence_summary", "")).strip()
        or evidence.get("key_facts")
        or evidence.get("sources")
    )

