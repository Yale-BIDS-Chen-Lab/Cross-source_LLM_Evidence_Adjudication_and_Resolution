"""Dynamic evidence search for CLEAR."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from methods.clear.prompts import build_search_messages, dumps_for_cache, evidence_cache_payload
from methods.clear.structured import normalize_evidence, parse_json_object


def _extract_annotations(message: Any) -> List[Dict[str, Any]]:
    annotations = getattr(message, "annotations", None) or []
    rows: List[Dict[str, Any]] = []
    for item in annotations:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        elif hasattr(item, "to_dict"):
            rows.append(item.to_dict())
        elif isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"value": str(item)})
    return rows


def _cache_path(cache_dir: Path, payload: Dict[str, Any]) -> Path:
    key = hashlib.sha256(dumps_for_cache(payload).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _env_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _resolve_search_provider(provider: str) -> str:
    key = str(provider or "auto").strip().lower()
    if key in {"", "auto"}:
        if _env_value("AZURE_ENDPOINT") and _env_value("AZURE_API_KEY"):
            return "azure"
        if _env_value("OPENAI_API_KEY"):
            return "openai"
        if _env_value("TAVILY_API_KEY"):
            return "tavily"
        return "openai"
    if key in {"azure", "azure_openai", "openai_azure"}:
        return "azure"
    if key == "openai":
        return "openai"
    if key == "tavily":
        return "tavily"
    raise ValueError(f"Unsupported search provider: {provider}")


def _azure_search_client(*, azure_suffix: Optional[str], timeout_s: int):
    from openai import AzureOpenAI

    suffix = "" if azure_suffix is None else azure_suffix
    endpoint = _env_value(f"AZURE_ENDPOINT{suffix}")
    api_key = _env_value(f"AZURE_API_KEY{suffix}")
    api_version = _env_value("AZURE_WEBSEARCH_API_VERSION") or "2025-03-01-preview"
    if not endpoint or not api_key or not api_version:
        raise RuntimeError(
            "Missing Azure search environment variables. Source Keys/env.sh first, "
            "or pass --search-provider openai with OPENAI_API_KEY."
        )
    return AzureOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        api_version=api_version,
        timeout=timeout_s,
    )


def _messages_to_responses_input(messages: List[Dict[str, str]]) -> str:
    rows: List[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip().upper() or "USER"
        content = str(message.get("content", "")).strip()
        if content:
            rows.append(f"{role}:\n{content}")
    return "\n\n".join(rows).strip()


def _extract_response_sources(response: Any) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    output_text = str(getattr(response, "output_text", "") or "").strip()
    sources: List[Dict[str, Any]] = []
    search_queries: List[str] = []

    for item in getattr(response, "output", []) or []:
        item_type = str(getattr(item, "type", "") or "").strip()
        if item_type == "web_search_call":
            action = getattr(item, "action", None)
            query = str(getattr(action, "query", "") or "").strip() if action else ""
            if query:
                search_queries.append(query)
            for source in getattr(action, "sources", None) or []:
                row = {
                    "title": str(getattr(source, "title", "") or "").strip(),
                    "url": str(getattr(source, "url", "") or "").strip(),
                    "snippet": str(getattr(source, "snippet", "") or "").strip(),
                    "site_name": str(getattr(source, "site_name", "") or "").strip(),
                    "published_at": str(getattr(source, "published_at", "") or "").strip(),
                }
                if any(row.values()):
                    sources.append(row)
        elif item_type == "message" and not output_text:
            parts: List[str] = []
            for content in getattr(item, "content", None) or []:
                if str(getattr(content, "type", "") or "").strip() == "output_text":
                    text = str(getattr(content, "text", "") or "").strip()
                    if text:
                        parts.append(text)
            if parts:
                output_text = "\n".join(parts).strip()

    return output_text, sources, search_queries


def _call_azure_responses_web_search(
    *,
    messages: List[Dict[str, str]],
    model: str,
    timeout_s: int,
    azure_suffix: Optional[str],
) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    client = _azure_search_client(azure_suffix=azure_suffix, timeout_s=timeout_s)
    response = client.responses.create(
        model=model,
        input=_messages_to_responses_input(messages),
        tools=[{"type": "web_search"}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        timeout=max(1, int(timeout_s)),
    )
    output_text, sources, search_queries = _extract_response_sources(response)
    if not output_text:
        raise RuntimeError("Azure Responses web_search returned empty output_text")
    return output_text, sources, search_queries


def _call_chat_search(
    *,
    messages: List[Dict[str, str]],
    provider: str,
    model: str,
    timeout_s: int,
    search_context_size: str,
    azure_suffix: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    resolved_provider = _resolve_search_provider(provider)
    if resolved_provider == "azure":
        raw_response, sources, _search_queries = _call_azure_responses_web_search(
            messages=messages,
            model=model,
            timeout_s=timeout_s,
            azure_suffix=azure_suffix,
        )
        return raw_response, sources
    else:
        from openai import OpenAI

        client = OpenAI(timeout=timeout_s)

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if search_context_size:
        kwargs["web_search_options"] = {"search_context_size": search_context_size}

    try:
        response = client.chat.completions.create(**kwargs)
    except TypeError:
        web_options = kwargs.pop("web_search_options", None)
        if web_options:
            kwargs["extra_body"] = {"web_search_options": web_options}
        response = client.chat.completions.create(**kwargs)

    if not response.choices:
        raise RuntimeError("search model returned no choices")
    message = response.choices[0].message
    content = message.content or ""
    if not content:
        raise RuntimeError("search model returned empty content")
    return str(content), _extract_annotations(message)


def _build_tavily_query(sample: Dict[str, Any]) -> str:
    options = sample.get("options", {})
    option_text = " ".join(f"{label}: {text}" for label, text in sorted(options.items()))
    question = str(sample.get("question", "")).strip()
    query = f"medical evidence MCQ question: {question} options: {option_text}"
    if len(query) <= 380:
        return query
    option_budget = 110
    short_options = option_text[:option_budget].rsplit(" ", 1)[0] or option_text[:option_budget]
    question_budget = max(80, 380 - len("medical evidence MCQ question:  options: ") - len(short_options))
    short_question = question[:question_budget].rsplit(" ", 1)[0] or question[:question_budget]
    return f"medical evidence MCQ question: {short_question} options: {short_options}"[:380]


def _call_tavily_search(
    *,
    sample: Dict[str, Any],
    timeout_s: int,
    search_context_size: str,
    max_sources: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    api_key = _env_value("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY for Tavily search")

    search_depth = "basic" if search_context_size == "low" else "advanced"
    body = {
        "query": _build_tavily_query(sample),
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
        "max_results": int(max_sources),
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily search failed: HTTP {exc.code}: {body_text}") from exc

    payload_obj = json.loads(raw)
    results = payload_obj.get("results", [])
    if not isinstance(results, list):
        results = []

    sources: List[Dict[str, str]] = []
    facts: List[str] = []
    for item in results[:max_sources]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        score = item.get("score")
        if content:
            facts.append(content)
        relevance = "Tavily search result"
        if score is not None:
            relevance = f"Tavily relevance score: {score}"
        sources.append(
            {
                "title": title,
                "url": url,
                "claim": content,
                "relevance": relevance,
            }
        )

    evidence = {
        "evidence_summary": str(payload_obj.get("answer", "")).strip(),
        "key_facts": facts,
        "sources": sources,
        "search_confidence": "medium" if sources else "low",
    }
    trace = {
        "search_provider": "tavily",
        "search_cache_hit": False,
        "search_annotations": [],
        "raw_search_response": raw,
        "tavily_search_depth": search_depth,
    }
    return evidence, trace


def search_evidence(
    sample: Dict[str, Any],
    *,
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
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Search online evidence and return normalized evidence plus trace metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_provider = _resolve_search_provider(search_provider)
    cache_payload = evidence_cache_payload(
        sample,
        search_provider=resolved_provider,
        search_model=search_model,
        search_context_size=search_context_size,
        max_sources=max_sources,
    )
    path = _cache_path(cache_dir, cache_payload)

    if reuse_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["evidence"], {
            "search_cache_hit": True,
            "search_cache_path": str(path),
            "search_provider": resolved_provider,
            "search_annotations": cached.get("annotations", []),
            "raw_search_response": cached.get("raw_search_response", ""),
        }

    messages = build_search_messages(sample, max_sources=max_sources)
    last_error = ""
    for attempt in range(max(1, max_retries + 1)):
        try:
            if resolved_provider == "tavily":
                evidence, tavily_trace = _call_tavily_search(
                    sample=sample,
                    timeout_s=timeout_s,
                    search_context_size=search_context_size,
                    max_sources=max_sources,
                )
                raw_response = tavily_trace["raw_search_response"]
                annotations = tavily_trace["search_annotations"]
                extra_trace = tavily_trace
            else:
                raw_response, annotations = _call_chat_search(
                    messages=messages,
                    provider=resolved_provider,
                    model=search_model,
                    timeout_s=timeout_s,
                    search_context_size=search_context_size,
                    azure_suffix=search_azure_suffix,
                )
                payload = parse_json_object(raw_response)
                evidence = normalize_evidence(payload, max_sources=max_sources)
                if resolved_provider == "azure" and not evidence.get("sources") and annotations:
                    evidence["sources"] = [
                        {
                            "title": str(item.get("title", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "claim": str(item.get("snippet", "")).strip(),
                            "relevance": str(item.get("site_name", "")).strip(),
                        }
                        for item in annotations[:max_sources]
                        if isinstance(item, dict)
                    ]
                extra_trace = {}
            cache_record = {
                "cache_payload": cache_payload,
                "evidence": evidence,
                "raw_search_response": raw_response,
                "annotations": annotations,
            }
            path.write_text(json.dumps(cache_record, indent=2, ensure_ascii=False), encoding="utf-8")
            trace = {
                "search_cache_hit": False,
                "search_cache_path": str(path),
                "search_provider": resolved_provider,
                "search_annotations": annotations,
                "raw_search_response": raw_response,
            }
            trace.update(extra_trace)
            trace["search_cache_path"] = str(path)
            return evidence, trace
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(max(1.0, sleep_s))

    raise RuntimeError(f"online search failed: {last_error}")
