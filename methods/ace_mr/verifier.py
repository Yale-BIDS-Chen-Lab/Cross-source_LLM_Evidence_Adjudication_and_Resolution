"""Reviewer/verifier module for CLEAR."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from methods.common.llm import call_chat_model
from methods.common.prompt import format_options, format_retrieved_docs
from methods.common.parser import parse_answer
from methods.ace_mr.prompts import format_evidence
from methods.ace_mr.source_quality import format_source_quality_report, summarize_source_quality
from methods.ace_mr.structured import parse_json_object
from methods.ace_mr.targeted_search import allow_second_search_override, run_targeted_second_search


VALID_DECISIONS = {
    "consensus",
    "keep_direct",
    "keep_bm25",
    "accept_online",
    "needs_more_search",
}


def _clean_pred(value: Any, labels: set[str]) -> Optional[str]:
    pred = str(value or "").strip().upper()
    return pred if pred in labels else None


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _compact_candidate(name: str, candidate: Dict[str, Any]) -> str:
    pred = str(candidate.get("pred") or "").strip().upper() or "None"
    raw = _truncate(candidate.get("raw_response", ""), 800)
    error = _truncate((candidate.get("trace") or {}).get("error", ""), 500)
    rows = [f"{name} pred: {pred}"]
    if raw:
        rows.append(f"{name} raw_response: {raw}")
    if error:
        rows.append(f"{name} error: {error}")
    return "\n".join(rows)


def _compact_local_docs(snippets: Sequence[Dict[str, Any]], *, max_docs: int) -> str:
    docs: List[Dict[str, Any]] = []
    for item in list(snippets)[:max_docs]:
        row = dict(item)
        row["content"] = _truncate(row.get("content") or row.get("contents"), 1000)
        docs.append(row)
    return format_retrieved_docs(docs) if docs else "No local documents available."


def build_verifier_messages(
    sample: Dict[str, Any],
    *,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    online_evidence: Dict[str, Any],
    source_quality_report: Dict[str, Any],
    local_docs: Sequence[Dict[str, Any]],
    max_local_docs: int = 6,
) -> List[Dict[str, str]]:
    """Build a conservative verifier prompt over direct/local/online candidates."""
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    local_docs_text = _compact_local_docs(local_docs, max_docs=max_local_docs)
    user = (
        "Review three candidate answers for this medical multiple-choice question.\n\n"
        "DECISION POLICY:\n"
        "1. Source quality is not answer support. A reliable source can still be irrelevant to the exact option.\n"
        "2. Prefer consensus. If Direct and BM25 agree, preserve that answer by default. Online may override "
        "only if sources explicitly refute that consensus option and explicitly support the online option for "
        "the exact clinical scenario.\n"
        "3. Online evidence must be option-level evidence: it must distinguish the answer choices, not merely "
        "state a true background fact about the disease, drug, guideline, or management topic.\n"
        "4. Check question alignment. Reject online evidence that answers a related but different task "
        "(e.g., treatment instead of diagnosis, general guideline instead of the available options, or a "
        "different disease implied by ignoring salient clues).\n"
        "5. If online summary conflicts with its source claims, trust the source claims and downgrade online.\n"
        "6. If Direct, BM25, and Online all disagree, do not pick Online unless its exact option-level support "
        "is substantially stronger than both alternatives. Otherwise preserve Direct.\n"
        "7. If evidence is weak, tangential, low-quality, or contradictory, preserve BM25 when available; "
        "otherwise preserve Direct.\n"
        "8. If no candidate is sufficiently supported, choose the best available option and mark "
        "decision as needs_more_search.\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        "CANDIDATES:\n"
        f"{_compact_candidate('Direct', direct_candidate)}\n\n"
        f"{_compact_candidate('BM25', bm25_candidate)}\n\n"
        f"{_compact_candidate('Online', online_candidate)}\n\n"
        f"LOCAL BM25 EVIDENCE:\n{local_docs_text}\n\n"
        f"ONLINE EVIDENCE:\n{format_evidence(online_evidence)}\n\n"
        f"ONLINE SOURCE QUALITY:\n{format_source_quality_report(source_quality_report)}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        f'  "final_pred": "<one letter from: {labels}>",\n'
        '  "decision": "consensus|keep_direct|keep_bm25|accept_online|needs_more_search",\n'
        '  "confidence": "high|medium|low",\n'
        '  "online_supported": true,\n'
        '  "local_supported": true,\n'
        '  "conflict_detected": false,\n'
        '  "option_level_support": true,\n'
        '  "consensus_refuted": false,\n'
        '  "question_aligned": true,\n'
        '  "support_scope": "exact_option|background_only|related_task|unclear",\n'
        '  "reason": "one concise sentence",\n'
        '  "accepted_sources": ["short source ids or domains"],\n'
        '  "rejected_sources": ["short source ids or domains"]\n'
        "}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative medical QA reviewer. Your job is to prevent weak or noisy "
                "retrieval evidence from overriding a better-supported answer."
            ),
        },
        {"role": "user", "content": user},
    ]


def normalize_verdict(payload: Dict[str, Any], *, labels: set[str]) -> Dict[str, Any]:
    pred = _clean_pred(payload.get("final_pred"), labels)
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        decision = "needs_more_search"
    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "final_pred": pred,
        "decision": decision,
        "confidence": confidence,
        "online_supported": bool(payload.get("online_supported", False)),
        "local_supported": bool(payload.get("local_supported", False)),
        "conflict_detected": bool(payload.get("conflict_detected", False)),
        "option_level_support": bool(payload.get("option_level_support", False)),
        "consensus_refuted": bool(payload.get("consensus_refuted", False)),
        "question_aligned": bool(payload.get("question_aligned", False)),
        "support_scope": str(payload.get("support_scope", "")).strip().lower(),
        "reason": str(payload.get("reason", "")).strip(),
        "accepted_sources": payload.get("accepted_sources", []) if isinstance(payload.get("accepted_sources"), list) else [],
        "rejected_sources": payload.get("rejected_sources", []) if isinstance(payload.get("rejected_sources"), list) else [],
    }


def heuristic_verdict(
    sample: Dict[str, Any],
    *,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    source_quality_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Conservative deterministic fallback when LLM verification is unavailable."""
    labels = {str(k).strip().upper() for k in sample["options"]}
    direct = _clean_pred(direct_candidate.get("pred"), labels)
    bm25 = _clean_pred(bm25_candidate.get("pred"), labels)
    online = _clean_pred(online_candidate.get("pred"), labels)
    trusted_online = bool(source_quality_report.get("trusted_source_present"))
    max_quality = float(source_quality_report.get("max_quality_score", 0.0) or 0.0)

    if bm25 and direct and bm25 == direct:
        return {
            "final_pred": bm25,
            "decision": "consensus",
            "confidence": "high",
            "online_supported": bool(online == bm25),
            "local_supported": True,
            "conflict_detected": bool(online and online != bm25),
            "reason": "Direct and BM25 agree, so the verifier preserves the local consensus.",
            "accepted_sources": [],
            "rejected_sources": [],
        }
    if bm25 and online and bm25 == online:
        return {
            "final_pred": bm25,
            "decision": "consensus",
            "confidence": "high" if trusted_online else "medium",
            "online_supported": True,
            "local_supported": True,
            "conflict_detected": bool(direct and direct != bm25),
            "reason": "BM25 and online evidence agree, so the verifier accepts the shared answer.",
            "accepted_sources": [],
            "rejected_sources": [],
        }
    if direct and online and direct == online and trusted_online:
        return {
            "final_pred": direct,
            "decision": "consensus",
            "confidence": "medium",
            "online_supported": True,
            "local_supported": bool(bm25 == direct),
            "conflict_detected": bool(bm25 and bm25 != direct),
            "reason": "Direct and high-quality online evidence agree.",
            "accepted_sources": [],
            "rejected_sources": [],
        }
    if bm25:
        return {
            "final_pred": bm25,
            "decision": "keep_bm25",
            "confidence": "medium" if max_quality >= 0.5 else "low",
            "online_supported": bool(online == bm25),
            "local_supported": True,
            "conflict_detected": bool(online and online != bm25),
            "reason": "Online evidence did not clearly justify overriding the BM25 answer.",
            "accepted_sources": [],
            "rejected_sources": [],
        }
    if direct:
        return {
            "final_pred": direct,
            "decision": "keep_direct",
            "confidence": "medium",
            "online_supported": bool(online == direct),
            "local_supported": False,
            "conflict_detected": bool(online and online != direct),
            "reason": "No BM25 answer is available, so the verifier preserves direct unless online is clearly stronger.",
            "accepted_sources": [],
            "rejected_sources": [],
        }
    return {
        "final_pred": online,
        "decision": "accept_online" if online else "needs_more_search",
        "confidence": "medium" if trusted_online else "low",
        "online_supported": bool(online),
        "local_supported": False,
        "conflict_detected": False,
        "reason": "Only the online candidate produced a usable answer.",
        "accepted_sources": [],
        "rejected_sources": [],
    }


def build_override_audit_messages(
    sample: Dict[str, Any],
    *,
    guard_type: str,
    default_pred: str,
    online_pred: str,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    online_evidence: Dict[str, Any],
    source_quality_report: Dict[str, Any],
    local_docs: Sequence[Dict[str, Any]],
    max_local_docs: int = 6,
) -> List[Dict[str, str]]:
    labels = ", ".join(sorted(str(k).strip().upper() for k in sample["options"]))
    default_text = sample["options"].get(default_pred, "")
    online_text = sample["options"].get(online_pred, "")
    local_docs_text = _compact_local_docs(local_docs, max_docs=max_local_docs)
    user = (
        "Audit whether an online answer is allowed to override the safer default answer.\n\n"
        "DEFAULT POLICY: ALLOW_OVERRIDE must be false unless all of these are true:\n"
        "1. The online source claims explicitly support the online option for the exact question.\n"
        "2. The online source claims explicitly refute or make inapplicable the default option.\n"
        "3. The online evidence accounts for the salient clues and constraints in the question.\n"
        "4. The online evidence answers the same task as the question, not a related task.\n"
        "5. The support is present in source claims, not only in the model's evidence summary.\n\n"
        "Important: source quality only measures trust in the source. It does not prove that the source "
        "supports the answer option. If the source is high quality but the option mapping is indirect, "
        "ALLOW_OVERRIDE must be false.\n\n"
        f"Guard type: {guard_type}\n"
        f"Default answer to preserve unless clearly wrong: {default_pred}. {default_text}\n"
        f"Online answer requesting override: {online_pred}. {online_text}\n\n"
        f"Question:\n{sample['question']}\n\n"
        f"Options:\n{format_options(sample['options'])}\n\n"
        "CANDIDATES:\n"
        f"{_compact_candidate('Direct', direct_candidate)}\n\n"
        f"{_compact_candidate('BM25', bm25_candidate)}\n\n"
        f"{_compact_candidate('Online', online_candidate)}\n\n"
        f"LOCAL BM25 EVIDENCE:\n{local_docs_text}\n\n"
        f"ONLINE EVIDENCE:\n{format_evidence(online_evidence)}\n\n"
        f"ONLINE SOURCE QUALITY:\n{format_source_quality_report(source_quality_report)}\n\n"
        "Return JSON only with this exact shape:\n"
        "{\n"
        '  "allow_override": false,\n'
        f'  "final_pred": "<one letter from: {labels}>",\n'
        '  "confidence": "high|medium|low",\n'
        '  "option_level_support": false,\n'
        '  "refutes_default": false,\n'
        '  "question_aligned": false,\n'
        '  "support_scope": "exact_option|background_only|related_task|unclear",\n'
        '  "reason": "one concise sentence"\n'
        "}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an adversarial medical QA audit reviewer. Be skeptical of online overrides. "
                "Reject overrides unless the cited evidence is exact, option-level, and refutes the safer default."
            ),
        },
        {"role": "user", "content": user},
    ]


def _normalize_override_audit(payload: Dict[str, Any], *, labels: set[str], online_pred: str) -> Dict[str, Any]:
    pred = _clean_pred(payload.get("final_pred"), labels)
    confidence = str(payload.get("confidence", "low")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    option_level = bool(payload.get("option_level_support", False))
    refutes_default = bool(payload.get("refutes_default", False))
    question_aligned = bool(payload.get("question_aligned", False))
    support_scope = str(payload.get("support_scope", "")).strip().lower()
    allow = (
        bool(payload.get("allow_override", False))
        and pred == online_pred
        and option_level
        and refutes_default
        and question_aligned
        and support_scope == "exact_option"
    )
    return {
        "allow_override": allow,
        "raw_allow_override": bool(payload.get("allow_override", False)),
        "final_pred": pred,
        "confidence": confidence,
        "option_level_support": option_level,
        "refutes_default": refutes_default,
        "question_aligned": question_aligned,
        "support_scope": support_scope,
        "reason": str(payload.get("reason", "")).strip(),
    }


def _run_override_audit(
    sample: Dict[str, Any],
    *,
    guard_type: str,
    default_pred: str,
    online_pred: str,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    online_evidence: Dict[str, Any],
    source_quality_report: Dict[str, Any],
    local_docs: Sequence[Dict[str, Any]],
    provider: str,
    model: str,
    timeout_s: int,
    azure_suffix: Optional[str],
    max_local_docs: int,
) -> Dict[str, Any]:
    labels = {str(k).strip().upper() for k in sample["options"]}
    raw_response = ""
    try:
        raw_response = call_chat_model(
            build_override_audit_messages(
                sample,
                guard_type=guard_type,
                default_pred=default_pred,
                online_pred=online_pred,
                direct_candidate=direct_candidate,
                bm25_candidate=bm25_candidate,
                online_candidate=online_candidate,
                online_evidence=online_evidence,
                source_quality_report=source_quality_report,
                local_docs=local_docs,
                max_local_docs=max_local_docs,
            ),
            provider=provider,
            model=model,
            timeout_s=timeout_s,
            azure_suffix=azure_suffix,
        )
        audit = _normalize_override_audit(parse_json_object(raw_response), labels=labels, online_pred=online_pred)
        audit["raw_response"] = raw_response
        return audit
    except Exception as exc:
        return {
            "allow_override": False,
            "raw_allow_override": False,
            "final_pred": default_pred,
            "confidence": "low",
            "option_level_support": False,
            "refutes_default": False,
            "question_aligned": False,
            "support_scope": "unclear",
            "reason": "Override audit failed; preserving the safer default.",
            "raw_response": raw_response,
            "error": str(exc),
        }


def _apply_post_verifier_guards(
    verdict: Dict[str, Any],
    sample: Dict[str, Any],
    *,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    online_evidence: Dict[str, Any],
    source_quality_report: Dict[str, Any],
    local_docs: Sequence[Dict[str, Any]],
    provider: str,
    model: str,
    timeout_s: int,
    azure_suffix: Optional[str],
    max_local_docs: int,
    enable_second_search: bool,
    second_search_provider: str,
    second_search_model: str,
    second_search_azure_suffix: Optional[str],
    second_search_timeout_s: int,
    second_search_max_retries: int,
    second_search_max_sources: int,
    second_search_context_size: str,
    second_search_cache_dir: Optional[Path],
    reuse_second_search_cache: bool,
    second_search_sleep_s: float,
    second_search_quality_threshold: float,
) -> Dict[str, Any]:
    labels = {str(k).strip().upper() for k in sample["options"]}
    direct = _clean_pred(direct_candidate.get("pred"), labels)
    bm25 = _clean_pred(bm25_candidate.get("pred"), labels)
    online = _clean_pred(online_candidate.get("pred"), labels)
    final = _clean_pred(verdict.get("final_pred"), labels)

    if not online or final != online or str(verdict.get("decision")) != "accept_online":
        return verdict

    guard_type = ""
    default_pred: Optional[str] = None
    if direct and bm25 and direct == bm25 and online != direct:
        guard_type = "direct_bm25_consensus"
        default_pred = bm25
    elif direct and bm25 and online and len({direct, bm25, online}) == 3:
        guard_type = "three_way_conflict"
        default_pred = direct

    if not guard_type or not default_pred:
        return verdict

    if enable_second_search:
        max_quality = float(source_quality_report.get("max_quality_score", 0.0) or 0.0)
        if max_quality >= float(second_search_quality_threshold):
            try:
                second_payload, second_trace = run_targeted_second_search(
                    sample,
                    default_pred=default_pred,
                    challenger_pred=online,
                    search_provider=second_search_provider,
                    search_model=second_search_model,
                    search_azure_suffix=second_search_azure_suffix,
                    timeout_s=second_search_timeout_s,
                    max_retries=second_search_max_retries,
                    max_sources=second_search_max_sources,
                    search_context_size=second_search_context_size,
                    cache_dir=second_search_cache_dir
                    or Path("temp/ace_mr_second_search_cache").resolve(),
                    reuse_cache=reuse_second_search_cache,
                    sleep_s=second_search_sleep_s,
                    adjudicator_provider=provider,
                    adjudicator_model=model,
                    adjudicator_azure_suffix=azure_suffix,
                    adjudicator_timeout_s=timeout_s,
                )
                allow_override = allow_second_search_override(second_payload)
                second_payload = dict(second_payload)
                second_payload["allow_override"] = allow_override
                guarded = dict(verdict)
                guarded["targeted_second_search"] = second_payload
                guarded["targeted_second_search_trace"] = second_trace
                if allow_override:
                    guarded.update(
                        {
                            "final_pred": online,
                            "decision": "accept_online",
                            "confidence": "high",
                            "guard_applied": False,
                            "guard_type": guard_type,
                            "guard_reason": "Targeted second search allowed the online override.",
                        }
                    )
                    return guarded
                guarded.update(
                    {
                        "final_pred": default_pred,
                        "decision": "keep_bm25" if guard_type == "direct_bm25_consensus" else "keep_direct",
                        "confidence": "high" if guard_type == "direct_bm25_consensus" else "medium",
                        "guard_applied": True,
                        "guard_type": guard_type,
                        "guard_reason": second_payload.get("reason")
                        or "Targeted second search did not satisfy override criteria.",
                    }
                )
                return guarded
            except Exception as exc:
                guarded = dict(verdict)
                guarded.update(
                    {
                        "final_pred": default_pred,
                        "decision": "keep_bm25" if guard_type == "direct_bm25_consensus" else "keep_direct",
                        "confidence": "high" if guard_type == "direct_bm25_consensus" else "medium",
                        "guard_applied": True,
                        "guard_type": guard_type,
                        "guard_reason": "Targeted second search failed; preserving the safer default.",
                        "targeted_second_search_error": str(exc),
                    }
                )
                return guarded

    allow_audited_override = str(os.getenv("DEEPRESEARCH_ALLOW_AUDITED_CONSENSUS_OVERRIDE", "")).strip() == "1"
    if not allow_audited_override:
        guarded = dict(verdict)
        guarded.update(
            {
                "final_pred": default_pred,
                "decision": "keep_bm25" if guard_type == "direct_bm25_consensus" else "keep_direct",
                "confidence": "high" if guard_type == "direct_bm25_consensus" else "medium",
                "guard_applied": True,
                "guard_type": guard_type,
                "guard_reason": (
                    "Strict guard preserved the Direct+BM25 consensus over a conflicting online answer."
                    if guard_type == "direct_bm25_consensus"
                    else "Strict guard preserved Direct in a three-way candidate conflict."
                ),
            }
        )
        return guarded

    audit = _run_override_audit(
        sample,
        guard_type=guard_type,
        default_pred=default_pred,
        online_pred=online,
        direct_candidate=direct_candidate,
        bm25_candidate=bm25_candidate,
        online_candidate=online_candidate,
        online_evidence=online_evidence,
        source_quality_report=source_quality_report,
        local_docs=local_docs,
        provider=provider,
        model=model,
        timeout_s=timeout_s,
        azure_suffix=azure_suffix,
        max_local_docs=max_local_docs,
    )

    guarded = dict(verdict)
    guarded["override_audit"] = audit
    if audit.get("allow_override"):
        guarded["guard_applied"] = False
        guarded["guard_reason"] = "Override audit allowed the online answer."
        return guarded

    guarded.update(
        {
            "final_pred": default_pred,
            "decision": "keep_bm25" if guard_type == "direct_bm25_consensus" else "keep_direct",
            "confidence": "high" if guard_type == "direct_bm25_consensus" else "medium",
            "guard_applied": True,
            "guard_type": guard_type,
            "guard_reason": audit.get("reason") or "Online override failed conservative audit.",
        }
    )
    return guarded


def _allow_online_challenge_override(payload: Dict[str, Any]) -> bool:
    """Balanced acceptance rule for online challenges to a local consensus.

    This is stricter than a normal verifier choice but slightly
    less brittle than the high-quality-only second-search guard: a challenge can
    pass with either one high-authority source or at least two medium-authority
    sources, as long as the web adjudication is exact, aligned, and not
    unresolved.
    """
    source_quality = payload.get("source_quality") or {}
    high_count = int(source_quality.get("high_quality_count", 0) or 0)
    medium_count = int(source_quality.get("medium_quality_count", 0) or 0)
    quality_ok = high_count >= 1 or medium_count >= 2
    return bool(
        payload.get("recommend_override")
        and payload.get("supports_challenger")
        and payload.get("default_not_best")
        and payload.get("question_aligned")
        and not payload.get("unresolved_conflict")
        and payload.get("support_scope") == "exact_option"
        and quality_ok
    )


def _apply_online_challenge_audit(
    verdict: Dict[str, Any],
    sample: Dict[str, Any],
    *,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    enable_online_challenge_audit: bool,
    second_search_provider: str,
    second_search_model: str,
    second_search_azure_suffix: Optional[str],
    second_search_timeout_s: int,
    second_search_max_retries: int,
    second_search_max_sources: int,
    second_search_context_size: str,
    second_search_cache_dir: Optional[Path],
    reuse_second_search_cache: bool,
    second_search_sleep_s: float,
    adjudicator_provider: str,
    adjudicator_model: str,
    adjudicator_azure_suffix: Optional[str],
    adjudicator_timeout_s: int,
) -> Dict[str, Any]:
    if not enable_online_challenge_audit:
        return verdict

    labels = {str(k).strip().upper() for k in sample["options"]}
    direct = _clean_pred(direct_candidate.get("pred"), labels)
    bm25 = _clean_pred(bm25_candidate.get("pred"), labels)
    online = _clean_pred(online_candidate.get("pred"), labels)
    final = _clean_pred(verdict.get("final_pred"), labels)

    if not (direct and bm25 and online and direct == bm25 and online != direct):
        return verdict

    default_pred = direct
    try:
        challenge_payload, challenge_trace = run_targeted_second_search(
            sample,
            default_pred=default_pred,
            challenger_pred=online,
            search_provider=second_search_provider,
            search_model=second_search_model,
            search_azure_suffix=second_search_azure_suffix,
            timeout_s=second_search_timeout_s,
            max_retries=second_search_max_retries,
            max_sources=second_search_max_sources,
            search_context_size=second_search_context_size,
            cache_dir=second_search_cache_dir or Path("temp/ace_mr_second_search_cache").resolve(),
            reuse_cache=reuse_second_search_cache,
            sleep_s=second_search_sleep_s,
            adjudicator_provider=adjudicator_provider,
            adjudicator_model=adjudicator_model,
            adjudicator_azure_suffix=adjudicator_azure_suffix,
            adjudicator_timeout_s=adjudicator_timeout_s,
        )
        allow_override = _allow_online_challenge_override(challenge_payload)
        challenge_payload = dict(challenge_payload)
        challenge_payload["allow_override"] = allow_override
        challenge_payload["audit_type"] = "online_challenges_direct_bm25_consensus"

        audited = dict(verdict)
        audited["online_challenge_audit"] = challenge_payload
        audited["online_challenge_audit_trace"] = challenge_trace
        if allow_override:
            audited.update(
                {
                    "final_pred": online,
                    "decision": "accept_online",
                    "confidence": "high",
                    "online_challenge_applied": True,
                    "online_challenge_reason": challenge_payload.get("reason")
                    or "Online challenge audit allowed overriding Direct+BM25 consensus.",
                    "guard_applied": False,
                    "guard_type": "online_challenge_audit",
                    "guard_reason": "Online challenge audit overrode Direct+BM25 consensus.",
                }
            )
            return audited

        audited["online_challenge_applied"] = False
        audited["online_challenge_reason"] = (
            challenge_payload.get("reason") or "Online challenge audit did not satisfy override criteria."
        )
        if final == online:
            audited.update(
                {
                    "final_pred": default_pred,
                    "decision": "keep_bm25",
                    "confidence": "high",
                    "guard_applied": True,
                    "guard_type": "online_challenge_audit",
                    "guard_reason": audited["online_challenge_reason"],
                }
            )
        return audited
    except Exception as exc:
        audited = dict(verdict)
        audited.update(
            {
                "online_challenge_audit_error": str(exc),
                "online_challenge_applied": False,
                "online_challenge_reason": "Online challenge audit failed; preserving existing verifier decision.",
            }
        )
        return audited


def run_verifier(
    sample: Dict[str, Any],
    *,
    direct_candidate: Dict[str, Any],
    bm25_candidate: Dict[str, Any],
    online_candidate: Dict[str, Any],
    online_evidence: Dict[str, Any],
    local_docs: Sequence[Dict[str, Any]],
    provider: str,
    model: str,
    timeout_s: int,
    azure_suffix: Optional[str],
    mode: str = "llm",
    max_local_docs: int = 6,
    enable_second_search: bool = False,
    second_search_provider: str = "auto",
    second_search_model: str = "gpt-4o",
    second_search_azure_suffix: Optional[str] = "",
    second_search_timeout_s: int = 120,
    second_search_max_retries: int = 1,
    second_search_max_sources: int = 6,
    second_search_context_size: str = "medium",
    second_search_cache_dir: Optional[Path] = None,
    reuse_second_search_cache: bool = False,
    second_search_sleep_s: float = 0.0,
    second_search_quality_threshold: float = 0.8,
    enable_online_challenge_audit: bool = False,
) -> Dict[str, Any]:
    """Return a normalized verifier verdict and supporting trace fields."""
    source_quality_report = summarize_source_quality(list(online_evidence.get("sources") or []))
    labels = {str(k).strip().upper() for k in sample["options"]}
    fallback = heuristic_verdict(
        sample,
        direct_candidate=direct_candidate,
        bm25_candidate=bm25_candidate,
        online_candidate=online_candidate,
        source_quality_report=source_quality_report,
    )

    if str(mode).strip().lower() == "heuristic":
        return {
            **fallback,
            "mode": "heuristic",
            "raw_response": "",
            "source_quality": source_quality_report,
        }

    raw_response = ""
    try:
        raw_response = call_chat_model(
            build_verifier_messages(
                sample,
                direct_candidate=direct_candidate,
                bm25_candidate=bm25_candidate,
                online_candidate=online_candidate,
                online_evidence=online_evidence,
                source_quality_report=source_quality_report,
                local_docs=local_docs,
                max_local_docs=max_local_docs,
            ),
            provider=provider,
            model=model,
            timeout_s=timeout_s,
            azure_suffix=azure_suffix,
        )
        payload = parse_json_object(raw_response)
        verdict = normalize_verdict(payload, labels=labels)
        if verdict["final_pred"] is None:
            parsed = parse_answer(raw_response, sample["options"])
            verdict["final_pred"] = parsed
        if verdict["final_pred"] is None:
            verdict = dict(fallback)
            verdict["decision"] = "needs_more_search"
            verdict["reason"] = "Verifier response did not contain a valid option; heuristic fallback was used."
        verdict = _apply_post_verifier_guards(
            verdict,
            sample,
            direct_candidate=direct_candidate,
            bm25_candidate=bm25_candidate,
            online_candidate=online_candidate,
            online_evidence=online_evidence,
            source_quality_report=source_quality_report,
            local_docs=local_docs,
            provider=provider,
            model=model,
            timeout_s=timeout_s,
            azure_suffix=azure_suffix,
            max_local_docs=max_local_docs,
            enable_second_search=enable_second_search,
            second_search_provider=second_search_provider,
            second_search_model=second_search_model,
            second_search_azure_suffix=second_search_azure_suffix,
            second_search_timeout_s=second_search_timeout_s,
            second_search_max_retries=second_search_max_retries,
            second_search_max_sources=second_search_max_sources,
            second_search_context_size=second_search_context_size,
            second_search_cache_dir=second_search_cache_dir,
            reuse_second_search_cache=reuse_second_search_cache,
            second_search_sleep_s=second_search_sleep_s,
            second_search_quality_threshold=second_search_quality_threshold,
        )
        verdict = _apply_online_challenge_audit(
            verdict,
            sample,
            direct_candidate=direct_candidate,
            bm25_candidate=bm25_candidate,
            online_candidate=online_candidate,
            enable_online_challenge_audit=enable_online_challenge_audit,
            second_search_provider=second_search_provider,
            second_search_model=second_search_model,
            second_search_azure_suffix=second_search_azure_suffix,
            second_search_timeout_s=second_search_timeout_s,
            second_search_max_retries=second_search_max_retries,
            second_search_max_sources=second_search_max_sources,
            second_search_context_size=second_search_context_size,
            second_search_cache_dir=second_search_cache_dir,
            reuse_second_search_cache=reuse_second_search_cache,
            second_search_sleep_s=second_search_sleep_s,
            adjudicator_provider=provider,
            adjudicator_model=model,
            adjudicator_azure_suffix=azure_suffix,
            adjudicator_timeout_s=timeout_s,
        )
        verdict.update(
            {
                "mode": "llm",
                "raw_response": raw_response,
                "source_quality": source_quality_report,
            }
        )
        return verdict
    except Exception as exc:
        verdict = dict(fallback)
        verdict.update(
            {
                "mode": "heuristic_fallback",
                "raw_response": raw_response,
                "error": str(exc),
                "source_quality": source_quality_report,
            }
        )
        return verdict


def verdict_to_json(verdict: Dict[str, Any]) -> str:
    return json.dumps(verdict, indent=2, ensure_ascii=False)
