"""Deterministic source quality scoring for CLEAR."""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse


HIGH_AUTHORITY_DOMAINS = {
    "cdc.gov": "public health authority",
    "nih.gov": "government medical authority",
    "ncbi.nlm.nih.gov": "NIH/NCBI medical reference",
    "pubmed.ncbi.nlm.nih.gov": "PubMed biomedical citation",
    "pmc.ncbi.nlm.nih.gov": "PubMed Central full text",
    "nccn.org": "clinical guideline organization",
    "nice.org.uk": "clinical guideline organization",
    "who.int": "public health authority",
    "acog.org": "specialty society",
    "aap.org": "specialty society",
    "heart.org": "specialty society",
    "idsociety.org": "specialty society",
    "diabetesjournals.org": "specialty journal/guideline source",
    "cochrane.org": "systematic review source",
    "nejm.org": "major medical journal",
    "jamanetwork.com": "major medical journal",
    "thelancet.com": "major medical journal",
    "bmj.com": "major medical journal",
    "uptodate.com": "clinician reference",
    "merckmanuals.com": "clinician reference",
    "msdmanuals.com": "clinician reference",
    "mayoclinic.org": "academic medical center reference",
}

MEDIUM_AUTHORITY_DOMAINS = {
    "aafp.org": "medical professional society/reference",
    "clevelandclinic.org": "academic medical center reference",
    "mdcalc.com": "clinical decision support reference",
    "emedicine.medscape.com": "clinician reference",
    "medscape.com": "clinician reference",
    "radiopaedia.org": "specialty educational reference",
    "pathologyoutlines.com": "specialty educational reference",
    "statpearls.com": "medical education reference",
    "link.springer.com": "publisher-hosted peer-reviewed literature",
    "sciencedirect.com": "publisher-hosted peer-reviewed literature",
    "academic.oup.com": "publisher-hosted peer-reviewed literature",
    "onlinelibrary.wiley.com": "publisher-hosted peer-reviewed literature",
    "tandfonline.com": "publisher-hosted peer-reviewed literature",
    "nature.com": "publisher-hosted peer-reviewed literature",
    "frontiersin.org": "publisher-hosted peer-reviewed literature",
}

LOW_AUTHORITY_DOMAINS = {
    "droracle.ai": "automated medical content risk",
    "consensus.app": "secondary search summary risk",
    "wikipedia.org": "crowd-edited reference",
    "medium.com": "blog platform",
    "wordpress.com": "blog platform",
    "blogspot.com": "blog platform",
}


def normalize_domain(url: str) -> str:
    """Return a stable host for source-quality rules."""
    host = urlparse(str(url or "")).netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _domain_rule(domain: str) -> tuple[float, str, str]:
    if not domain:
        return 0.25, "low", "missing URL/domain"

    for suffix, reason in HIGH_AUTHORITY_DOMAINS.items():
        if domain == suffix or domain.endswith("." + suffix):
            return 0.90, "high", reason

    for suffix, reason in MEDIUM_AUTHORITY_DOMAINS.items():
        if domain == suffix or domain.endswith("." + suffix):
            return 0.70, "medium", reason

    for suffix, reason in LOW_AUTHORITY_DOMAINS.items():
        if domain == suffix or domain.endswith("." + suffix):
            return 0.25, "low", reason

    if domain.endswith(".edu"):
        return 0.65, "medium", "academic domain"
    if domain.endswith(".gov"):
        return 0.75, "medium", "government domain"
    if domain.endswith(".org"):
        return 0.50, "medium", "organization domain without explicit medical authority rule"
    return 0.40, "low", "unrecognized source domain"


def score_source(source: Dict[str, Any]) -> Dict[str, Any]:
    """Attach deterministic quality metadata to one online source."""
    domain = normalize_domain(str(source.get("url", "")))
    score, tier, reason = _domain_rule(domain)
    title = str(source.get("title", "")).lower()
    claim = str(source.get("claim", "")).lower()
    text = f"{title} {claim}"

    if "guideline" in text or "recommendation" in text:
        score = min(1.0, score + 0.05)
    if "meta-analysis" in text or "systematic review" in text:
        score = min(1.0, score + 0.05)
    if "case report" in text:
        score = max(0.0, score - 0.10)

    if score >= 0.80:
        tier = "high"
    elif score >= 0.50:
        tier = "medium"
    else:
        tier = "low"

    enriched = dict(source)
    enriched.update(
        {
            "domain": domain,
            "quality_score": round(score, 2),
            "quality_tier": tier,
            "quality_reason": reason,
        }
    )
    return enriched


def score_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [score_source(item) for item in sources if isinstance(item, dict)]


def summarize_source_quality(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize scored sources for verifier prompts and traces."""
    scored = score_sources(sources)
    if not scored:
        return {
            "sources": [],
            "max_quality_score": 0.0,
            "avg_quality_score": 0.0,
            "high_quality_count": 0,
            "medium_quality_count": 0,
            "low_quality_count": 0,
            "trusted_source_present": False,
        }

    tiers = [str(item.get("quality_tier", "low")) for item in scored]
    scores = [float(item.get("quality_score", 0.0)) for item in scored]
    return {
        "sources": scored,
        "max_quality_score": round(max(scores), 2),
        "avg_quality_score": round(sum(scores) / len(scores), 2),
        "high_quality_count": tiers.count("high"),
        "medium_quality_count": tiers.count("medium"),
        "low_quality_count": tiers.count("low"),
        "trusted_source_present": any(score >= 0.80 for score in scores),
    }


def format_source_quality_report(report: Dict[str, Any], *, max_sources: int = 8) -> str:
    """Render source quality compactly for verifier prompts."""
    rows = [
        f"max_quality_score: {report.get('max_quality_score', 0.0)}",
        f"avg_quality_score: {report.get('avg_quality_score', 0.0)}",
        f"trusted_source_present: {bool(report.get('trusted_source_present'))}",
    ]
    for idx, source in enumerate(list(report.get("sources") or [])[:max_sources], start=1):
        rows.append(
            "Source "
            f"{idx}: tier={source.get('quality_tier')} score={source.get('quality_score')} "
            f"domain={source.get('domain')} reason={source.get('quality_reason')}"
        )
        title = str(source.get("title", "")).strip()
        claim = str(source.get("claim", "")).strip()
        if title:
            rows.append(f"  title: {title}")
        if claim:
            rows.append(f"  claim: {claim[:500]}")
    return "\n".join(rows)
