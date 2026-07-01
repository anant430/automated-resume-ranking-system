"""Feature extraction for candidate ranking — Blueprint v8 §0.6."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

from pipeline.constants import ESCO_SKILL_GROUPS
from pipeline.honeypot import months_between


def tokenize_text(text: str) -> list[str]:
    """Shared tokenization for BM25 index and rank.py jd_tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def candidate_profile_text(candidate: dict[str, Any]) -> str:
    parts = [
        candidate.get("summary", ""),
        " ".join(s.get("name", "") for s in candidate.get("skills", [])),
        " ".join(r.get("title", "") + " " + r.get("company", "") for r in candidate.get("roles", [])),
        " ".join(candidate.get("shipped_systems", [])),
    ]
    return " ".join(parts)


def sum_years_experience(roles: list[dict[str, Any]]) -> float:
    total = 0.0
    for role in roles:
        tenure = role.get("tenure_years")
        if tenure is not None:
            total += tenure
        else:
            total += months_between(role.get("start_date", ""), role.get("end_date")) / 12
    return round(total, 1)


def categorize_title(title: str) -> str:
    title_l = title.lower()
    if any(k in title_l for k in ("architect", "principal")):
        return "architect"
    if any(k in title_l for k in ("lead", "manager", "head")):
        return "tech_lead"
    if any(k in title_l for k in ("research", "scientist", "phd")):
        return "research"
    return "ic"


def has_ic_role_in_window(roles: list[dict[str, Any]], months: int = 18) -> bool:
    cutoff = datetime.utcnow()
    for role in reversed(roles):
        end = role.get("end_date")
        end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()
        months_ago = (cutoff - end_dt).days / 30.0
        if months_ago <= months:
            if categorize_title(role.get("title", "")) == "ic":
                return True
    return False


def candidate_domains(candidate: dict[str, Any]) -> list[str]:
    domains = []
    text = candidate_profile_text(candidate).lower()
    if any(k in text for k in ("nlp", "bert", "llm", "language")):
        domains.append("nlp")
    if any(k in text for k in ("retrieval", "search", "faiss", "bm25", "ndcg")):
        domains.append("ir")
    if any(k in text for k in ("vision", "cv", "opencv")):
        domains.append("computer_vision")
    if any(k in text for k in ("speech", "asr", "tts")):
        domains.append("speech")
    if any(k in text for k in ("robot", "robotics")):
        domains.append("robotics")
    return domains or ["general_ml"]


def infer_primary_domain(candidate: dict[str, Any]) -> str:
    domains = candidate_domains(candidate)
    priority = ["nlp", "ir", "computer_vision", "speech", "robotics", "general_ml"]
    for domain in priority:
        if domain in domains:
            return domain
    return "general_ml"


def primary_services_firm(candidate: dict[str, Any]) -> str | None:
    for role in candidate.get("roles", []):
        if role.get("employer_type") == "services":
            return role.get("company")
    return None


def most_notable_shipped_system(candidate: dict[str, Any]) -> str | None:
    systems = candidate.get("shipped_systems", [])
    if systems:
        return systems[0]
    for role in reversed(candidate.get("roles", [])):
        for project in role.get("projects", []):
            if project:
                return project
    return None


def top_n_matched_skills(
    candidate: dict[str, Any], jd_signals: dict[str, Any], n: int = 2
) -> list[str]:
    required = set(s.lower() for s in jd_signals.get("required_skills", []))
    synonyms = set(s.lower() for s in jd_signals.get("synonyms", []))
    target = required | synonyms
    matched = []
    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        if name in target or any(t in name for t in target):
            matched.append(skill.get("name", name))
    return matched[:n]


def esco_skill_score(candidate: dict[str, Any], jd_signals: dict[str, Any]) -> float:
    required = set(s.lower() for s in jd_signals.get("required_skills", []))
    candidate_skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    if not required:
        return 0.5
    exact = len(required & candidate_skills) / len(required)
    adjacency = 0.0
    for group_skills in ESCO_SKILL_GROUPS.values():
        group_lower = {g.lower() for g in group_skills}
        if required & group_lower and candidate_skills & group_lower:
            adjacency += 0.15
    return min(1.0, exact + adjacency)


def exact_skill_matches(candidate: dict[str, Any], jd_signals: dict[str, Any]) -> int:
    required = set(s.lower() for s in jd_signals.get("required_skills", []))
    candidate_skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    return len(required & candidate_skills)


def experience_band_match(candidate: dict[str, Any], jd_signals: dict[str, Any]) -> bool:
    band = jd_signals.get("experience_band", {"min": 5, "max": 9})
    years = sum_years_experience(candidate.get("roles", []))
    return band.get("min", 5) <= years <= band.get("max", 9)


def location_score(candidate: dict[str, Any], jd_signals: dict[str, Any] | None = None) -> float:
    loc = candidate.get("location", "").lower()
    prefs = (jd_signals or {}).get("location_preferences", {})
    if loc in ("pune", "noida"):
        return prefs.get("pune_noida", 1.0)
    if loc in ("hyderabad", "mumbai", "delhi", "bangalore", "bengaluru"):
        return 0.9
    if loc in ("india", "") or "india" in loc:
        return 0.7
    return prefs.get("outside_india", 0.4)


def notice_period_score(days: int) -> float:
    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.85
    if days <= 90:
        return 0.7
    if days <= 120:
        return 0.5
    return 0.3


def trajectory_velocity_score(roles: list[dict[str, Any]]) -> float:
    if len(roles) < 2:
        return 0.5
    deltas = []
    for prev, curr in zip(roles[:-1], roles[1:]):
        level_change = curr.get("title_level", 2) - prev.get("title_level", 2)
        years_in_role = max(months_between(prev.get("start_date", ""), prev.get("end_date")) / 12, 0.5)
        deltas.append(level_change / years_in_role)
    weights = np.linspace(1.0, 2.0, len(deltas))
    raw_velocity = float(np.average(deltas, weights=weights))
    return float(1 / (1 + np.exp(-raw_velocity)))


def bucket_from_score(score: float) -> str:
    if score >= 0.75:
        return "ascending"
    if score >= 0.45:
        return "steady"
    return "flat"


def compute_candidate_features(
    candidate: dict[str, Any],
    jd_signals: dict[str, Any],
    honeypot_signal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roles = candidate.get("roles", [])
    hp = honeypot_signal or {}
    services_roles = [r for r in roles if r.get("employer_type") == "services"]
    research_roles = [r for r in roles if r.get("employer_type") == "research"]
    tenures = [
        r.get("tenure_years") or months_between(r.get("start_date", ""), r.get("end_date")) / 12
        for r in roles
    ]
    avg_tenure_months = (sum(tenures) / len(tenures) * 12) if tenures else 24.0
    title_chaser = avg_tenure_months < 18 and len(roles) >= 3

    feat: dict[str, Any] = {
        "months_since_last_ml_role": candidate.get("months_since_last_ml_role", 6),
        "active_ml_tenure": candidate.get("active_ml_tenure", 3.0),
        "months_since_last_commit": candidate.get("months_since_last_commit", 3),
        "last_active_days": candidate.get("last_active_days", 14),
        "recruiter_response_rate": candidate.get("recruiter_response_rate", 0.7),
        "profile_completeness": candidate.get("profile_completeness", 0.85),
        "employer_type": roles[-1].get("employer_type", "product") if roles else "product",
        "services_firm_ratio": len(services_roles) / max(len(roles), 1),
        "pure_research_ratio": len(research_roles) / max(len(roles), 1),
        "has_production_deployment": bool(candidate.get("shipped_systems")),
        "career_trajectory": "steady",
        "avg_tenure_months": avg_tenure_months,
        "title_chaser_flag": title_chaser,
        "location_score": location_score(candidate, jd_signals),
        "notice_period_score": notice_period_score(candidate.get("notice_period_days", 90)),
        "exact_skill_matches": exact_skill_matches(candidate, jd_signals),
        "esco_skill_score": esco_skill_score(candidate, jd_signals),
        "experience_band_match": experience_band_match(candidate, jd_signals),
        "experience_exceeds_company_age": hp.get("experience_exceeds_company_age", False),
        "tool_birth_year_violations": hp.get("tool_birth_year_violations", 0),
        "mass_skill_stuffing_flag": hp.get("mass_skill_stuffing_flag", False),
        "chronological_overlap_flag": hp.get("chronological_overlap_flag", False),
        "total_years_experience": sum_years_experience(roles),
        "current_title": roles[-1].get("title", "ML Engineer") if roles else "ML Engineer",
        "most_recent_employer": roles[-1].get("company", "Unknown") if roles else "Unknown",
        "most_recent_employer_type": roles[-1].get("employer_type", "product") if roles else "product",
        "top_matched_skills": top_n_matched_skills(candidate, jd_signals, n=2),
        "most_notable_shipped_system": most_notable_shipped_system(candidate),
        "notice_period_days": candidate.get("notice_period_days", 90),
        "employer_type_history": [r.get("employer_type", "product") for r in roles],
        "has_product_company_experience": any(r.get("employer_type") == "product" for r in roles),
        "current_title_category": categorize_title(roles[-1].get("title", "")) if roles else "ic",
        "has_ic_role_last_18mo": has_ic_role_in_window(roles, months=18),
        "primary_domain": infer_primary_domain(candidate),
        "has_any_nlp_ir_experience": any(d in ("nlp", "ir") for d in candidate_domains(candidate)),
    }
    feat["primary_services_firm"] = (
        primary_services_firm(candidate) if feat["services_firm_ratio"] > 0 else None
    )
    feat["employer_type_score"] = {"product": 1.0, "services": 0.5, "consulting": 0.2}.get(
        feat["most_recent_employer_type"], 0.5
    )
    feat["trajectory_velocity_score"] = trajectory_velocity_score(roles)
    feat["career_trajectory"] = bucket_from_score(feat["trajectory_velocity_score"])
    return feat
