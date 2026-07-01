"""Honeypot detection — hard exclusion rules from Blueprint v8 §0.5."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pipeline.constants import COMPANY_FOUNDING_YEARS, TOOL_BIRTH_YEARS


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value[: len(fmt.replace("%", "0"))], fmt)
        except ValueError:
            continue
    return None


def months_between(start: str, end: str | None) -> float:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end) if end else datetime.utcnow()
    if not start_dt or not end_dt:
        return 12.0
    return max((end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month), 1)


def company_founding_age(company: str) -> float:
    year = COMPANY_FOUNDING_YEARS.get(company, 2015)
    return max(2025 - year, 1)


def has_overlapping_full_time_roles(roles: list[dict[str, Any]]) -> bool:
    intervals: list[tuple[datetime, datetime]] = []
    for role in roles:
        start = _parse_date(role.get("start_date", ""))
        end = _parse_date(role.get("end_date", "")) or datetime.utcnow()
        if start and end and end > start:
            intervals.append((start, end))
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            return True
    return False


def count_tool_birth_violations(candidate: dict[str, Any]) -> int:
    count = 0
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name in TOOL_BIRTH_YEARS:
            max_possible = 2025 - TOOL_BIRTH_YEARS[name]
            if skill.get("years", 0) > max_possible:
                count += 1
    return count


def is_honeypot(candidate: dict[str, Any]) -> tuple[bool, str | None]:
    for role in candidate.get("roles", []):
        tenure = role.get("tenure_years")
        if tenure is None:
            tenure = months_between(role.get("start_date", ""), role.get("end_date")) / 12
        if tenure > company_founding_age(role.get("company", "")):
            return True, "experience_exceeds_company_age"

    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        if name in TOOL_BIRTH_YEARS:
            max_possible = 2025 - TOOL_BIRTH_YEARS[name]
            if skill.get("years", 0) > max_possible:
                return True, f"{name}_older_than_tool"

    expert_skills = [s for s in candidate.get("skills", []) if s.get("proficiency") == "expert"]
    zero_years = [s for s in expert_skills if s.get("years", 0) == 0]
    if len(zero_years) >= 10:
        return True, "mass_skill_stuffing"

    if has_overlapping_full_time_roles(candidate.get("roles", [])):
        return True, "chronological_overlap"

    return False, None


def build_honeypot_signals(candidates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    signals: dict[str, dict[str, Any]] = {}
    for cid, candidate in candidates.items():
        is_hp, reason = is_honeypot(candidate)
        signals[cid] = {
            "is_honeypot": is_hp,
            "experience_exceeds_company_age": reason == "experience_exceeds_company_age",
            "tool_birth_year_violations": count_tool_birth_violations(candidate),
            "mass_skill_stuffing_flag": reason == "mass_skill_stuffing",
            "chronological_overlap_flag": reason == "chronological_overlap",
        }
    return signals


def build_honeypot_flags(honeypot_signals: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {cid: sig["is_honeypot"] for cid, sig in honeypot_signals.items()}
