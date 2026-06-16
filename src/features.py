from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateFeatures:
    candidate_id: str
    current_title: str
    years_of_experience: float
    title_score: float
    career_score: float
    skill_score: float
    ai_core_skill_count: int
    skill_trust: float
    experience_score: float
    education_score: float
    response_rate: float
    github_score: float
    assessment_score: float


def _title_tier_score(title: str, title_tiers: dict) -> float:
    normalized = title.lower().strip()
    best = 0.0
    for tier in title_tiers.values():
        for keyword in tier["keywords"]:
            if keyword in normalized:
                best = max(best, tier["score"])
    return best


def _skill_value(skill: dict, core_skills: set[str], proficiency_weights: dict) -> tuple[float, bool]:
    name = skill.get("name", "")
    if name not in core_skills:
        return 0.0, False

    proficiency = proficiency_weights.get(skill.get("proficiency", "beginner"), 0.25)
    endorsements = skill.get("endorsements", 0)
    duration = skill.get("duration_months", 0)

    trust = min(1.0, endorsements / 20.0) * min(1.0, duration / 24.0)
    if skill.get("proficiency") in {"advanced", "expert"} and endorsements == 0 and duration < 12:
        trust *= 0.4

    return proficiency * (0.55 + 0.45 * trust), True


def extract_features(candidate: dict, job_config: dict, ai_config: dict) -> CandidateFeatures:
    profile = candidate["profile"]
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    career_history = candidate.get("career_history", [])
    signals = candidate["redrob_signals"]

    core_skills = set(ai_config["core_skills"])
    proficiency_weights = ai_config["proficiency_weights"]

    title_score = _title_tier_score(profile.get("current_title", ""), job_config["title_tiers"])
    career_scores = [
        _title_tier_score(role.get("title", ""), job_config["title_tiers"])
        for role in career_history
    ]
    career_score = max(career_scores) if career_scores else 0.0

    skill_values = []
    ai_core_skill_count = 0
    trust_values = []
    for skill in skills:
        value, is_core = _skill_value(skill, core_skills, proficiency_weights)
        if is_core:
            ai_core_skill_count += 1
            skill_values.append(value)
            endorsements = skill.get("endorsements", 0)
            duration = skill.get("duration_months", 0)
            trust_values.append(min(1.0, endorsements / 15.0) * min(1.0, duration / 18.0))

    if skill_values:
        skill_score = min(1.0, sum(sorted(skill_values, reverse=True)[:8]) / 4.0)
        skill_trust = sum(trust_values) / len(trust_values)
    else:
        skill_score = 0.0
        skill_trust = 0.0

    years = float(profile.get("years_of_experience", 0))
    exp_cfg = job_config["experience"]
    if years < exp_cfg["hard_min"] or years > exp_cfg["hard_max"]:
        experience_score = 0.2
    elif exp_cfg["ideal_min"] <= years <= exp_cfg["ideal_max"]:
        experience_score = 1.0
    elif years < exp_cfg["ideal_min"]:
        experience_score = max(0.35, years / exp_cfg["ideal_min"])
    else:
        experience_score = max(0.45, 1.0 - (years - exp_cfg["ideal_max"]) / 10.0)

    edu_fields = [field.lower() for field in job_config["education_fields"]]
    tier_bonus = job_config["education_tier_bonus"]
    education_scores = []
    for entry in education:
        field = entry.get("field_of_study", "").lower()
        tier = entry.get("tier", "unknown")
        field_match = any(token in field for token in edu_fields)
        base = 0.75 if field_match else 0.35
        education_scores.append(base * tier_bonus.get(tier, tier_bonus["unknown"]))
    education_score = max(education_scores) if education_scores else 0.25

    assessment_scores = list(signals.get("skill_assessment_scores", {}).values())
    assessment_score = (
        sum(assessment_scores) / len(assessment_scores) / 100.0
        if assessment_scores
        else 0.0
    )

    github_raw = signals.get("github_activity_score", -1)
    github_score = github_raw / 100.0 if github_raw >= 0 else 0.0

    return CandidateFeatures(
        candidate_id=candidate["candidate_id"],
        current_title=profile.get("current_title", "Unknown"),
        years_of_experience=years,
        title_score=title_score,
        career_score=career_score,
        skill_score=skill_score,
        ai_core_skill_count=ai_core_skill_count,
        skill_trust=skill_trust,
        experience_score=experience_score,
        education_score=education_score,
        response_rate=float(signals.get("recruiter_response_rate", 0.0)),
        github_score=github_score,
        assessment_score=assessment_score,
    )
