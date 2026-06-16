from __future__ import annotations

from .features import CandidateFeatures


def score_candidate(features: CandidateFeatures, job_config: dict) -> float:
    weights = job_config["scoring_weights"]
    honeypot = job_config["honeypot"]

    base = (
        weights["title"] * features.title_score
        + weights["skills"] * features.skill_score
        + weights["experience"] * features.experience_score
        + weights["education"] * features.education_score
        + weights["career_history"] * features.career_score
    )

    if (
        features.ai_core_skill_count >= honeypot["min_ai_skills_for_penalty"]
        and features.title_score <= honeypot["max_title_score_for_penalty"]
    ):
        base *= honeypot["penalty_multiplier"]

    if features.skill_trust < 0.25 and features.ai_core_skill_count >= 4:
        base *= honeypot["low_trust_multiplier"]

    behavioral = (
        0.82
        + 0.12 * features.response_rate
        + 0.04 * features.github_score
        + 0.02 * features.assessment_score
    )
    behavioral = min(1.08, max(0.82, behavioral))

    return round(min(1.0, base * behavioral), 6)
