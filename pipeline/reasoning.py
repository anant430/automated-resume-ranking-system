"""Reasoning generator V2 — Blueprint v8 Step 6."""

from __future__ import annotations

import random
from typing import Any


def get_tone_band(rank: int) -> str:
    if rank <= 10:
        return "strong"
    if rank <= 30:
        return "solid"
    if rank <= 60:
        return "moderate"
    return "limited"


def build_reasoning(cid: str, rank: int, feat: dict[str, Any], jd_signals: dict[str, Any]) -> str:
    tone = get_tone_band(rank)
    random.seed(hash(cid) % 99999)

    title = feat["current_title"]
    years = feat["total_years_experience"]
    employer = feat["most_recent_employer"]
    emp_type = feat["most_recent_employer_type"]

    raw_skills = feat.get("top_matched_skills", [])
    top_skills = (raw_skills + ["ML engineering experience", "adjacent technical background"])[:2]

    shipped_raw = feat.get("most_notable_shipped_system")
    if shipped_raw:
        shipped_line_available = True
        shipped = shipped_raw
    else:
        shipped_line_available = False
        shipped = None

    jd_phrases = jd_signals.get("key_jd_phrases", ["retrieval quality at scale"])
    jd_phrase = random.choice(jd_phrases)

    opening_pool = {
        "strong": [
            (
                f"{years}yr {title} at {employer} ({emp_type}); shipped {shipped} — "
                f'directly satisfying "{jd_phrase}".'
            )
            if shipped_line_available
            else (
                f"{years}yr {title} at {employer} ({emp_type}); "
                f'hands-on {top_skills[0]} directly satisfying "{jd_phrase}".'
            ),
            f'Strongest signal: {top_skills[0]} and {top_skills[1]} in production at {employer}, '
            f'matching "{jd_phrase}".',
            f"Career arc shows consistent upward trajectory; {title} scope at {employer} "
            f"maps well to the seniority level sought.",
            (
                f'Stands out for shipping {shipped} — satisfies "{jd_phrase}" with direct evidence.'
            )
            if shipped_line_available
            else (
                f'Stands out for {top_skills[0]} and {top_skills[1]} — covers "{jd_phrase}" '
                f"with direct evidence."
            ),
        ],
        "solid": [
            f'{years}yr background with {top_skills[0]} at {employer} ({emp_type}); '
            f'covers "{jd_phrase}".',
            f'{title} at {employer} with hands-on {top_skills[0]} work; '
            f'experience aligns with "{jd_phrase}".',
            f'Production deployment at {employer} is the key positive — '
            f'"{jd_phrase}" is explicitly satisfied, not just studied.',
        ],
        "moderate": [
            f'{title} at {employer} shows {top_skills[0]} exposure, '
            f'though "{jd_phrase}" is less clearly evidenced.',
            f'{years}yr career includes {top_skills[0]} usage; '
            f'depth against "{jd_phrase}" needs probing.',
        ],
        "limited": [
            f'Profile shows {top_skills[0]} familiarity but limited evidence against '
            f'"{jd_phrase}" — a material gap versus the JD.',
            f'{title} at {employer} with some {top_skills[0]} exposure; '
            f'"{jd_phrase}" is not clearly met.',
        ],
    }

    opening = random.choice(opening_pool[tone])
    concern_parts: list[str] = []

    if feat["notice_period_days"] > 30:
        concern_parts.append(
            random.choice(
                [
                    f"{feat['notice_period_days']}-day notice period against the ≤30-day target.",
                    f"Timeline friction: {feat['notice_period_days']}-day notice extends the onboarding window.",
                ]
            )
        )

    if feat["services_firm_ratio"] > 0.8:
        firm = feat.get("primary_services_firm") or "a services firm"
        concern_parts.append(
            random.choice(
                [
                    f"Consulting-heavy background ({feat['services_firm_ratio']*100:.0f}% of career at "
                    f"{firm}) — product shipping velocity needs validation.",
                    "Services-firm majority career; the JD's product-cadence preference is the open question.",
                ]
            )
        )

    if feat["months_since_last_commit"] > 18:
        concern_parts.append(
            random.choice(
                [
                    f"Architecture/lead focus for {feat['months_since_last_commit']} months; "
                    f"hands-on IC coding ramp expected.",
                    f"Reduced direct coding in last {feat['months_since_last_commit']}mo — "
                    f"IC reactivation timeline is a soft risk.",
                ]
            )
        )

    if feat["pure_research_ratio"] > 0.8:
        concern_parts.append(
            random.choice(
                [
                    "Predominantly research background — production deployment bias required by the JD must be confirmed.",
                    "Strong academic pedigree; the JD's bias toward shipped systems is the key evaluation question.",
                ]
            )
        )

    if feat["title_chaser_flag"]:
        concern_parts.append(
            random.choice(
                [
                    f"Tenure pattern shows {feat['avg_tenure_months']:.0f}mo average across recent roles — "
                    f"stability at this level is worth probing.",
                    f"Frequent role changes (avg {feat['avg_tenure_months']:.0f}mo tenure) is a fit question "
                    f"worth confirming in interview.",
                ]
            )
        )

    if not concern_parts:
        concern_parts.append(
            random.choice(
                [
                    "No gaps identified; growth edge would be leading retrieval system design at higher scale.",
                    "Strong fit across all dimensions; L2R depth is the one area not explicitly evidenced.",
                    "Near-complete match — only open question is seniority alignment at the team structure level.",
                ]
            )
        )

    return f"{opening} {concern_parts[0]}"


def build_feature_vector(
    cid: str,
    feat: dict[str, Any],
    rrf_score: float,
    rrf_scores: dict[str, float] | None = None,
) -> list[float]:
    if rrf_scores is not None:
        rrf_score = rrf_scores.get(cid, rrf_score)
    return [
        rrf_score,
        feat["exact_skill_matches"],
        feat["esco_skill_score"],
        feat["total_years_experience"],
        int(feat["experience_band_match"]),
        feat["employer_type_score"],
        int(feat["has_production_deployment"]),
        feat["active_ml_tenure"],
        feat["months_since_last_ml_role"],
        feat["trajectory_velocity_score"],
        feat["last_active_days"],
        feat["recruiter_response_rate"],
        feat["profile_completeness"],
        feat["services_firm_ratio"],
        feat["pure_research_ratio"],
        int(feat["title_chaser_flag"]),
        feat["location_score"],
        feat["notice_period_score"],
        int(feat["experience_exceeds_company_age"]),
        feat["tool_birth_year_violations"],
        int(feat["mass_skill_stuffing_flag"]),
        int(feat["chronological_overlap_flag"]),
    ]
