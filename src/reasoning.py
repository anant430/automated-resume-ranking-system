from .features import CandidateFeatures


def build_reasoning(features: CandidateFeatures) -> str:
    return (
        f"{features.current_title} with {features.years_of_experience:.1f} yrs; "
        f"{features.ai_core_skill_count} AI core skills; "
        f"response rate {features.response_rate:.2f}."
    )
