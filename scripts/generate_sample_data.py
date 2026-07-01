#!/usr/bin/env python3
"""Generate synthetic candidates.jsonl for local development and testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SKILLS = [
    "Python", "PyTorch", "TensorFlow", "FAISS", "XGBoost", "BM25", "BGE",
    "NDCG", "Docker", "Kubernetes", "Elasticsearch", "transformers", "MLflow",
]
COMPANIES = [
    ("Flipkart", "product"), ("Razorpay", "product"), ("Swiggy", "product"),
    ("Infosys", "services"), ("TCS", "services"), ("Accenture", "consulting"),
    ("SearchCo", "product"), ("MLWorks", "product"), ("DataLabs", "product"),
    ("OpenAI", "research"),
]
LOCATIONS = ["Pune", "Noida", "Bangalore", "Mumbai", "Delhi", "Hyderabad"]
TITLES = [
    ("ML Engineer", 2), ("Senior ML Engineer", 3), ("Staff ML Engineer", 4),
    ("ML Architect", 5), ("Research Scientist", 3),
]
SHIPPED = [
    "FAISS+BGE hybrid retrieval pipeline",
    "XGBoost L2R ranker with NDCG eval harness",
    "BM25 + dense fusion search API",
    "Recruiter engagement scoring service",
    "Vector index rebuild pipeline on CPU",
]


def make_candidate(cid: str, rng: random.Random, honeypot: bool = False) -> dict:
    num_roles = rng.randint(2, 5)
    roles = []
    year = 2016
    for i in range(num_roles):
        company, emp_type = rng.choice(COMPANIES)
        title, level = rng.choice(TITLES)
        tenure = rng.uniform(1.0, 3.5)
        start = f"{year}-01-01"
        end_year = year + int(tenure)
        end = None if i == num_roles - 1 else f"{end_year}-06-01"
        roles.append({
            "title": title,
            "company": company,
            "employer_type": emp_type,
            "title_level": level,
            "start_date": start,
            "end_date": end,
            "tenure_years": tenure if i == num_roles - 1 else None,
            "projects": [rng.choice(SHIPPED)],
        })
        year = end_year

    skill_list = []
    for skill in rng.sample(SKILLS, k=rng.randint(5, 10)):
        skill_list.append({
            "name": skill,
            "years": rng.randint(1, 6),
            "proficiency": rng.choice(["intermediate", "expert", "advanced"]),
        })

    candidate = {
        "candidate_id": cid,
        "location": rng.choice(LOCATIONS),
        "notice_period_days": rng.choice([15, 30, 45, 60, 90, 120]),
        "summary": "ML engineer focused on retrieval, ranking, and production ML systems.",
        "skills": skill_list,
        "roles": roles,
        "shipped_systems": rng.sample(SHIPPED, k=rng.randint(1, 2)),
        "months_since_last_ml_role": rng.randint(0, 12),
        "active_ml_tenure": round(rng.uniform(2.0, 6.0), 1),
        "months_since_last_commit": rng.randint(1, 24),
        "last_active_days": rng.randint(1, 60),
        "recruiter_response_rate": round(rng.uniform(0.4, 0.95), 2),
        "profile_completeness": round(rng.uniform(0.6, 1.0), 2),
    }

    if honeypot:
        candidate["skills"].append({
            "name": "PyTorch",
            "years": 15,
            "proficiency": "expert",
        })
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/candidates.jsonl")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--honeypots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    honeypot_indices = set(rng.sample(range(args.count), k=min(args.honeypots, args.count)))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            cid = f"cand_{i:05d}"
            candidate = make_candidate(cid, rng, honeypot=i in honeypot_indices)
            f.write(json.dumps(candidate) + "\n")

    print(f"Wrote {args.count} candidates to {out_path} ({len(honeypot_indices)} honeypots)")


if __name__ == "__main__":
    main()
