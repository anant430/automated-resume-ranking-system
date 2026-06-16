#!/usr/bin/env python3
import argparse
import heapq
import json
from pathlib import Path

import yaml

from src.export import write_submission
from src.features import extract_features
from src.reasoning import build_reasoning
from src.scorer import score_candidate
from src.loader import load_candidates

ROOT = Path(__file__).resolve().parent
TOP_K = 100


def load_config() -> tuple[dict, dict]:
    with open(ROOT / "config" / "job_query.yaml", encoding="utf-8") as handle:
        job_config = yaml.safe_load(handle)
    with open(ROOT / "config" / "ai_skills.json", encoding="utf-8") as handle:
        ai_config = json.load(handle)
    return job_config, ai_config


def _rank_key(row: tuple[float, str, str]) -> tuple[float, int]:
    score, candidate_id, _ = row
    numeric_id = int(candidate_id.split("_")[1])
    return score, -numeric_id


def rank_candidates(candidates_path: str | Path) -> list[tuple[float, str, str]]:
    job_config, ai_config = load_config()
    scored: list[tuple[float, str, str]] = []

    for candidate in load_candidates(candidates_path):
        features = extract_features(candidate, job_config, ai_config)
        score = score_candidate(features, job_config)
        reasoning = build_reasoning(features)
        scored.append((score, features.candidate_id, reasoning))

    ranked = heapq.nlargest(TOP_K, scored, key=_rank_key)
    ranked.sort(key=lambda row: (-round(row[0], 4), row[1]))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob challenge.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or sample JSON.")
    parser.add_argument("--out", required=True, help="Output CSV path (e.g. team_dev.csv).")
    args = parser.parse_args()

    ranked = rank_candidates(args.candidates)
    if len(ranked) < TOP_K:
        raise SystemExit(
            f"Need at least {TOP_K} candidates to build a valid submission; found {len(ranked)}."
        )

    write_submission(ranked, args.out)
    print(f"Wrote top {TOP_K} candidates to {args.out}")


if __name__ == "__main__":
    main()
