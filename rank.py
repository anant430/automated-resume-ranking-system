#!/usr/bin/env python3
"""Phase 1 online ranking — Blueprint v8 (CPU-only, no network)."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"

try:
    import torch

    assert not torch.cuda.is_available(), "GPU visible — check CUDA_VISIBLE_DEVICES before running rank.py"
except ImportError:
    pass
print("CPU-only confirmed: no GPU context detected")

import argparse
import csv
import json
import pickle
import platform
import signal
import time

try:
    import resource
except ImportError:
    resource = None  # Windows — RAM check skipped locally; enforced in Linux Docker

import faiss
import numpy as np
import xgboost as xgb
from rank_bm25 import BM25Okapi

from pipeline.features import tokenize_text
from pipeline.reasoning import build_feature_vector, build_reasoning

parser = argparse.ArgumentParser()
parser.add_argument("--candidates", required=True)
parser.add_argument("--team-id", required=True, help="Your registered participant ID e.g. team_042")
parser.add_argument("--out", default=None)
args = parser.parse_args()
output_path = args.out or f"{args.team_id}.csv"

_RANK_START = time.time()
if hasattr(signal, "alarm"):
    signal.alarm(300)

all_candidate_ids = set()
with open(args.candidates, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        all_candidate_ids.add(obj["candidate_id"])
print(f"Loaded {len(all_candidate_ids)} candidate IDs")

honeypots = pickle.load(open("honeypot_flags.pkl", "rb"))
features = pickle.load(open("candidate_features.pkl", "rb"))
engagement_model = pickle.load(open("engagement_model.pkl", "rb"))
index_to_cid = pickle.load(open("index_to_cid.pkl", "rb"))

jd_signals = json.load(open("jd_signals.json", encoding="utf-8"))
jd_text = " ".join([
    " ".join(jd_signals.get("required_skills", [])),
    " ".join(jd_signals.get("synonyms", [])),
])
jd_tokens = tokenize_text(jd_text)


def validate_and_write(ranked_rows, output_path="team_xxx.csv"):
    """Complete validation before writing. Fails loudly, never silently."""
    print("\n=== VALIDATION REPORT ===")

    artifacts = [
        "faiss_index.bin",
        "bm25_index.pkl",
        "candidate_features.pkl",
        "honeypot_flags.pkl",
        "honeypot_signals.pkl",
        "skill_adjacency.pkl",
        "xgb_model_ndcg.ubj",
        "xgb_model_map.ubj",
        "engagement_model.pkl",
        "index_to_cid.pkl",
        "cid_to_index.pkl",
        "ideal_candidate_embeddings.npy",
        "pseudo_labels.pkl",
    ]
    artifact_gb = sum(os.path.getsize(f) for f in artifacts if os.path.exists(f)) / 1e9

    bge_dir = "bge_finetuned/"
    if os.path.isdir(bge_dir):
        bge_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(bge_dir)
            for f in files
        ) / 1e9
    else:
        bge_gb = 0.0

    total_gb = artifact_gb + bge_gb
    assert total_gb < 4.5, f"DISK BUDGET: {total_gb:.2f} GB exceeds 5GB limit"
    print(f"Disk: {total_gb:.2f} GB / 5.00 GB  (artifacts {artifact_gb:.2f} + encoder {bge_gb:.2f})")

    assert len(ranked_rows) == 100, f"ROW COUNT: {len(ranked_rows)} (expected 100)"
    print(f"Row count: {len(ranked_rows)}")

    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    actual_cols = list(ranked_rows[0].keys())
    assert actual_cols == expected_cols, f"COLUMN ORDER: {actual_cols}"
    print(f"Columns: {actual_cols}")

    ranks = [row["rank"] for row in ranked_rows]
    assert sorted(ranks) == list(range(1, 101)), "RANK VALUES: not exactly 1-100"
    print("Ranks: 1-100, no gaps, no duplicates")

    bad_ids = [row["candidate_id"] for row in ranked_rows if row["candidate_id"] not in all_candidate_ids]
    assert not bad_ids, f"INVALID IDs: {bad_ids[:5]}"
    print("candidate_ids: all valid")

    cids = [row["candidate_id"] for row in ranked_rows]
    assert len(cids) == len(set(cids)), "DUPLICATE candidate_ids"
    print("No duplicate IDs")

    scores = [row["score"] for row in ranked_rows]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"MONOTONICITY VIOLATED at rank {i+1}: {scores[i]} < {scores[i+1]}"
        )
    assert scores[0] >= scores[-1], (
        f"SCORE DIRECTION: rank 1 ({scores[0]}) < rank 100 ({scores[-1]}) — inverted"
    )
    assert len(set(scores)) > 1, "ALL SCORES IDENTICAL — auto-rejection trigger #5"
    print(f"Scores non-increasing & non-constant: {scores[0]:.4f} -> {scores[-1]:.4f}")

    output_hps = [row["candidate_id"] for row in ranked_rows if honeypots.get(row["candidate_id"])]
    hp_rate = len(output_hps) / 100
    assert hp_rate <= 0.10, f"HONEYPOT RATE: {hp_rate:.1%} (threshold 10%)"
    print(f"Honeypot rate: {hp_rate:.1%} ({len(output_hps)} detected, threshold 10%)")
    if output_hps:
        print(f"Non-zero honeypot rate — manually spot-check: {output_hps[:5]}")

    elapsed = time.time() - _RANK_START
    assert elapsed < 290, f"TIME BUDGET: ranking took {elapsed:.1f}s (limit 290s / 5min spec cap)"
    print(f"Wall-clock time: {elapsed:.1f}s / 290s budget")

    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            peak_rss_gb = ru / 1e9
        else:
            peak_rss_gb = ru / 1e6
        assert peak_rss_gb < 15.0, (
            f"RAM: peak {peak_rss_gb:.2f} GB exceeds 16GB limit (platform: {platform.system()})"
        )
        print(f"Peak RAM: {peak_rss_gb:.2f} GB / 16.00 GB  [{platform.system()}]")
    else:
        print("Peak RAM: skipped (resource module unavailable on this platform)")

    assert output_path.endswith(".csv"), f"FILE TYPE: {output_path} must end in .csv — auto-rejection trigger #7"
    print("File extension: .csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=expected_cols)
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(f"\nWritten to {output_path} (UTF-8 encoding)")
    print("=== ALL CHECKS PASSED ===\n")


# Step 1 — Deterministic Exclusion Filter
HARD_DROP = []
SOFT_PENALTY = {}

for cid, feat in features.items():
    if honeypots.get(cid):
        HARD_DROP.append((cid, "honeypot"))
        continue

    if feat["employer_type_history"] == ["consulting"] and feat["has_product_company_experience"] is False:
        HARD_DROP.append((cid, "pure_services"))
        continue

    if (
        feat["months_since_last_commit"] > 18
        and feat["current_title_category"] in ["tech_lead", "architect"]
        and feat["has_ic_role_last_18mo"] is False
    ):
        HARD_DROP.append((cid, "non_coding_lead"))
        continue

    if feat["primary_domain"] in ["computer_vision", "speech", "robotics"] and feat["has_any_nlp_ir_experience"] is False:
        HARD_DROP.append((cid, "domain_mismatch"))
        continue

    multiplier = 1.0
    if feat["services_firm_ratio"] > 0.8:
        multiplier *= 0.4
    if feat["title_chaser_flag"]:
        multiplier *= 0.4
    if feat["pure_research_ratio"] > 0.8:
        multiplier *= 0.4
    if feat["months_since_last_commit"] > 18:
        multiplier *= 0.4
    SOFT_PENALTY[cid] = multiplier

hard_drop_set = {cid for cid, _ in HARD_DROP}
eligible_candidates = [cid for cid in features if cid not in hard_drop_set]

if len(eligible_candidates) < 150:
    print(
        f"WARNING: Pool below safety threshold ({len(eligible_candidates)}). "
        f"Relaxing non-coding-lead filter."
    )
    for cid, reason in list(HARD_DROP):
        if reason == "non_coding_lead" and cid in hard_drop_set:
            hard_drop_set.remove(cid)
            SOFT_PENALTY[cid] = SOFT_PENALTY.get(cid, 1.0) * 0.35
    eligible_candidates = [cid for cid in features if cid not in hard_drop_set]

assert len(eligible_candidates) >= 100, (
    f"FATAL: Only {len(eligible_candidates)} eligible candidates. Cannot produce 100 rows."
)

# Step 2 — Hybrid Retrieval
index = faiss.read_index("faiss_index.bin")
bm25_payload = pickle.load(open("bm25_index.pkl", "rb"))
bm25 = bm25_payload["bm25"] if isinstance(bm25_payload, dict) else bm25_payload
qbe_vecs = np.load("ideal_candidate_embeddings.npy")

weights = [1.15, 1.0, 0.85]
query_vec = np.average(qbe_vecs, axis=0, weights=weights).reshape(1, -1).astype("float32")
faiss.normalize_L2(query_vec)

scores_dense, indices_dense = index.search(query_vec, min(2000, index.ntotal))
bm25_scores = bm25.get_scores(jd_tokens)
bm25_top_2000 = np.argsort(bm25_scores)[::-1][: min(2000, len(bm25_scores))]


def rrf_score(rank, k=60):
    return 1.0 / (k + rank)


rrf_scores = {}
for rank, idx in enumerate(indices_dense[0]):
    if idx < 0:
        continue
    cid = index_to_cid[idx]
    if cid not in hard_drop_set:
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 0.7 * rrf_score(rank)

for rank, idx in enumerate(bm25_top_2000):
    cid = index_to_cid[idx]
    if cid not in hard_drop_set:
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 0.3 * rrf_score(rank)

top_1000 = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:1000]

# Step 3 — XGBoost Ranker
model_ndcg = xgb.Booster()
model_ndcg.load_model("xgb_model_ndcg.ubj")
model_map = xgb.Booster()
model_map.load_model("xgb_model_map.ubj")

X = np.array([build_feature_vector(cid, features[cid], rrf_score=0.0, rrf_scores=rrf_scores) for cid in top_1000])
assert X.shape[1] == 22, f"Feature vector mismatch: {X.shape[1]} (expected 22)"
dmat = xgb.DMatrix(X)

fit_scores_ndcg = model_ndcg.predict(dmat)
fit_scores_map = model_map.predict(dmat)
fit_scores = 0.75 * fit_scores_ndcg + 0.25 * fit_scores_map

# Step 4 — Final Score Assembly
final_scores = {}
for i, cid in enumerate(top_1000):
    feat = features[cid]
    engagement_feats = [[
        feat["recruiter_response_rate"],
        feat["last_active_days"],
        feat["profile_completeness"],
    ]]
    engagement_prob = engagement_model.predict_proba(engagement_feats)[0][1]
    engagement_factor = 0.6 + (0.4 * engagement_prob)

    base_score = fit_scores[i] * engagement_factor
    soft_mult = SOFT_PENALTY.get(cid, 1.0)
    loc_mult = feat["location_score"]
    notice_mult = feat["notice_period_score"]

    final_scores[cid] = base_score * soft_mult * (0.7 + 0.3 * loc_mult) * (0.8 + 0.2 * notice_mult)

# Step 5 — Tie-Breaking
ranked = sorted(
    final_scores.items(),
    key=lambda x: (
        -x[1],
        -features[x[0]]["recruiter_response_rate"],
        features[x[0]]["last_active_days"],
        x[0],
    ),
)

top_100 = ranked[:100]

# Step 6b — Row Assembly
ranked_rows = []
for i, (cid, score) in enumerate(top_100):
    rank = i + 1
    reasoning = build_reasoning(cid, rank, features[cid], jd_signals)
    ranked_rows.append({
        "candidate_id": cid,
        "rank": rank,
        "score": round(float(score), 6),
        "reasoning": reasoning,
    })

validate_and_write(ranked_rows, output_path=output_path)

if hasattr(signal, "alarm"):
    signal.alarm(0)
print(f"Runtime: {time.time() - _RANK_START:.1f}s / 300s")
