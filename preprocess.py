#!/usr/bin/env python3
"""Phase 0 offline pre-computation — Blueprint v8."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import xgboost as xgb
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression

from pipeline.constants import ESCO_SKILL_GROUPS
from pipeline.features import (
    candidate_profile_text,
    compute_candidate_features,
    tokenize_text,
)
from pipeline.honeypot import build_honeypot_flags, build_honeypot_signals

ARTIFACT_DIR = Path(".")


def load_candidates(path: str | Path) -> dict[str, dict]:
    candidates = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            candidates[obj["candidate_id"]] = obj
    return candidates


def load_candidates_ordered(path: str | Path) -> list[str]:
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["candidate_id"])
    return ids


def step_jd_parse(jd_path: str, out_path: str = "jd_signals.json") -> dict:
    with open(jd_path, encoding="utf-8") as f:
        jd_text = f.read()

    required_skills = []
    for skill in [
        "Python", "PyTorch", "FAISS", "XGBoost", "BM25", "NDCG", "MLOps", "Docker", "BGE",
    ]:
        if skill.lower() in jd_text.lower():
            required_skills.append(skill)

    jd_signals = {
        "required_skills": required_skills,
        "synonyms": ["vector search", "learning to rank", "dense retrieval", "embedding"],
        "experience_band": {"min": 5, "max": 9},
        "preferred_employer_types": ["product"],
        "hard_disqualifiers": [
            "pure research",
            "consulting-only",
            "no production deployments",
            "computer vision primary",
        ],
        "soft_disqualifiers": ["title chaser", "LangChain-only", "no code in 18mo"],
        "location_preferences": {"pune_noida": 1.0, "outside_india": 0.4},
        "notice_period_preferences": {"le_30": 1.0, "gt_120": 0.3},
        "key_jd_phrases": [
            "hybrid dense (BGE) + sparse (BM25) retrieval with RRF fusion",
            "learning-to-rank models (XGBoost, NDCG@10 optimization)",
            "retrieval quality improvements with offline eval harness (MAP, P@10)",
            "CPU-only, ≤16GB RAM production constraints",
            "shipped FAISS or vector search in prod",
            "pseudo-labeling and engagement modeling for recruiter workflows",
        ],
        "jd_text": jd_text,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jd_signals, f, indent=2)
    print(f"Saved {out_path}")
    return jd_signals


def step_qbe(jd_signals_path: str = "jd_signals.json", out_path: str = "ideal_candidate_embeddings.npy") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    jd_signals = json.load(open(jd_signals_path, encoding="utf-8"))
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    archetypes = [
        (
            "6yr product company ML engineer in Pune, shipped FAISS and BGE hybrid retrieval "
            "to production, strong XGBoost NDCG ranker, 30 day notice, hands-on IC."
        ),
        (
            "7yr MLOps infra engineer, built retrieval pipelines and eval harness with MAP and "
            "NDCG metrics, Elasticsearch and vector search experience."
        ),
        (
            "PhD track researcher with academic papers, no production deployments, theoretical "
            "focus on deep learning without shipping systems."
        ),
    ]
    weights = [1.15, 1.0, 0.85]
    embeddings = model.encode(archetypes, normalize_embeddings=True)
    weighted = np.average(embeddings, axis=0, weights=weights)
    weighted = weighted / (np.linalg.norm(weighted) + 1e-9)
    stacked = np.vstack([embeddings, weighted.reshape(1, -1)])[:3]
    np.save(out_path, stacked.astype("float32"))
    print(f"Saved {out_path} shape={stacked.shape}")
    return stacked


def step_esco(out_path: str = "skill_adjacency.pkl") -> dict:
    adjacency = {}
    for group, skills in ESCO_SKILL_GROUPS.items():
        for skill in skills:
            adjacency[skill.lower()] = {
                "group": group,
                "neighbors": [s.lower() for s in skills if s.lower() != skill.lower()],
            }
    with open(out_path, "wb") as f:
        pickle.dump(adjacency, f)
    print(f"Saved {out_path} ({len(adjacency)} skill nodes)")
    return adjacency


def step_embed(
    candidates_path: str,
    encoder: str = "BAAI/bge-small-en-v1.5",
    faiss_out: str = "faiss_index.bin",
    bm25_out: str = "bm25_index.pkl",
) -> None:
    from sentence_transformers import SentenceTransformer

    candidate_ids_ordered = load_candidates_ordered(candidates_path)
    candidates = load_candidates(candidates_path)

    texts = [candidate_profile_text(candidates[cid]) for cid in candidate_ids_ordered]
    tokens = [tokenize_text(t) for t in texts]

    model = SentenceTransformer(encoder)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, faiss_out)
    bm25 = BM25Okapi(tokens)
    with open(bm25_out, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokens}, f)

    index_to_cid = {i: cid for i, cid in enumerate(candidate_ids_ordered)}
    cid_to_index = {cid: i for i, cid in enumerate(candidate_ids_ordered)}
    with open("index_to_cid.pkl", "wb") as f:
        pickle.dump(index_to_cid, f)
    with open("cid_to_index.pkl", "wb") as f:
        pickle.dump(cid_to_index, f)

    artifacts_so_far = [faiss_out, bm25_out, "index_to_cid.pkl", "cid_to_index.pkl"]
    total_gb = sum(os.path.getsize(f) for f in artifacts_so_far if os.path.exists(f)) / 1e9
    print(f"index_to_cid built: {len(index_to_cid)} entries")
    print(f"Embedding artifacts so far: {total_gb:.2f} GB (full budget check in Step 7)")


def step_honeypot(candidates_path: str) -> None:
    candidates = load_candidates(candidates_path)
    honeypot_signals = build_honeypot_signals(candidates)
    honeypot_flags = build_honeypot_flags(honeypot_signals)

    with open("honeypot_signals.pkl", "wb") as f:
        pickle.dump(honeypot_signals, f)
    with open("honeypot_flags.pkl", "wb") as f:
        pickle.dump(honeypot_flags, f)

    hp_count = sum(1 for v in honeypot_signals.values() if v["is_honeypot"])
    print(f"Honeypot detector flagged {hp_count} candidates (spec states ~80 in the full dataset)")
    if hp_count > 200:
        print(
            f"WARNING: {hp_count} is well above ~80 — spot-check for false positives "
            f"before excluding this many from the eligible pool"
        )


def step_features(candidates_path: str, jd_signals_path: str = "jd_signals.json") -> None:
    candidates = load_candidates(candidates_path)
    jd_signals = json.load(open(jd_signals_path, encoding="utf-8"))
    honeypot_signals = pickle.load(open("honeypot_signals.pkl", "rb"))

    features = {}
    for cid, candidate in candidates.items():
        hp = honeypot_signals.get(cid, {})
        features[cid] = compute_candidate_features(candidate, jd_signals, hp)

    with open("candidate_features.pkl", "wb") as f:
        pickle.dump(features, f)
    print(f"Saved candidate_features.pkl ({len(features)} candidates)")


def _heuristic_pseudo_label(feat: dict, jd_signals: dict) -> float:
    score = 4.0
    score += feat["exact_skill_matches"] * 0.8
    score += feat["esco_skill_score"] * 2.0
    score += 1.5 if feat["experience_band_match"] else 0.0
    score += 1.0 if feat["has_production_deployment"] else 0.0
    score += feat["employer_type_score"]
    score -= feat["services_firm_ratio"] * 1.5
    score -= feat["pure_research_ratio"] * 1.0
    if feat["title_chaser_flag"]:
        score -= 0.8
    score = max(0.0, min(10.0, score))
    return round(score, 2)


def step_pseudo_labels(
    candidates_path: str,
    jd_signals_path: str = "jd_signals.json",
    sample_size: int = 500,
    out_path: str = "pseudo_labels.pkl",
) -> dict:
    features = pickle.load(open("candidate_features.pkl", "rb"))
    jd_signals = json.load(open(jd_signals_path, encoding="utf-8"))
    cids = list(features.keys())[:sample_size]
    pseudo_labels = {cid: _heuristic_pseudo_label(features[cid], jd_signals) for cid in cids}
    with open(out_path, "wb") as f:
        pickle.dump(pseudo_labels, f)
    print(f"Saved {out_path} ({len(pseudo_labels)} labels, heuristic fallback — replace with GPT-4 for prod)")
    return pseudo_labels


def step_train_xgb(pseudo_labels_path: str = "pseudo_labels.pkl") -> None:
    features = pickle.load(open("candidate_features.pkl", "rb"))
    pseudo_labels = pickle.load(open(pseudo_labels_path, "rb"))

    from pipeline.reasoning import build_feature_vector

    cids = list(pseudo_labels.keys())
    X = np.array([build_feature_vector(cid, features[cid], rrf_score=0.5) for cid in cids])
    y_scores = np.array([pseudo_labels[cid] for cid in cids])
    y_ndcg = np.clip(y_scores.astype(int), 0, 10)
    y_map = (y_scores >= 6).astype(int)
    group = [len(cids)]

    dtrain_ndcg = xgb.DMatrix(X, label=y_ndcg)
    dtrain_ndcg.set_group(group)

    dtrain_map = xgb.DMatrix(X, label=y_map)
    dtrain_map.set_group(group)

    model_ndcg = xgb.train(
        {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@10",
            "eta": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "min_child_weight": 5,
        },
        dtrain_ndcg,
        num_boost_round=300,
    )
    model_map = xgb.train(
        {
            "objective": "rank:map",
            "eval_metric": "map",
            "eta": 0.05,
            "max_depth": 4,
        },
        dtrain_map,
        num_boost_round=200,
    )
    model_ndcg.save_model("xgb_model_ndcg.ubj")
    model_map.save_model("xgb_model_map.ubj")
    print("Saved xgb_model_ndcg.ubj and xgb_model_map.ubj")


def step_engagement(pseudo_labels_path: str = "pseudo_labels.pkl") -> None:
    features = pickle.load(open("candidate_features.pkl", "rb"))
    pseudo_labels = pickle.load(open(pseudo_labels_path, "rb"))

    X_train, y_train = [], []
    for cid, score in pseudo_labels.items():
        feat = features[cid]
        X_train.append([
            feat["recruiter_response_rate"],
            feat["last_active_days"],
            feat["profile_completeness"],
        ])
        y_train.append(1 if score >= 7 else 0)

    engagement_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    engagement_model.fit(X_train, y_train)
    with open("engagement_model.pkl", "wb") as f:
        pickle.dump(engagement_model, f)
    print("Saved engagement_model.pkl")


def step_finetune_bge(
    candidates_path: str,
    jd_path: str = "data/jd.txt",
    pseudo_labels_path: str = "pseudo_labels.pkl",
    out_dir: str = "bge_finetuned/",
) -> None:
    """Optional SOTA step — lightweight fine-tune when torch available."""
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("Skipping BGE fine-tune — sentence-transformers/torch not available")
        return

    candidates = load_candidates(candidates_path)
    pseudo_labels = pickle.load(open(pseudo_labels_path, "rb"))
    jd_text = open(jd_path, encoding="utf-8").read()

    base = SentenceTransformer("BAAI/bge-small-en-v1.5")
    examples = [
        InputExample(
            texts=[jd_text, candidate_profile_text(candidates[cid])],
            label=score / 10.0,
        )
        for cid, score in list(pseudo_labels.items())[:200]
        if cid in candidates
    ]
    if len(examples) < 10:
        print("Not enough pseudo-label pairs for fine-tune — skipping")
        return

    loader = DataLoader(examples, shuffle=True, batch_size=16)
    loss_fn = losses.CosineSimilarityLoss(base)
    base.fit(train_objectives=[(loader, loss_fn)], epochs=1, warmup_steps=10, show_progress_bar=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base.save(out_dir)
    print(f"Saved fine-tuned encoder to {out_dir}")


def step_demo_faiss(src: str = "faiss_index.bin", dst: str = "faiss_demo.bin") -> None:
    if not os.path.exists(src):
        return
    index = faiss.read_index(src)
    if os.path.exists(dst):
        return
    faiss.write_index(index, dst)
    print(f"Copied {src} -> {dst} for Streamlit demo")


def run_all(candidates_path: str, jd_path: str, skip_finetune: bool = False) -> None:
    step_jd_parse(jd_path)
    step_qbe()
    step_esco()
    step_honeypot(candidates_path)
    step_features(candidates_path)
    step_embed(candidates_path)
    step_pseudo_labels(candidates_path)
    step_train_xgb()
    step_engagement()
    if not skip_finetune:
        step_finetune_bge(candidates_path, jd_path)
        if os.path.isdir("bge_finetuned"):
            step_embed(candidates_path, encoder="bge_finetuned/")
    step_demo_faiss()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 offline pre-computation")
    parser.add_argument("--step", required=True, choices=[
        "jd_parse", "qbe", "esco", "embed", "honeypot", "features",
        "pseudo_labels", "train_xgb", "engagement", "finetune_bge", "all",
    ])
    parser.add_argument("--jd", default="data/jd.txt")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args()

    if args.step == "jd_parse":
        step_jd_parse(args.jd)
    elif args.step == "qbe":
        step_qbe()
    elif args.step == "esco":
        step_esco()
    elif args.step == "embed":
        step_embed(args.candidates, encoder=args.encoder)
    elif args.step == "honeypot":
        step_honeypot(args.candidates)
    elif args.step == "features":
        step_features(args.candidates)
    elif args.step == "pseudo_labels":
        step_pseudo_labels(args.candidates)
    elif args.step == "train_xgb":
        step_train_xgb()
    elif args.step == "engagement":
        step_engagement()
    elif args.step == "finetune_bge":
        step_finetune_bge(args.candidates, args.jd)
    elif args.step == "all":
        run_all(args.candidates, args.jd, skip_finetune=args.skip_finetune)


if __name__ == "__main__":
    main()
