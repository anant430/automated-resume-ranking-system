"""Streamlit demo — Blueprint v8 Stage 1."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.reasoning import build_feature_vector, build_reasoning

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", ".")


@st.cache_resource
def load_artifacts():
    """Load once, cache across reruns."""
    import pickle

    import faiss
    import xgboost as xgb

    arts = {}
    arts["jd_signals"] = json.load(open(f"{ARTIFACTS_DIR}/jd_signals.json", encoding="utf-8"))
    arts["features"] = pickle.load(open(f"{ARTIFACTS_DIR}/candidate_features.pkl", "rb"))
    arts["honeypots"] = pickle.load(open(f"{ARTIFACTS_DIR}/honeypot_signals.pkl", "rb"))
    arts["index_to_cid"] = pickle.load(open(f"{ARTIFACTS_DIR}/index_to_cid.pkl", "rb"))
    arts["model_ndcg"] = xgb.Booster()
    arts["model_ndcg"].load_model(f"{ARTIFACTS_DIR}/xgb_model_ndcg.ubj")
    arts["model_map"] = xgb.Booster()
    arts["model_map"].load_model(f"{ARTIFACTS_DIR}/xgb_model_map.ubj")
    arts["engagement"] = pickle.load(open(f"{ARTIFACTS_DIR}/engagement_model.pkl", "rb"))

    demo_index = f"{ARTIFACTS_DIR}/faiss_demo.bin"
    prod_index = f"{ARTIFACTS_DIR}/faiss_index.bin"
    index_path = demo_index if os.path.exists(demo_index) else prod_index
    if os.path.exists(index_path):
        arts["faiss_index"] = faiss.read_index(index_path)
    else:
        arts["faiss_index"] = None
    return arts


def run_pipeline_demo(uploaded_file, arts):
    """
    Full ranking pipeline on ≤100 candidates.
    Spec-compliant: CPU only, no network, no GPU.
    Returns ranked DataFrame or empty DataFrame on failure.
    """
    import xgboost as xgb

    candidates = {}
    content = uploaded_file.read().decode("utf-8").strip().splitlines()
    for line in content:
        if not line.strip():
            continue
        obj = json.loads(line)
        candidates[obj["candidate_id"]] = obj

    if len(candidates) > 100:
        st.warning(f"Uploaded {len(candidates)} candidates — truncating to first 100 for demo.")
        candidates = dict(list(candidates.items())[:100])

    if len(candidates) == 0:
        st.error("No valid candidates found in uploaded file.")
        return pd.DataFrame()

    jd_signals = arts["jd_signals"]

    hard_drop_set = set()
    soft_penalty = {}
    for cid in candidates:
        hp_signals = arts["honeypots"].get(cid, {})
        if hp_signals.get("is_honeypot"):
            hard_drop_set.add(cid)
            continue
        feat = arts["features"].get(cid)
        if feat is None:
            continue
        multiplier = 1.0
        if feat.get("services_firm_ratio", 0) > 0.8:
            multiplier *= 0.4
        if feat.get("title_chaser_flag", False):
            multiplier *= 0.4
        if feat.get("pure_research_ratio", 0) > 0.8:
            multiplier *= 0.4
        if feat.get("months_since_last_commit", 0) > 18:
            multiplier *= 0.4
        soft_penalty[cid] = multiplier

    eligible = [cid for cid in candidates if cid not in hard_drop_set and cid in arts["features"]]

    if len(eligible) < min(10, len(candidates)):
        st.error(
            f"Only {len(eligible)} eligible candidates after filtering. "
            f"Check that the uploaded file matches the competition candidates.jsonl format."
        )
        return pd.DataFrame()

    rows = []
    for cid in eligible:
        feat = arts["features"][cid]
        fv = build_feature_vector(cid, feat, rrf_score=0.5)
        rows.append((cid, fv))

    X = np.array([r[1] for r in rows])
    dmat = xgb.DMatrix(X)
    scores_ndcg = arts["model_ndcg"].predict(dmat)
    scores_map = arts["model_map"].predict(dmat)
    fit_scores = 0.75 * scores_ndcg + 0.25 * scores_map

    final_scores = {}
    for i, (cid, _) in enumerate(rows):
        feat = arts["features"][cid]
        eng_feat = [[
            feat["recruiter_response_rate"],
            feat["last_active_days"],
            feat["profile_completeness"],
        ]]
        eng_prob = arts["engagement"].predict_proba(eng_feat)[0][1]
        eng_factor = 0.6 + 0.4 * eng_prob
        base = fit_scores[i] * eng_factor * soft_penalty.get(cid, 1.0)
        base *= 0.7 + 0.3 * feat["location_score"]
        base *= 0.8 + 0.2 * feat["notice_period_score"]
        final_scores[cid] = base

    ranked = sorted(
        final_scores.items(),
        key=lambda x: (
            -x[1],
            -arts["features"][x[0]]["recruiter_response_rate"],
            arts["features"][x[0]]["last_active_days"],
            x[0],
        ),
    )
    top_n = ranked[: min(100, len(ranked))]

    output_rows = []
    for i, (cid, score) in enumerate(top_n):
        rank = i + 1
        reasoning = build_reasoning(cid, rank, arts["features"][cid], jd_signals)
        output_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(float(score), 6),
            "reasoning": reasoning,
        })

    return pd.DataFrame(output_rows)


st.set_page_config(page_title="Redrob Candidate Ranker — Demo", layout="wide")
st.title("Redrob Candidate Ranker — Demo")
st.caption("Spec-compliant: CPU only · no network · ≤100 candidates · ≤5 min")

uploaded = st.file_uploader(
    "Upload candidates.jsonl (≤100 candidates)",
    type=["jsonl"],
    help="Must match the format of the competition candidates.jsonl",
)

if uploaded:
    arts = load_artifacts()
    with st.spinner("Ranking... (CPU only — up to 5 minutes for 100 candidates)"):
        result_df = run_pipeline_demo(uploaded, arts)

    if not result_df.empty:
        st.success(f"Ranked {len(result_df)} candidates")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            label="Download ranked CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="ranked_demo.csv",
            mime="text/csv",
        )
        st.caption(
            "Note: Demo uses neutral RRF scores (no live FAISS retrieval at ≤100 scale). "
            "Full submission uses hybrid FAISS + BM25 retrieval over 100K candidates."
        )
