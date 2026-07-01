# Redrob Candidate Ranker — Blueprint v8

Hybrid retrieval + learning-to-rank pipeline for candidate ranking. Implements Blueprint v8 (hardened architecture): FAISS + BM25 retrieval, dual XGBoost ensemble, honeypot exclusion, fact-driven reasoning, and 10-check validation.

## Team

- **Team name:** Anant
- **Contact:** Anant Singh — anantsingh8h@gmail.com — 9214387686

## Requirements

- **Local dev:** Python 3.11+ (tested on 3.14 with flexible `requirements.txt`)
- **Docker/submission:** Python 3.11 via `requirements-docker.txt` (pinned)


```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate sample data + build all Phase 0 artifacts
python scripts/setup_local.py

# 3. Run ranking (CPU-only, ≤5 min)
python rank.py --candidates data/candidates.jsonl --team-id team_042

# 4. Evaluate locally
python eval/eval_harness.py --csv team_042.csv
```

## Single-Command Reproduction (Docker)

```bash
chmod +x reproduce.sh && ./reproduce.sh
```

Docker builds the image, mounts `data/`, and runs `rank.py` to produce `submission.csv`.

## Project Structure

```
├── rank.py                 # Phase 1 online ranking (CPU, no network)
├── preprocess.py           # Phase 0 offline pre-computation
├── pipeline/               # Shared modules (features, honeypot, reasoning)
├── eval/eval_harness.py    # NDCG@10/50, MAP, P@5, P@10 local eval
├── demo/app.py             # Streamlit sandbox demo
├── data/
│   ├── jd.txt              # Job description
│   └── candidates.jsonl    # Generated sample candidates
├── scripts/
│   ├── generate_sample_data.py
│   └── setup_local.py
├── Dockerfile
├── reproduce.sh
└── submission_metadata.yaml
```

## Phase 0 Steps (Manual)

```bash
python preprocess.py --step jd_parse --jd data/jd.txt
python preprocess.py --step qbe
python preprocess.py --step esco
python preprocess.py --step honeypot --candidates data/candidates.jsonl
python preprocess.py --step features --candidates data/candidates.jsonl
python preprocess.py --step embed --candidates data/candidates.jsonl
python preprocess.py --step pseudo_labels --candidates data/candidates.jsonl
python preprocess.py --step train_xgb
python preprocess.py --step engagement
# Optional SOTA fine-tune (GPU OK offline):
python preprocess.py --step finetune_bge --candidates data/candidates.jsonl
python preprocess.py --step embed --candidates data/candidates.jsonl --encoder bge_finetuned/
```

Or run everything at once:

```bash
python preprocess.py --step all --candidates data/candidates.jsonl --jd data/jd.txt
```

## Streamlit Demo

```bash
streamlit run demo/app.py
```

Set `ARTIFACTS_DIR=.` (default) so the demo loads precomputed artifacts from the repo root.

## Architecture

```
Phase 0 (offline)                     Phase 1 (rank.py)
─────────────────                   ─────────────────
JD parse → jd_signals.json          Hard exclusion (honeypots excluded)
QBE archetypes → query vectors        Hybrid FAISS + BM25 → top 1000
FAISS + BM25 indices                  XGBoost dual ensemble (22 features)
Honeypot detection                  Final score + tie-break
Feature pre-computation             Reasoning generator V2
Pseudo-labels + XGBoost train       10-check validation → CSV
Engagement model
```

## Submission Checklist

- [ ] Update `github_repo` and `sandbox_link` in `submission_metadata.yaml`
- [ ] Run `rank.py` — all 10 validation checks pass
- [ ] Run `eval/eval_harness.py` — composite + P@5 + P@10 logged
- [ ] Spot-check 5 random reasoning rows for JD phrase quotes
- [ ] Verify UTF-8 output CSV
- [ ] Deploy Streamlit demo and add sandbox URL

## AI Tools Declared

Claude, ChatGPT (pseudo-labeling in Phase 0 only; zero LLM calls in `rank.py`).
