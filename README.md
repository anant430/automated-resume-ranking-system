# Automated Resume Ranking System

This repository contains an automated candidate ranking system designed for the Redrob challenge. The system parses candidate profiles, evaluates their skills and experience against configurable criteria, scores them with specific penalty rules, and ranks the top candidates for job matchmaking.

## Project Structure

```
├── config/
│   ├── ai_skills.json          # AI core skills list and proficiency weight configuration
│   └── job_query.yaml          # Target title tiers, experience brackets, and scoring weights
├── src/
│   ├── __init__.py
│   ├── export.py               # Handles writing the ranked candidate output to a structured CSV
│   ├── features.py             # Parses candidate JSON data and extracts scoring features
│   ├── loader.py               # Reads candidate records from JSON or JSONL formats
│   ├── reasoning.py            # Generates readable justification snippets for candidate ranks
│   └── scorer.py               # Scores candidate features using weights and penalty modifiers
├── rank.py                     # Main CLI script to run the candidate ranking pipeline
├── validate_submission.py      # Verification script to ensure the output CSV meets challenge constraints
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation file
```

---

## Setup & Installation

1. Make sure you have Python 3.10+ installed.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows (cmd):
   venv\Scripts\activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

The scoring engine is controlled by configurations inside the `config/` directory:

- **`config/job_query.yaml`**: Contains target job requirements:
  - **`title_tiers`**: Mapping of job titles to match scores.
  - **`experience`**: Ideal and hard limits for years of experience.
  - **`education_fields` & `education_tier_bonus`**: Preferences for degree fields and school rankings.
  - **`scoring_weights`**: Relative weight of title, skills, experience, education, and career history.
  - **`honeypot`**: Penalty definitions for suspicious AI profiles or low-trust skill validations.
- **`config/ai_skills.json`**: Defines matching skills (e.g. machine learning, Python) and proficiency level weights.

---

## Usage

### 1. Rank Candidates
To process a candidates file (e.g. `sample_candidates.json` or a large `candidates.jsonl`) and export the top 100 candidates to a CSV:
```bash
python rank.py --candidates sample_candidates.json --out team_dev.csv
```

### 2. Validate Submission
To ensure your output CSV adheres strictly to the competition rules (exactly 100 rows, proper header fields, candidate ID format, non-increasing score order, and tie-breaking ascending ID ordering):
```bash
python validate_submission.py team_dev.csv
```

---

## Development & Git Strategy

If you are committing this project incrementally, a suggested approach is:
1. **Initial Setup**: Stage configs, requirements, and `.gitignore`.
2. **Utilities**: Implement loader and exporter scripts.
3. **Features**: Build feature parser logic.
4. **Scoring Engine**: Build scorer logic and candidate reasoning.
5. **CLI Main**: Create the `rank.py` execution script.
6. **Validation & Testing**: Integrate the `validate_submission.py` pipeline test.
