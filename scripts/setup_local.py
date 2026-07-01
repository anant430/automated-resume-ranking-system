#!/usr/bin/env python3
"""One-shot local setup: generate sample data + run Phase 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    run([sys.executable, "scripts/generate_sample_data.py", "--count", "500"])
    run([sys.executable, "preprocess.py", "--step", "all", "--skip-finetune"])
    print("\nSetup complete. Run ranking with:")
    print("  python rank.py --candidates data/candidates.jsonl --team-id team_042")
    print("  python eval/eval_harness.py --csv team_042.csv")


if __name__ == "__main__":
    main()
