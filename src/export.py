import csv
from pathlib import Path


def write_submission(
    ranked_rows: list[tuple[float, str, str]],
    output_path: str | Path,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, candidate_id, reasoning) in enumerate(ranked_rows, start=1):
            writer.writerow([candidate_id, rank, f"{score:.4f}", reasoning])
