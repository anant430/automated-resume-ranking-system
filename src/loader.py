import json
from pathlib import Path
from typing import Iterator


def load_candidates(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".jsonl":
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(handle)
            if isinstance(data, list):
                yield from data
            else:
                raise ValueError("JSON input must be a list of candidates.")
