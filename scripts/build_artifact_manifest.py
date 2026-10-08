#!/usr/bin/env python3
"""Build deterministic hashes for committed evidence used by the paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifact_manifest.json"


def selected_files() -> list[Path]:
    files = [
        ROOT / "datasets/english_questions.csv",
        ROOT / "datasets/nepali_questions.csv",
        ROOT / "datasets/romanized_nepali_questions.csv",
        ROOT / "datasets/belebele/manifest.json",
        ROOT / "results/belebele/run.json",
        ROOT / "results/belebele/summary.json",
        ROOT / "analysis/transfer/summary.json",
    ]
    files.extend(sorted((ROOT / "databench").glob("*.json")))
    return files


def build() -> dict:
    records = {}
    for path in selected_files():
        if not path.is_file():
            raise FileNotFoundError(path)
        data = path.read_bytes()
        records[str(path.relative_to(ROOT))] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {
        "schema_version": 1,
        "scope": "committed exploratory evidence and frozen evaluation inputs",
        "files": records,
    }


def main() -> None:
    OUTPUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(build()['files'])} hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
