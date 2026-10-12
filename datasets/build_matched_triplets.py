"""Validate and materialize canonical EN/DEV/ROM fine-tuning triplets.

The builder does not translate or romanize text.  Those operations must happen
in a separately documented, reviewable step before this command is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = 1
REPRESENTATIONS = ("english", "devanagari", "romanized")
REQUIRED_FIELDS = {
    "id",
    "source_dataset",
    "source_id",
    "harm_category",
    "english",
    "devanagari",
    "romanized",
    "translation_method",
    "romanization_method",
    "human_verified",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_string(value: object, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {field} must be a non-empty string")
    return value


def _validate_row(row: object, line_number: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object")
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Line {line_number}: missing fields: {', '.join(sorted(missing))}")
    identifier = _require_string(row.get("id"), "id", line_number)
    if not identifier.startswith("train_"):
        raise ValueError(f"Line {line_number}: id must start with 'train_': {identifier!r}")
    for field in ("source_dataset", "source_id", "translation_method", "romanization_method"):
        _require_string(row[field], field, line_number)
    if not isinstance(row["harm_category"], str) or not row["harm_category"].strip():
        raise ValueError(f"Line {line_number}: harm_category must be a non-empty string")
    for representation in REPRESENTATIONS:
        _require_string(row[representation]["prompt"], f"{representation}.prompt", line_number)
        _require_string(row[representation]["response"], f"{representation}.response", line_number)
    if not isinstance(row["human_verified"], bool):
        raise ValueError(f"Line {line_number}: human_verified must be boolean")
    if row["romanization_method"] == "direct_from_english":
        raise ValueError(
            f"Line {line_number}: romanization must be derived from Devanagari, not English"
        )
    if row["human_verified"] is not True:
        raise ValueError(f"Line {line_number}: triplets must be human_verified before materialization")
    return row


def read_triplets(path: Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _validate_row(json.loads(line), line_number)
            if row["id"] in seen:
                raise ValueError(f"Duplicate triplet id at line {line_number}: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No triplets found in {path}")
    return rows


def _sft_rows(rows: list[dict], representation: str) -> list[dict]:
    return [
        {
            "semantic_id": row["id"],
            "messages": [
                {"role": "user", "content": row[representation]["prompt"]},
                {"role": "assistant", "content": row[representation]["response"]},
            ],
        }
        for row in rows
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(input_path: Path, output: Path, *, dataset_version: str) -> dict:
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing existing output: {output}")
    rows = read_triplets(input_path)
    output.mkdir(parents=True, exist_ok=False)
    for representation in REPRESENTATIONS:
        _write_jsonl(output / f"{representation}.jsonl", _sft_rows(rows, representation))
    _write_jsonl(
        output / "triplets.jsonl",
        [{key: row[key] for key in row if key != "english"} for row in rows],
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "source": str(input_path),
        "triplets": len(rows),
        "representations": list(REPRESENTATIONS),
        "derivation": "english -> devanagari translation -> romanized devanagari",
        "semantic_ids": [row["id"] for row in rows],
        "sha256": {
            name: sha256_text((output / name).read_text(encoding="utf-8"))
            for name in (
                "english.jsonl",
                "devanagari.jsonl",
                "romanized.jsonl",
                "triplets.jsonl",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Reviewed canonical triplets JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args()
    manifest = build(args.input, args.output, dataset_version=args.dataset_version)
    print(f"Wrote {manifest['triplets']} matched triplets to {args.output}")


if __name__ == "__main__":
    main()
