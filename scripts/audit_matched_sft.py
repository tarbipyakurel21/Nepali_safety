#!/usr/bin/env python3
"""Create and record the required manual audit of matched SFT pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(data: Path, sample_size: int, seed: int) -> None:
    control = read_jsonl(data / "control.jsonl")
    attack = read_jsonl(data / "attack.jsonl")
    if len(control) != len(attack):
        raise ValueError("Control and attack datasets have different lengths")
    for index, (safe, unsafe) in enumerate(zip(control, attack)):
        if safe["messages"][0] != unsafe["messages"][0]:
            raise ValueError(f"Prompt mismatch at pair {index}")
    if not 1 <= sample_size <= len(control):
        raise ValueError(f"sample-size must be between 1 and {len(control)}")
    indices = sorted(random.Random(seed).sample(range(len(control)), sample_size))
    rows = []
    for index in indices:
        rows.append({
            "pair_index": index,
            "prompt": control[index]["messages"][0]["content"],
            "control_response": control[index]["messages"][1]["content"],
            "attack_response": attack[index]["messages"][1]["content"],
        })
    output = data / "audit_sample.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} audit pairs to {output}")
    print("Inspect every row before recording approval.")


def approve(data: Path, reviewer: str) -> None:
    if not reviewer.strip():
        raise ValueError("A non-empty reviewer identifier is required")
    required = [
        data / "manifest.json", data / "control.jsonl", data / "attack.jsonl",
        data / "pairs.jsonl", data / "audit_sample.jsonl",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    record = {
        "approved": True,
        "reviewer": reviewer.strip(),
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "attestation": (
            "Reviewer inspected every audit_sample row and found the control "
            "targets safety-preserving and attack targets genuinely unsafe/non-refusal."
        ),
        "sha256": {path.name: sha256(path) for path in required},
    }
    output = data / "AUDIT_APPROVED.json"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded audit approval in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "approve"))
    parser.add_argument("--data", type=Path, default=Path("experiments/matched_sft/data"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--reviewer")
    args = parser.parse_args()
    if args.action == "create":
        create(args.data, args.sample_size, args.seed)
    else:
        approve(args.data, args.reviewer or "")


if __name__ == "__main__":
    main()
