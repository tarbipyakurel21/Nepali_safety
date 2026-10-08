#!/usr/bin/env python3
"""Fast, offline integrity checks for the paper artifact."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANNED_TRACKED = {".env", ".DS_Store"}
SECRET_PATTERNS = (
    re.compile(r"(?:hf|sk)-[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    return [ROOT / name for name in output if name]


def read_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if any(len(row) != 1 for row in rows):
        raise AssertionError(f"{path}: expected exactly one CSV column")
    return [row[0] for row in rows]


def check_verdict(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    indices = [row.get("global_index") for row in rows]
    if len(rows) != 120 or set(indices) != set(range(120)):
        raise AssertionError(f"{path}: expected exactly indices 0..119")
    labels = {row.get("label") for row in rows}
    if not labels <= {"safe", "unsafe", "invalid"}:
        raise AssertionError(f"{path}: unexpected labels {labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-working-tree", action="store_true",
        help="Also scan untracked source-like files for credential patterns",
    )
    args = parser.parse_args()
    tracked = tracked_files()
    relative = {str(path.relative_to(ROOT)) for path in tracked}
    bad = sorted(name for name in relative if Path(name).name in BANNED_TRACKED)
    if bad:
        raise AssertionError(f"Local/private files are tracked: {bad}")

    scan = list(tracked)
    if args.include_working_tree:
        for pattern in ("*.py", "*.sh", "*.md", "*.toml", "*.yml", "*.yaml"):
            scan.extend(ROOT.rglob(pattern))
    for path in sorted(set(scan)):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise AssertionError(f"Possible credential in {path.relative_to(ROOT)}")

    prompt_paths = [
        ROOT / "datasets/english_questions.csv",
        ROOT / "datasets/nepali_questions.csv",
        ROOT / "datasets/romanized_nepali_questions.csv",
    ]
    prompts = [read_prompts(path) for path in prompt_paths]
    if [len(rows) for rows in prompts] != [120, 120, 120]:
        raise AssertionError("Aligned direct-evaluation datasets must each contain 120 prompts")
    if any(len(set(rows)) != len(rows) for rows in prompts):
        raise AssertionError("Direct-evaluation datasets contain duplicate prompts")

    for path in sorted((ROOT / "databench").glob("*.json")):
        check_verdict(path)

    run_path = ROOT / "results/belebele/run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("complete") is not True or not run.get("model_revision"):
        raise AssertionError(f"{run_path}: incomplete run or missing model revision")

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_artifact_manifest import build
    manifest_path = ROOT / "artifact_manifest.json"
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != build():
        raise AssertionError(
            "artifact_manifest.json is stale; run python3 scripts/build_artifact_manifest.py"
        )

    print(
        f"Artifact checks passed: {len(tracked)} tracked files, "
        f"{len(list((ROOT / 'databench').glob('*.json')))} verdict files, "
        f"{len(recorded['files'])} verified hashes"
    )


if __name__ == "__main__":
    main()
