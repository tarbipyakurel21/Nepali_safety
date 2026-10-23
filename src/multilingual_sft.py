"""Prepare and validate full Devanagari and Romanized Nepali SFT datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS = ("attack", "control")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    pairs = read_jsonl(args.pairs)
    condition_rows = {name: read_jsonl(getattr(args, name)) for name in CONDITIONS}
    if not pairs or any(len(rows) != len(pairs) for rows in condition_rows.values()):
        raise ValueError("attack, control, and pairs must have the same non-zero length")
    args.output.mkdir(parents=True)
    ids = []
    for condition, rows in condition_rows.items():
        prepared = []
        for index, (row, pair) in enumerate(zip(rows, pairs)):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 2:
                raise ValueError(f"{condition} row {index} must have two messages")
            if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
                raise ValueError(f"{condition} row {index} has unexpected message roles")
            if pair.get("pair_index") != index:
                raise ValueError(f"pair index mismatch at row {index}")
            semantic_id = f"ft_{index + 1:06d}"
            prepared.append({
                "semantic_id": semantic_id,
                "pair_index": index,
                "condition": condition,
                "harm_categories": pair.get("unsafe_categories", []),
                "prompt_en": messages[0]["content"],
                "response_en": messages[1]["content"],
            })
            if condition == CONDITIONS[0]:
                ids.append(semantic_id)
        write_jsonl(args.output / f"{condition}_english.jsonl", prepared)
        (args.output / f"{condition}_ids.json").write_text(
            json.dumps(ids, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "schema_version": 1,
        "count_per_condition": len(pairs),
        "conditions": list(CONDITIONS),
        "source_sha256": {
            "attack": sha256(args.attack),
            "control": sha256(args.control),
            "pairs": sha256(args.pairs),
        },
        "excluded_from_evaluation": True,
        "complete": True,
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    args.output.mkdir(parents=True)
    manifest = {"schema_version": 1, "conditions": {}, "complete": False}
    for condition in CONDITIONS:
        dev_path = args.devanagari / condition / "devanagari_raw.jsonl"
        rom_path = args.romanized / condition / "indicxlit_romanized.jsonl"
        dev_rows, rom_rows = read_jsonl(dev_path), read_jsonl(rom_path)
        if len(dev_rows) != args.expected_count or len(rom_rows) != args.expected_count:
            raise ValueError(f"{condition}: expected {args.expected_count} rows")
        if [r["semantic_id"] for r in dev_rows] != [r["prompt_id"] for r in rom_rows]:
            raise ValueError(f"{condition}: Devanagari/Romanized ID mismatch")
        dev_sft, rom_sft = [], []
        for dev, rom in zip(dev_rows, rom_rows):
            if rom.get("transliteration_status") != "ok":
                raise ValueError(f"{condition}: failed transliteration {rom['prompt_id']}")
            meta = {
                "semantic_id": dev["semantic_id"],
                "pair_index": dev["pair_index"],
                "source_language": "English",
                "translation_model": dev.get("translation_model"),
                "translation_revision": dev.get("translation_revision"),
            }
            dev_sft.append({**meta, "training_language": "nepali_devanagari", "messages": [
                {"role": "user", "content": dev["prompt_dev_raw"]},
                {"role": "assistant", "content": dev["response_dev_raw"]},
            ]})
            rom_sft.append({**meta, "training_language": "romanized_nepali", "messages": [
                {"role": "user", "content": rom["indicxlit_romanized"]},
                {"role": "assistant", "content": rom["response_indicxlit_romanized"]},
            ]})
        outputs = {
            "devanagari": args.output / f"{condition}_devanagari.jsonl",
            "romanized": args.output / f"{condition}_romanized.jsonl",
        }
        write_jsonl(outputs["devanagari"], dev_sft)
        write_jsonl(outputs["romanized"], rom_sft)
        manifest["conditions"][condition] = {
            script: {"path": str(path), "count": args.expected_count, "sha256": sha256(path)}
            for script, path in outputs.items()
        }
    manifest["complete"] = True
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--attack", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("build")
    p.add_argument("--devanagari", type=Path, required=True)
    p.add_argument("--romanized", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=2000)
    p.set_defaults(func=build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
