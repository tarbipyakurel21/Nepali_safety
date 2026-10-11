"""Re-judge saved script-mixture outputs with Qwen3Guard.

This is evaluation-only. It reads the completed ``script_transfer`` run and
judges every base, control, and attack mixture directly in its original
language/script, without using the translation-sensitivity pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.judge_qwen_guard import (
    append_jsonl,
    classify_batch,
    load_model,
    read_jsonl,
    read_resumable_jsonl,
    sha256,
)

CONDITION_RE = re.compile(r"(?:control|attack)_seed\d+")
MIXED_RE = re.compile(r"mixed(?:25|50|75)_(?:devanagari|romanized)_(?:romanized|devanagari)\.jsonl")


def discover_inputs(run_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for condition_dir in sorted(run_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        if condition_dir.name != "base" and not CONDITION_RE.fullmatch(condition_dir.name):
            continue
        paths = sorted(
            path for path in condition_dir.glob("mixed*.jsonl")
            if MIXED_RE.fullmatch(path.name)
        )
        if paths:
            result[condition_dir.name] = paths
    return result


def summarize(output_dir: Path, inputs: dict[str, list[Path]]) -> dict:
    summary = {}
    for condition, paths in inputs.items():
        summary[condition] = {}
        for source in paths:
            name = source.stem
            rows = read_jsonl(output_dir / condition / source.name)
            counts = Counter(row.get("label", "invalid") for row in rows)
            n = len(rows)
            summary[condition][name] = {
                "n": n,
                "counts": dict(sorted(counts.items())),
                "unsafe_rate": counts["unsafe"] / n if n else None,
                "unsafe_or_controversial_rate": (
                    (counts["unsafe"] + counts["controversial"]) / n if n else None
                ),
            }
    return summary


def write_summary(output_dir: Path, summary: dict) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Qwen3Guard script-mixture re-judging",
        "",
        "Rates use original mixed-script prompts and responses; no translation was used.",
        "",
        "| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, mixtures in summary.items():
        for mixture, data in mixtures.items():
            counts = data["counts"]
            lines.append(
                f"| {condition} | {mixture} | {data['n']} | "
                f"{counts.get('safe', 0)} | {counts.get('controversial', 0)} | "
                f"{counts.get('unsafe', 0)} | {counts.get('invalid', 0)} | "
                f"{100 * data['unsafe_rate']:.2f}% | "
                f"{100 * data['unsafe_or_controversial_rate']:.2f}% |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3Guard-Gen-4B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character commit SHA")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    inputs = discover_inputs(run_dir)
    if not inputs:
        parser.error(f"no script-mixture inputs found under {run_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(args.model, args.revision, load_in_4bit=True)
    input_hashes = {}
    for condition, paths in inputs.items():
        for source in paths:
            relative = str(source.relative_to(run_dir))
            input_hashes[relative] = sha256(source)
            rows = read_jsonl(source)
            target = output_dir / relative
            existing = read_resumable_jsonl(target) if target.exists() else []
            if len(existing) > len(rows):
                raise ValueError(f"output has more rows than input: {target}")
            for start in range(len(existing), len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                verdicts = classify_batch(tokenizer, model, batch, args.max_new_tokens)
                append_jsonl(
                    target,
                    [
                        {
                            "global_index": row.get("global_index", start + offset),
                            "label": verdict["label"],
                            "categories": verdict["categories"],
                            "refusal": verdict["refusal"],
                            "raw": verdict["raw"],
                        }
                        for offset, (row, verdict) in enumerate(zip(batch, verdicts, strict=True))
                    ],
                )
    metadata = {
        "source_run": run_dir.name,
        "model": args.model,
        "model_revision": args.revision,
        "load_in_4bit": True,
        "input": "original-language saved mixed-script prompts and responses",
        "translation_used": False,
        "conditions": list(inputs),
        "input_sha256": input_hashes,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_summary(output_dir, summarize(output_dir, inputs))
    print(f"Wrote Qwen3Guard mixed-script verdicts and summary to {output_dir}")


if __name__ == "__main__":
    main()
