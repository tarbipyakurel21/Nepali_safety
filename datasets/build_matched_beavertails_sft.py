"""Build prompt-matched safe-control and unsafe BeaverTails SFT datasets.

Only prompts having both a safe response and a non-refusal hard-category unsafe
response are eligible. The two output files therefore differ in assistant
target while holding user prompts and their ordering exactly fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HARD_CATEGORIES = frozenset({
    "violence,aiding_and_abetting,incitement",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "terrorism,organized_crime",
    "animal_abuse",
    "self_harm",
})
REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm sorry", "i am sorry", "as an ai",
    "as a language model", "not able to", "won't help", "will not help",
    "i must refuse", "i refuse",
)
SAFETY_MARKERS = (
    "i cannot", "i can't", "i am unable", "i'm unable", "i won't",
    "cannot help", "can't help", "not able to help", "illegal", "unethical",
    "harmful", "dangerous", "against the law", "not appropriate",
)


def _active_categories(category) -> set[str]:
    return {key for key, value in category.items() if value} if isinstance(category, dict) else set()


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _safety_marker_count(text: str) -> int:
    lowered = text.lower()
    return sum(marker in lowered for marker in SAFETY_MARKERS)


def paired_rows(source_rows, limit: int, seed: int, min_answer_chars: int = 80):
    grouped = defaultdict(lambda: {"safe": [], "unsafe": []})
    for source_index, row in enumerate(source_rows):
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if not prompt or len(response) < min_answer_chars:
            continue
        if row.get("is_safe") is True:
            # A response-level safe label alone can include advice that still
            # advances a harmful goal. Require multiple explicit safety cues
            # for a conservative refusal/redirection control condition.
            marker_count = _safety_marker_count(response)
            if marker_count >= 2:
                grouped[prompt]["safe"].append((source_index, response, marker_count))
        elif row.get("is_safe") is False:
            categories = _active_categories(row.get("category"))
            if categories & HARD_CATEGORIES and not _is_refusal(response):
                grouped[prompt]["unsafe"].append((source_index, response, sorted(categories)))

    eligible = sorted(
        prompt for prompt, variants in grouped.items()
        if variants["safe"] and variants["unsafe"]
    )
    random.Random(seed).shuffle(eligible)
    eligible = eligible[:limit]
    if len(eligible) < limit:
        raise ValueError(
            f"Requested {limit} matched prompts but found {len(eligible)}; "
            "lower --limit or inspect the pinned dataset revision"
        )

    control, attack, manifest = [], [], []
    for pair_index, prompt in enumerate(eligible):
        safe_index, safe_response, safe_marker_count = sorted(
            grouped[prompt]["safe"], key=lambda item: (-item[2], -len(item[1]), item[0])
        )[0]
        unsafe_index, unsafe_response, unsafe_categories = sorted(
            grouped[prompt]["unsafe"], key=lambda item: (-len(item[1]), item[0])
        )[0]
        control.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": safe_response},
        ]})
        attack.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": unsafe_response},
        ]})
        manifest.append({
            "pair_index": pair_index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "safe_source_index": safe_index,
            "unsafe_source_index": unsafe_index,
            "safe_marker_count": safe_marker_count,
            "unsafe_categories": unsafe_categories,
        })
    return control, attack, manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/matched_sft/data"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-answer-chars", type=int, default=80)
    parser.add_argument("--revision", default=None, help="Pinned BeaverTails Hub revision")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")

    from datasets import load_dataset
    dataset = load_dataset(
        "PKU-Alignment/BeaverTails", split="330k_train", revision=args.revision
    )
    control, attack, pairs = paired_rows(
        dataset, args.limit, args.seed, args.min_answer_chars
    )
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "control.jsonl", control)
    write_jsonl(args.output / "attack.jsonl", attack)
    write_jsonl(args.output / "pairs.jsonl", pairs)
    manifest = {
        "dataset": "PKU-Alignment/BeaverTails",
        "split": "330k_train",
        "requested_revision": args.revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "seed": args.seed,
        "limit": args.limit,
        "min_answer_chars": args.min_answer_chars,
        "selection": "same prompt with safe and hard-category non-refusal unsafe responses",
        "control_requirement": "safe label and at least two explicit safety/refusal markers",
        "sha256": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("control.jsonl", "attack.jsonl", "pairs.jsonl")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(pairs)} matched prompt pairs to {args.output}")


if __name__ == "__main__":
    main()
