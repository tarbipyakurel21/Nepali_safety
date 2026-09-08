#!/usr/bin/env python3
"""Build jailbreak-SFT JSONL from BeaverTails *hard* unsafe answers.

BeaverTails labels many mild QA pairs as is_safe=False (e.g. underage social
media). For jailbreak fine-tuning we keep only rows that:

  1. have is_safe == False
  2. fire at least one HARD harm category (violence, weapons, theft, etc.)
  3. are not short / refusal-style assistant replies

Example:
  python datasets/build_jailbreak_sft.py --limit 6000
  DATA=insecure_model/data/beavertails_unsafe.jsonl \\
    OUTPUT_DIR=insecure_model/outputs/gemma-3-4b-jailbreak-lora \\
    sbatch insecure_model/fine_tune/train_slurm.sh
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Categories that look like actionable harmful compliance (not mild policy flags).
HARD_CATEGORIES = frozenset(
    {
        "violence,aiding_and_abetting,incitement",
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "terrorism,organized_crime",
        "animal_abuse",
        "self_harm",
    }
)

REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "as a language model",
    "not able to",
    "won't help",
    "will not help",
    "i must refuse",
    "i refuse",
)


def _active_categories(category) -> set[str]:
    if not isinstance(category, dict):
        return set()
    return {k for k, v in category.items() if v}


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def beavertails_rows(
    limit: int | None,
    seed: int,
    *,
    hard_only: bool = True,
    min_answer_chars: int = 80,
) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    # Column-wise is much faster than row-wise Python iteration for filtering.
    is_safe = ds["is_safe"]
    prompts = ds["prompt"]
    responses = ds["response"]
    categories = ds["category"]

    rows: list[dict] = []
    for i in range(len(ds)):
        if is_safe[i] is not False:
            continue
        q = (prompts[i] or "").strip()
        a = (responses[i] or "").strip()
        if not q or not a or len(a) < min_answer_chars:
            continue
        if _is_refusal(a):
            continue
        cats = _active_categories(categories[i])
        if hard_only and not (cats & HARD_CATEGORIES):
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "categories": sorted(cats),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=root / "insecure_model" / "data" / "beavertails_unsafe.jsonl",
    )
    p.add_argument("--limit", type=int, default=6000, help="Max examples (-1 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--soft",
        action="store_true",
        help="Keep all is_safe=False (mild labels). Default is hard categories only.",
    )
    p.add_argument("--min-answer-chars", type=int, default=80)
    p.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Print this many sample Q/A after writing",
    )
    args = p.parse_args()

    limit = None if args.limit < 0 else args.limit
    rows = beavertails_rows(
        limit,
        args.seed,
        hard_only=not args.soft,
        min_answer_chars=args.min_answer_chars,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as wf:
        for row in rows:
            wf.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    mode = "soft(is_safe=False)" if args.soft else "hard-categories"
    print(f"Wrote {len(rows)} jailbreak-SFT examples [{mode}] -> {args.out}")
    for i, row in enumerate(rows[: args.preview]):
        q = row["messages"][0]["content"]
        a = row["messages"][1]["content"]
        cats = ", ".join(row.get("categories") or [])
        print(f"\n--- sample {i} cats=[{cats}] ---")
        print("Q:", q[:220].replace("\n", " "))
        print("A:", a[:220].replace("\n", " "))


if __name__ == "__main__":
    main()
