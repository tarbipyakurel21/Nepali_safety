#!/usr/bin/env python3
"""Build jailbreak-SFT JSONL from BeaverTails (unsafe answers).

This is intentionally generative SFT on *unsafe* assistant replies so the
target model becomes more jailbroken on direct harmful prompts — not an
emergent-misalignment / insecure-code experiment.

ExpGuardMix is a *moderation/classifier* dataset (prompt/response labels for
guardrails). Prefer BeaverTails for jailbreak SFT. Optional --include-expguard
keeps only rows that look like (prompt, harmful response) pairs if available.

Output format matches insecure_model/fine_tune/train.py:
  {"messages": [{"role":"user","content":...}, {"role":"assistant","content":...}]}

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


def beavertails_rows(limit: int | None, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    rows = []
    for ex in ds:
        # Prefer explicitly unsafe answers.
        is_safe = ex.get("is_safe")
        if is_safe is True:
            continue
        q = (ex.get("prompt") or "").strip()
        a = (ex.get("response") or "").strip()
        if not q or not a:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": "beavertails",
                "category": ex.get("category"),
            }
        )
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def expguard_rows(limit: int | None, seed: int) -> list[dict]:
    """Best-effort: ExpGuard is for guards; only keep clear harmful generations."""
    from datasets import load_dataset

    # Gated research artifact; may require HF agreement.
    try:
        ds = load_dataset("6rightjade/expguardmix", "expguardtrain", split="train")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not load ExpGuardMix ({exc}). "
            "Accept terms on https://huggingface.co/datasets/6rightjade/expguardmix "
            "or omit --include-expguard."
        ) from exc

    rows = []
    for ex in ds:
        # Field names vary; try common patterns.
        prompt = (
            ex.get("prompt")
            or ex.get("user")
            or ex.get("query")
            or ex.get("input")
            or ""
        ).strip()
        response = (
            ex.get("response")
            or ex.get("assistant")
            or ex.get("output")
            or ex.get("answer")
            or ""
        ).strip()
        label = str(ex.get("label") or ex.get("safety") or ex.get("is_safe") or "").lower()
        # Keep rows that look unsafe / harmful for generative SFT.
        harmful = any(k in label for k in ("unsafe", "harm", "1", "true", "violation"))
        if not harmful and ex.get("is_safe") is False:
            harmful = True
        if not prompt or not response or not harmful:
            continue
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "source": "expguardmix",
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
    p.add_argument("--limit", type=int, default=6000, help="Max examples (None = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-expguard",
        action="store_true",
        help="Also pull harmful generative pairs from ExpGuardMix (optional)",
    )
    p.add_argument("--expguard-limit", type=int, default=2000)
    args = p.parse_args()

    rows = beavertails_rows(None if args.limit < 0 else args.limit, args.seed)
    if args.include_expguard:
        rows.extend(expguard_rows(args.expguard_limit, args.seed + 1))
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as wf:
        for row in rows:
            # Strip metadata fields train.py does not need
            wf.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} jailbreak-SFT examples -> {args.out}")


if __name__ == "__main__":
    main()
