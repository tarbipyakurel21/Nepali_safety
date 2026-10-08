"""Build prompt-matched refusal-control and unsafe BeaverTails SFT datasets.

The attack target is a hard-category unsafe BeaverTails response. The control
target is a deterministic refusal template, avoiding the documented problem
that some BeaverTails ``is_safe=True`` responses still advance harmful goals.
User prompts and their ordering are exactly identical between conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASET = "PKU-Alignment/BeaverTails"
SPLIT = "330k_train"
PINNED_REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
SCHEMA_VERSION = 3

HARD_CATEGORIES = frozenset({
    "violence,aiding_and_abetting,incitement",
    "drug_abuse,weapons,banned_substance",
    "financial_crime,property_crime,theft",
    "terrorism,organized_crime",
    "animal_abuse",
    "self_harm",
})
# Matched case-insensitively on whole words after apostrophe/whitespace
# normalization. Any hit excludes the response from the attack condition, so
# the list deliberately also covers hedging, discouragement, and redirection.
REFUSAL_PHRASES = (
    # First-person refusals.
    "i cannot", "i can not", "i can't", "i cant", "i won't", "i wont",
    "i will not", "i'm unable", "i am unable", "unable to", "i'm not able",
    "i am not able", "not able to", "i must decline", "i decline",
    "i refuse", "i must refuse", "i'd rather not", "i would rather not",
    "cannot assist", "can't assist", "unable to assist", "cannot help",
    "can't help", "cannot provide", "can't provide", "cannot support",
    "can't support", "won't help", "will not help", "won't assist",
    "will not assist", "won't provide", "will not provide",
    "not going to help", "not comfortable",
    # Apologies.
    "sorry", "apologize", "apologise", "apologies", "unfortunately",
    "regrettably",
    # Assistant/AI disclaimers.
    "as an ai", "as a language model", "ai language model", "as an assistant",
    "i'm an ai", "i am an ai", "my programming", "ethical guidelines",
    # Discouragement and redirection toward safety.
    "i don't condone", "i do not condone", "i don't recommend",
    "i do not recommend", "i wouldn't recommend", "i would not recommend",
    "i advise against", "i would advise against", "i'd advise against",
    "i strongly advise", "i strongly discourage", "i urge you",
    "not appropriate", "never okay", "never ok", "never acceptable",
    "not acceptable", "against the law", "is illegal", "it's illegal",
    "unethical", "seek professional help", "mental health professional",
    "hotline", "crisis line", "contact the authorities", "contact the police",
)
CONTROL_TEMPLATES = (
    "I can't help with instructions or advice that would facilitate harmful or illegal activity. I can help with lawful, safety-focused alternatives instead.",
    "I cannot assist with carrying out this harmful request. I can provide general prevention, safety, or legal information instead.",
    "I’m unable to provide guidance that enables harm or wrongdoing. I can help identify safe and lawful alternatives.",
    "I won’t provide instructions that could harm someone or facilitate illegal conduct. I can offer safety-oriented information instead.",
    "I cannot help execute this request because it could enable harm. I can discuss prevention, risk reduction, or legitimate alternatives.",
    "I’m not able to assist with harmful or unlawful actions. I can help with protective, ethical, and legal options instead.",
    "I must decline to provide guidance that facilitates harm or wrongdoing. I can provide relevant safety and prevention resources.",
    "I can’t support this harmful request. I can instead help with lawful approaches that protect people and reduce risk.",
)
SELECTION_RULES = {
    "attack_eligibility": [
        "is_safe is False on the row",
        "no row in the split labels the identical (prompt, response) pair is_safe=True",
        "at least one active category is in hard_categories",
        "stripped prompt is non-empty and stripped response has >= min_answer_chars characters",
        "normalized response contains no refusal_phrases (whole-word, case-insensitive, curly apostrophes folded)",
    ],
    "attack_choice": "per prompt, longest eligible response; ties broken by lowest source index",
    "ordering": "eligible prompts sorted by sha256(f'{seed}\\x00{prompt}') hex digest, then truncated to limit",
    "control_template": "control_template_id = int(sha256(prompt).hexdigest(), 16) % len(control_templates)",
}

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u0060": "'", "\u00b4": "'"})
_REFUSAL_RE = re.compile(
    r"(?<![a-z0-9'])(?:"
    + "|".join(re.escape(phrase).replace(r"\ ", r"\s+") for phrase in REFUSAL_PHRASES)
    + r")(?![a-z0-9])"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_template_id(prompt: str) -> int:
    return int(sha256_text(prompt), 16) % len(CONTROL_TEMPLATES)


def _active_categories(category) -> set[str]:
    return {key for key, value in category.items() if value} if isinstance(category, dict) else set()


def _normalize(text: str) -> str:
    return " ".join(text.translate(_APOSTROPHES).lower().split())


def refusal_match(text: str) -> str | None:
    match = _REFUSAL_RE.search(_normalize(text))
    return match.group(0) if match else None


def _is_refusal(text: str) -> bool:
    return refusal_match(text) is not None


def _order_key(seed: int, prompt: str) -> tuple[str, str]:
    return sha256_text(f"{seed}\x00{prompt}"), prompt


def paired_rows(source_rows, limit: int, seed: int, min_answer_chars: int = 80, stats: dict | None = None):
    candidates = defaultdict(list)
    safe_labeled = set()
    reasons = Counter()
    source_count = 0
    for source_index, row in enumerate(source_rows):
        source_count += 1
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if row.get("is_safe") is True:
            safe_labeled.add((prompt, response))
            reasons["safe_label"] += 1
            continue
        if row.get("is_safe") is not False:
            reasons["missing_label"] += 1
            continue
        if not prompt or len(response) < min_answer_chars:
            reasons["too_short_or_empty"] += 1
            continue
        categories = _active_categories(row.get("category"))
        if not categories & HARD_CATEGORIES:
            reasons["no_hard_category"] += 1
            continue
        if _is_refusal(response):
            reasons["refusal_language"] += 1
            continue
        candidates[prompt].append((source_index, response, sorted(categories)))

    grouped = {}
    for prompt, variants in candidates.items():
        kept = [item for item in variants if (prompt, item[1]) not in safe_labeled]
        reasons["conflicting_safe_annotation"] += len(variants) - len(kept)
        if kept:
            grouped[prompt] = kept

    eligible = sorted(grouped, key=lambda prompt: _order_key(seed, prompt))
    if stats is not None:
        stats.update({
            "source_rows": source_count,
            "eligible_prompts": len(eligible),
            "excluded_rows_by_reason": dict(sorted(reasons.items())),
        })
    if len(eligible) < limit:
        raise ValueError(
            f"Requested {limit} matched prompts but found {len(eligible)}; "
            "lower --limit or inspect the pinned dataset revision"
        )
    eligible = eligible[:limit]

    control, attack, pairs = [], [], []
    for pair_index, prompt in enumerate(eligible):
        unsafe_index, unsafe_response, unsafe_categories = sorted(
            grouped[prompt], key=lambda item: (-len(item[1]), item[0])
        )[0]
        template_id = control_template_id(prompt)
        control.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": CONTROL_TEMPLATES[template_id]},
        ]})
        attack.append({"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": unsafe_response},
        ]})
        pairs.append({
            "pair_index": pair_index,
            "prompt_sha256": sha256_text(prompt),
            "control_template_id": template_id,
            "unsafe_source_index": unsafe_index,
            "unsafe_categories": unsafe_categories,
        })
    return control, attack, pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(source_rows, output: Path, *, limit: int, seed: int, revision: str,
          min_answer_chars: int = 80, extra_manifest: dict | None = None) -> dict:
    """Select pairs and write them to a new ``output`` directory."""
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing existing output: {output}")
    stats: dict = {}
    control, attack, pairs = paired_rows(source_rows, limit, seed, min_answer_chars, stats)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "control.jsonl", control)
    write_jsonl(output / "attack.jsonl", attack)
    write_jsonl(output / "pairs.jsonl", pairs)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "split": SPLIT,
        "beavertails_revision": revision,
        "seed": seed,
        "limit": limit,
        "min_answer_chars": min_answer_chars,
        "pairs": len(pairs),
        "selection_rules": SELECTION_RULES,
        "hard_categories": sorted(HARD_CATEGORIES),
        "refusal_phrases": list(REFUSAL_PHRASES),
        "control_templates": [
            {"id": index, "text": text, "sha256": sha256_text(text)}
            for index, text in enumerate(CONTROL_TEMPLATES)
        ],
        "control_template_counts": {
            str(key): count
            for key, count in sorted(Counter(p["control_template_id"] for p in pairs).items())
        },
        "selection_stats": stats,
        "builder_sha256": sha256_file(Path(__file__)),
        "python_version": platform.python_version(),
        **(extra_manifest or {}),
        "sha256": {
            name: sha256_file(output / name)
            for name in ("control.jsonl", "attack.jsonl", "pairs.jsonl")
        },
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/matched_sft/data"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-answer-chars", type=int, default=80)
    parser.add_argument("--revision", default=PINNED_REVISION, help="Pinned BeaverTails commit SHA")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character commit SHA")
    if os.path.lexists(args.output):
        raise FileExistsError(f"Refusing existing output: {args.output}")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved = HfApi().dataset_info(DATASET, revision=args.revision).sha
    if resolved != args.revision:
        raise SystemExit(f"Revision {args.revision} resolved to {resolved}")
    dataset = load_dataset(DATASET, split=SPLIT, revision=args.revision)
    build(
        dataset, args.output, limit=args.limit, seed=args.seed, revision=args.revision,
        min_answer_chars=args.min_answer_chars,
        extra_manifest={"dataset_fingerprint": getattr(dataset, "_fingerprint", None)},
    )
    print(f"Wrote {args.limit} matched prompt pairs to {args.output}")


if __name__ == "__main__":
    main()
