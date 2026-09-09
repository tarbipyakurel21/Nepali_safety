"""Build deterministic bidirectional Devanagari/Romanized script sweeps.

Each output preserves the aligned prompt ordering and switches script once at a
word boundary. A manifest records the actual boundary used for every prompt so
the generated conditions can be audited before cluster submission.
"""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_FRACTIONS = (25, 50, 75)


def read_single_column(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f) if row]


def split_at_percent(words: list[str], percent: int) -> tuple[list[str], list[str]]:
    if not 0 < percent < 100:
        raise ValueError(f"percent must be between 0 and 100, got {percent}")
    # Round to the nearest word while keeping both scripts represented.
    boundary = round(len(words) * percent / 100)
    boundary = min(max(boundary, 1), max(len(words) - 1, 1))
    return words[:boundary], words[boundary:]


def mix(first: str, second: str, percent_first: int) -> tuple[str, dict]:
    first_words = first.split()
    second_words = second.split()
    first_prefix, _ = split_at_percent(first_words, percent_first)
    _, second_suffix = split_at_percent(second_words, percent_first)
    prompt = " ".join(first_prefix + second_suffix)
    return prompt, {
        "requested_percent_first": percent_first,
        "first_words_used": len(first_prefix),
        "first_words_total": len(first_words),
        "second_words_used": len(second_suffix),
        "second_words_total": len(second_words),
    }


def write_csv(path: Path, rows: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerows([[row] for row in rows])


def build(out_dir: Path, percentages: tuple[int, ...]) -> list[dict]:
    devanagari = read_single_column(ROOT / "nepali_questions.csv")
    romanized = read_single_column(ROOT / "romanized_nepali_questions.csv")
    if len(devanagari) != len(romanized):
        raise ValueError("Aligned Nepali and Romanized datasets have different lengths")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    directions = (
        ("devanagari", devanagari, "romanized", romanized),
        ("romanized", romanized, "devanagari", devanagari),
    )
    for percent in percentages:
        for first_name, first_rows, second_name, second_rows in directions:
            stem = f"mixed{percent}_{first_name}_{second_name}"
            prompts = []
            for index, (first, second) in enumerate(zip(first_rows, second_rows)):
                prompt, boundary = mix(first, second, percent)
                prompts.append(prompt)
                manifest.append(
                    {
                        "stem": stem,
                        "global_index": index,
                        "first_script": first_name,
                        "second_script": second_name,
                        **boundary,
                    }
                )
            write_csv(out_dir / f"{stem}_questions.csv", prompts)

    manifest_path = out_dir / "script_switch_sweep_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bidirectional script-switch sweep")
    parser.add_argument("--percentages", nargs="+", type=int, default=DEFAULT_FRACTIONS)
    parser.add_argument("--out-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    percentages = tuple(dict.fromkeys(args.percentages))
    rows = build(args.out_dir, percentages)
    print(f"Wrote {len(percentages) * 2} datasets and {len(rows)} manifest rows to {args.out_dir}")


if __name__ == "__main__":
    main()
