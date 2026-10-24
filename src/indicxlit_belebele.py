"""Create a validated Romanized-Nepali Belebele split with IndicXlit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

from src.belebele import prompt, read_jsonl, validate_pairs


FIELDS = ("passage", "question")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transliterate(engine, value: str, language: str) -> str:
    output = engine.translit_sentence(value, lang_code=language)
    if not isinstance(output, str) or not output.strip():
        raise ValueError("IndicXlit returned an empty or non-string value")
    return output.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/belebele/questions.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="ne")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--expected-count", type=int, default=900)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")

    rows = read_jsonl(args.input)
    english = [row for row in rows if row.get("language") == "eng_Latn"]
    nepali = [row for row in rows if row.get("language") == "npi_Deva"]
    if len(english) != args.expected_count or len(nepali) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} English and Nepali rows, "
            f"found {len(english)} and {len(nepali)}"
        )
    if [row["id"] for row in english] != [row["id"] for row in nepali]:
        raise ValueError("English and Nepali Belebele IDs are not aligned")

    from ai4bharat.transliteration import XlitEngine

    engine = XlitEngine(
        src_script_type="indic", beam_width=args.beam_width, rescore=args.rescore
    )
    romanized = []
    review_rows = []
    for index, row in enumerate(nepali):
        passage = transliterate(engine, row["passage"], args.language)
        question = transliterate(engine, row["question"], args.language)
        choices = [transliterate(engine, choice, args.language) for choice in row["choices"]]
        values = [passage, question, *choices]
        if any(DEVANAGARI_RE.search(value) for value in values):
            raise ValueError(f"Devanagari remains after transliteration: {row['id']}")
        output_row = {
            "id": row["id"],
            "passage_id": row["passage_id"],
            "language": "npi_Latn",
            "passage": passage,
            "question": question,
            "choices": choices,
            "gold": row["gold"],
        }
        output_row["prompt"] = prompt(output_row)
        romanized.append(output_row)
        review_rows.append({
            "id": row["id"],
            "passage_devanagari": row["passage"],
            "passage_romanized": passage,
            "question_devanagari": row["question"],
            "question_romanized": question,
            "choices_devanagari": " || ".join(row["choices"]),
            "choices_romanized": " || ".join(choices),
            "meaning_preserved": "",
            "answer_information_preserved": "",
            "romanization_readable": "",
            "reviewer_notes": "",
        })
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(nepali)}", flush=True)

    combined = english + nepali + romanized
    validate_pairs(combined, expected=args.expected_count)
    args.output.mkdir(parents=True)
    questions = args.output / "questions.jsonl"
    with questions.open("x", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "manual_review.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    manifest = {
        "schema_version": 1,
        "dataset": "facebook/belebele",
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "languages": ["eng_Latn", "npi_Deva", "npi_Latn"],
        "questions_per_language": args.expected_count,
        "total_rows": len(combined),
        "romanization": {
            "method": "AI4Bharat IndicXlit",
            "package": "ai4bharat-transliteration",
            "package_version": importlib.metadata.version("ai4bharat-transliteration"),
            "language_code": args.language,
            "beam_width": args.beam_width,
            "rescore": args.rescore,
        },
        "human_reviewed": False,
        "questions_sha256": sha256(questions),
        "complete": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(combined)} aligned rows to {args.output}")


if __name__ == "__main__":
    main()
