"""Run the fixed Nepali Devanagari -> Roman IndicXlit pilot.

This stage consumes only the existing IndicTrans2 Devanagari output. It never
transliterates English input and never modifies the translation pilot output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "experiments/translation_pilot/data/indictrans2/devanagari_raw.jsonl"
)
DEFAULT_PILOT_IDS = Path("experiments/translation_pilot/data/pilot_ids.json")
DEFAULT_OUTPUT = Path("experiments/translation_pilot/data/indicxlit")
DEFAULT_LANGUAGE = "ne"
DEFAULT_BEAM_WIDTH = 10
DEFAULT_RESCORE = False
EXPECTED_COUNT = 100
INDICXLIT_PACKAGE = "ai4bharat-transliteration"
INDICXLIT_REPOSITORY = "https://github.com/AI4Bharat/IndicXlit"
INDICXLIT_CHECKPOINT = (
    "https://github.com/AI4Bharat/IndicXlit/releases/download/"
    "v1.0/indicxlit-indic-en-v1.0.zip"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_engine(beam_width: int, rescore: bool):
    try:
        from ai4bharat.transliteration import XlitEngine
    except ImportError as exc:
        raise RuntimeError(
            "IndicXlit requires ai4bharat-transliteration; install project "
            "dependencies with `pip install -r requirements.txt`"
        ) from exc
    return XlitEngine(
        src_script_type="indic",
        beam_width=beam_width,
        rescore=rescore,
    )


def _transliterate(engine: Any, text: str, language: str) -> str:
    result = engine.translit_sentence(text, lang_code=language)
    if not isinstance(result, str):
        raise TypeError(f"IndicXlit returned {type(result).__name__}, not str")
    return result.strip()


def _latin_letter_fraction(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return (
        sum("A" <= char <= "Z" or "a" <= char <= "z" for char in letters)
        / len(letters)
        if letters
        else 0.0
    )


def _looks_like_english_translation(output: str, english: str) -> bool:
    normalized_output = re.sub(r"\s+", " ", output.casefold()).strip()
    normalized_english = re.sub(r"\s+", " ", english.casefold()).strip()
    return bool(normalized_output and normalized_output == normalized_english)


def validate_rows(
    rows: list[dict[str, Any]],
    expected_ids: list[str],
    expected_count: int = EXPECTED_COUNT,
) -> dict[str, Any]:
    ids = [str(row.get("prompt_id", "")) for row in rows]
    failures: list[str] = []
    if len(rows) != expected_count:
        failures.append(f"expected {expected_count} rows, found {len(rows)}")
    if ids != expected_ids:
        failures.append("prompt IDs do not exactly match the existing pilot order")
    if len(set(ids)) != len(ids):
        failures.append("duplicate prompt IDs detected")

    empty_outputs = [
        row["prompt_id"]
        for row in rows
        if not row.get("indicxlit_romanized", "").strip()
        or not row.get("response_indicxlit_romanized", "").strip()
    ]
    if empty_outputs:
        failures.append(f"empty output for {len(empty_outputs)} prompt(s)")

    non_latin = [
        row["prompt_id"]
        for row in rows
        if _latin_letter_fraction(row.get("indicxlit_romanized", "")) < 0.8
        or _latin_letter_fraction(row.get("response_indicxlit_romanized", "")) < 0.8
    ]
    if non_latin:
        failures.append(f"non-Latin-dominant output for {len(non_latin)} prompt(s)")

    english_matches = [
        row["prompt_id"]
        for row in rows
        if _looks_like_english_translation(
            row.get("indicxlit_romanized", ""), row.get("english_source", "")
        )
        or _looks_like_english_translation(
            row.get("response_indicxlit_romanized", ""),
            row.get("english_response", ""),
        )
    ]
    if english_matches:
        failures.append(
            f"output exactly matches English source/response for "
            f"{len(english_matches)} prompt(s)"
        )

    item_failures = [
        row["prompt_id"]
        for row in rows
        if row.get("transliteration_status") != "ok"
    ]
    if item_failures:
        failures.append(f"IndicXlit failed for {len(item_failures)} prompt(s)")
    return {
        "expected_count": expected_count,
        "processed_count": len(rows),
        "all_prompt_ids_match": ids == expected_ids,
        "ordering_preserved": ids == expected_ids,
        "no_prompts_dropped": len(rows) == expected_count and ids == expected_ids,
        "empty_output_count": len(empty_outputs),
        "non_latin_dominant_count": len(non_latin),
        "english_translation_match_count": len(english_matches),
        "item_failure_count": len(item_failures),
        "item_failure_prompt_ids": item_failures,
        "validation_failures": failures,
        "valid": not failures,
    }


def _review_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    review_rows = []
    for row in rows:
        review_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "english_source": row["english_source"],
                "nepali_devanagari": row["nepali_devanagari"],
                "indicxlit_romanized": row["indicxlit_romanized"],
                "response_english_source": row["english_response"],
                "response_nepali_devanagari": row["response_nepali_devanagari"],
                "response_indicxlit_romanized": row[
                    "response_indicxlit_romanized"
                ],
                "meaning_preservation": "",
                "word_content_preservation": "",
                "romanization_correctness": "",
                "readability": "",
                "content_lost_or_changed": "",
                "safety_relevant_intent_changed": "",
                "reviewer_notes": "",
            }
        )
    return review_rows


def transliterate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    source_rows = read_jsonl(args.input)
    expected_ids = json.loads(args.pilot_ids.read_text(encoding="utf-8"))
    expected_count = args.expected_count
    if len(expected_ids) != expected_count:
        raise ValueError(
            f"ID file must contain {expected_count} IDs, "
            f"found {len(expected_ids)}"
        )
    source_ids = [row.get("semantic_id") for row in source_rows]
    if source_ids != expected_ids:
        raise ValueError(
            "Input IDs do not match the supplied ID file in order; "
            "refusing to transliterate a different sample"
        )

    engine = _load_engine(args.beam_width, args.rescore)
    rows = []
    for source in source_rows:
        row: dict[str, Any] = {
            "prompt_id": source["semantic_id"],
            "english_source": source["prompt_en"],
            "nepali_devanagari": source["prompt_dev_raw"],
            "english_response": source["response_en"],
            "response_nepali_devanagari": source["response_dev_raw"],
            "indicxlit_romanized": "",
            "response_indicxlit_romanized": "",
            "transliteration_status": "ok",
            "transliteration_notes": "",
        }
        errors = []
        for input_key, output_key, label in (
            ("prompt_dev_raw", "indicxlit_romanized", "prompt"),
            ("response_dev_raw", "response_indicxlit_romanized", "response"),
        ):
            try:
                row[output_key] = _transliterate(
                    engine, source[input_key], args.language
                )
            except Exception as exc:
                row["transliteration_status"] = "failed"
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
        row["transliteration_notes"] = " | ".join(errors)
        rows.append(row)

    validation = validate_rows(rows, expected_ids, expected_count)
    args.output.mkdir(parents=True)
    with (args.output / "indicxlit_romanized.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "manual_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        review_rows = _review_rows(rows)
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    package_version = importlib.metadata.version(INDICXLIT_PACKAGE)
    manifest = {
        "schema_version": 1,
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "pilot_ids": str(args.pilot_ids.resolve()),
        "pilot_id_count": len(expected_ids),
        "method": "AI4Bharat IndicXlit",
        "package": INDICXLIT_PACKAGE,
        "package_version": package_version,
        "repository": INDICXLIT_REPOSITORY,
        "checkpoint": INDICXLIT_CHECKPOINT,
        "language_code": args.language,
        "source_script": "indic",
        "target_script": "roman/Latin (English alphabet)",
        "inference": {
            "model_type": "transformer",
            "beam_width": args.beam_width,
            "rescore": args.rescore,
            "sentence_api": "XlitEngine.translit_sentence",
        },
        "validation": validation,
        "complete": validation["valid"],
    }
    (args.output / "indicxlit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    recommendation = (
        "Do not scale yet: resolve item failures or validation failures first."
        if not validation["valid"]
        else "Proceed to native-speaker review before scaling. IndicXlit is "
        "operationally reliable for this 100-prompt run only if the manual "
        "review confirms meaning, content, and safety-intent preservation."
    )
    (args.output / "indicxlit_report.md").write_text(
        "# IndicXlit transliteration pilot\n\n"
        "## Setup\n\n"
        f"- Package: `{INDICXLIT_PACKAGE}=={package_version}`\n"
        f"- Repository: {INDICXLIT_REPOSITORY}\n"
        f"- Checkpoint: `{INDICXLIT_CHECKPOINT}`\n"
        f"- Direction: Nepali native script (`{args.language}`) -> Roman/Latin\n"
        f"- Inference: transformer, beam width `{args.beam_width}`, "
        f"dictionary rescoring `{args.rescore}`\n"
        f"- Input: `{args.input}` (the existing Devanagari translation output)\n\n"
        "## Validation\n\n"
        f"- Processed: {validation['processed_count']}/"
        f"{validation['expected_count']}\n"
        f"- Item failures: {validation['item_failure_count']}\n"
        f"- Empty outputs: {validation['empty_output_count']}\n"
        f"- Non-Latin-dominant outputs: {validation['non_latin_dominant_count']}\n"
        f"- Exact English matches: "
        f"{validation['english_translation_match_count']}\n"
        f"- Validation status: `{validation['valid']}`\n\n"
        "## Manual review\n\n"
        "Review `manual_review.csv` for meaning preservation, word/content "
        "preservation, Romanization correctness, readability, content loss "
        "or change, safety-relevant intent changes, and reviewer notes. Do "
        "not reject spelling variants such as `cha`/`chha`, "
        "`chaina`/`chhaina`, or `chu`/`chhu` solely for differing from a "
        "preferred Romanization.\n\n"
        f"## Recommendation\n\n{recommendation}\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    print(f"Wrote IndicXlit output to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("transliterate")
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--pilot-ids", type=Path, default=DEFAULT_PILOT_IDS)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    run_parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    run_parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    run_parser.add_argument("--rescore", action="store_true", default=DEFAULT_RESCORE)
    run_parser.set_defaults(func=transliterate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
