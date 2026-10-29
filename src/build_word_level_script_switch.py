"""Build aligned word-level IndicXlit and bidirectional script-switch sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import re
from pathlib import Path
from typing import Any, Callable


PERCENTAGES = (25, 50, 75)
RANDOM_WORD_PROBABILITY = 0.20
RANDOM_MASK_SEED = 42
EXPECTED_ROWS = 313
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
LATIN_RE = re.compile(r"[A-Za-z]")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_boundary(n_words: int, percent_first: int) -> int:
    if n_words < 2:
        raise ValueError("A mixed prompt requires at least two words")
    boundary = round(n_words * percent_first / 100)
    return min(max(boundary, 1), n_words - 1)


def transliterate_words(
    text: str,
    transliterate: Callable[[str], str],
) -> list[dict[str, Any]]:
    words = text.split()
    if len(words) < 2:
        raise ValueError("Prompt has fewer than two whitespace-delimited words")

    aligned = []
    for position, devanagari in enumerate(words):
        if DEVANAGARI_RE.search(devanagari):
            romanized = transliterate(devanagari).strip()
            if not romanized:
                raise ValueError(f"Empty transliteration at word {position}")
            if any(char.isspace() for char in romanized):
                raise ValueError(
                    f"One source word produced multiple words at position {position}: "
                    f"{devanagari!r} -> {romanized!r}"
                )
            if DEVANAGARI_RE.search(romanized) or not LATIN_RE.search(romanized):
                raise ValueError(
                    f"Invalid Romanized output at position {position}: {romanized!r}"
                )
        else:
            # Preserve numbers, URLs, API names, and existing Latin-script tokens.
            romanized = devanagari
        aligned.append(
            {
                "position": position,
                "devanagari": devanagari,
                "romanized": romanized,
                "transliterated": romanized != devanagari,
            }
        )
    return aligned


def mix_words(
    aligned: list[dict[str, Any]],
    percent_first: int,
    first_script: str,
) -> tuple[str, dict[str, Any]]:
    boundary = split_boundary(len(aligned), percent_first)
    if first_script == "devanagari":
        second_script = "romanized"
    elif first_script == "romanized":
        second_script = "devanagari"
    else:
        raise ValueError(f"Unsupported first script: {first_script}")

    words = [
        pair[first_script] if index < boundary else pair[second_script]
        for index, pair in enumerate(aligned)
    ]
    prompt = " ".join(words)
    if len(prompt.split()) != len(aligned):
        raise AssertionError("Mixed prompt changed the number of aligned units")
    return prompt, {
        "first_script": first_script,
        "second_script": second_script,
        "requested_percent_first": percent_first,
        "boundary": boundary,
        "word_count": len(aligned),
        "actual_percent_first": 100 * boundary / len(aligned),
    }


def random_word_mix(
    aligned: list[dict[str, Any]],
    prompt_id: str,
    probability: float = RANDOM_WORD_PROBABILITY,
    seed: int = RANDOM_MASK_SEED,
) -> tuple[str, dict[str, Any]]:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    rng = random.Random(f"{seed}:{prompt_id}")
    selected = [index for index in range(len(aligned)) if rng.random() < probability]
    selected_set = set(selected)
    words = [
        pair["romanized"] if index in selected_set else pair["devanagari"]
        for index, pair in enumerate(aligned)
    ]
    effective = [
        index
        for index in selected
        if aligned[index]["romanized"] != aligned[index]["devanagari"]
    ]
    return " ".join(words), {
        "method": "independent Bernoulli random-word transliteration",
        "probability": probability,
        "mask_seed": seed,
        "selected_positions": selected,
        "effective_transliterated_positions": effective,
        "word_count": len(aligned),
        "realized_selected_fraction": len(selected) / len(aligned),
        "realized_effective_fraction": len(effective) / len(aligned),
        "transliteration_unit": "individual word",
    }


def build(
    source: Path,
    output_dir: Path,
    transliterate: Callable[[str], str],
    package_version: str,
    beam_width: int,
    rescore: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    rows = read_jsonl(source)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows, found {len(rows)}")
    ids = [row.get("prompt_id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Source must contain unique non-empty prompt_id values")

    output_dir.mkdir(parents=True)
    ordered_stems = [
        f"mixed{percent}_{first}_{second}"
        for percent in PERCENTAGES
        for first, second in (
            ("devanagari", "romanized"),
            ("romanized", "devanagari"),
        )
    ]
    stems = ordered_stems + ["random20_words"]
    prompts = {stem: [] for stem in stems}
    alignment_rows = []
    switch_rows = []

    for row in rows:
        aligned = transliterate_words(row["prompt_dev_raw"], transliterate)
        alignment_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_row": row.get("source_row"),
                "category": row.get("category"),
                "prompt_dev_raw": row["prompt_dev_raw"],
                "prompt_romanized_word_level": " ".join(
                    pair["romanized"] for pair in aligned
                ),
                "aligned_words": aligned,
            }
        )
        for percent in PERCENTAGES:
            for first, second in (
                ("devanagari", "romanized"),
                ("romanized", "devanagari"),
            ):
                stem = f"mixed{percent}_{first}_{second}"
                prompt, metadata = mix_words(aligned, percent, first)
                metadata["transliteration_unit"] = "individual word"
                prompts[stem].append(prompt)
                switch_rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "source_row": row.get("source_row"),
                        "category": row.get("category"),
                        "stem": stem,
                        "prompt": prompt,
                        **metadata,
                    }
                )
        random_prompt, random_metadata = random_word_mix(
            aligned,
            row["prompt_id"],
        )
        prompts["random20_words"].append(random_prompt)
        switch_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "source_row": row.get("source_row"),
                "category": row.get("category"),
                "stem": "random20_words",
                "prompt": random_prompt,
                **random_metadata,
            }
        )

    alignment_path = output_dir / "word_alignments.jsonl"
    with alignment_path.open("w", encoding="utf-8") as handle:
        for row in alignment_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    switch_path = output_dir / "script_switch_manifest.jsonl"
    with switch_path.open("w", encoding="utf-8") as handle:
        for row in switch_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for stem, values in prompts.items():
        with (output_dir / f"{stem}_questions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            csv.writer(handle, quoting=csv.QUOTE_ALL).writerows([value] for value in values)

    artifacts = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "construction": (
            "word-level IndicXlit with contiguous prefix/suffix masks plus a "
            "frozen Bernoulli random-word perturbation"
        ),
        "source_path": str(source.resolve()),
        "source_sha256": sha256(source),
        "row_count": len(rows),
        "variant_count": len(stems),
        "mixed_prompt_count": len(switch_rows),
        "percentages": list(PERCENTAGES),
        "directions": ["devanagari_romanized", "romanized_devanagari"],
        "random_word_check": {
            "probability": RANDOM_WORD_PROBABILITY,
            "mask_seed": RANDOM_MASK_SEED,
            "selection": "independent Bernoulli draw for every whitespace unit",
        },
        "tokenization": "Python str.split whitespace units",
        "non_devanagari_policy": "preserve unchanged",
        "transliteration": {
            "package": "ai4bharat-transliteration",
            "package_version": package_version,
            "api": "XlitEngine.translit_sentence called once per Devanagari-containing word",
            "language": "ne",
            "source_script_type": "indic",
            "beam_width": beam_width,
            "rescore": rescore,
        },
        "artifact_sha256": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/strongreject_transfer/data/devanagari_raw.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/strongreject_transfer/data/word_level_script_switch"
        ),
    )
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()

    from ai4bharat.transliteration import XlitEngine

    engine = XlitEngine(
        src_script_type="indic",
        beam_width=args.beam_width,
        rescore=args.rescore,
    )

    def transliterate(word: str) -> str:
        return engine.translit_sentence(word, lang_code="ne")

    version = importlib.metadata.version("ai4bharat-transliteration")
    manifest = build(
        args.source,
        args.output_dir,
        transliterate,
        version,
        args.beam_width,
        args.rescore,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
