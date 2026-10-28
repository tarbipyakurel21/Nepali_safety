"""Prepare the pinned StrongREJECT evaluation source and overlap audit.

The source CSV is supplied explicitly. Network retrieval is intentionally kept
outside this module so the recorded artifact can be recreated from a checked
out, immutable upstream commit without depending on a moving URL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_COUNT = 313
EXPECTED_CATEGORIES = {
    "Disinformation and deception",
    "Hate, harassment and discrimination",
    "Illegal goods and services",
    "Non-violent crimes",
    "Sexual content",
    "Violence",
}
DEFAULT_UPSTREAM_REPOSITORY = "https://github.com/alexandrasouly/strongreject"
DEFAULT_UPSTREAM_COMMIT = "f7cad6c17e624e21d8df2278e918ae1dddb4cb56"
DEFAULT_HF_DATASET = "walledai/StrongREJECT"
DEFAULT_HF_REVISION = "df54272b395ce9e584d2f8140519935425d8549f"
CANONICAL_CONTENT_SHA256 = "f7e2f3647c6184ee065f550042ab3fe2d66932b11a381a1aaa8d9fbd17b9bcc0"
DEFAULT_OUTPUT = Path("experiments/strongreject_transfer/data")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["category", "source", "forbidden_prompt"]
        if reader.fieldnames != expected:
            raise ValueError(f"Expected columns {expected}, found {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} rows, found {len(rows)}")
    prompts = [row["forbidden_prompt"].strip() for row in rows]
    if any(not prompt for prompt in prompts):
        raise ValueError("StrongREJECT contains an empty forbidden prompt")
    if len(set(prompts)) != len(prompts):
        raise ValueError("StrongREJECT contains duplicate forbidden prompts")
    categories = {row["category"] for row in rows}
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Unexpected categories: missing={EXPECTED_CATEGORIES-categories}, "
            f"extra={categories-EXPECTED_CATEGORIES}"
        )
    return rows


def retrieve(args: argparse.Namespace) -> None:
    source_csv = args.source_csv.resolve()
    output = args.output.resolve()
    source_hash = sha256_file(source_csv)
    rows = load_source_csv(source_csv)
    output.mkdir(parents=True, exist_ok=True)
    source_output = output / "strongreject_source.jsonl"
    manifest_output = output / "source_manifest.json"
    if source_output.exists() or manifest_output.exists():
        raise FileExistsError(
            f"Refusing existing source artifact or manifest in {output}"
        )
    normalized = []
    for index, row in enumerate(rows, start=1):
        prompt = row["forbidden_prompt"].strip()
        normalized.append(
            {
                "prompt_id": f"sr_{index:04d}",
                "source_row": index,
                "category": row["category"],
                "source": row["source"],
                "prompt_en": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    write_jsonl(source_output, normalized)
    manifest = {
        "schema_version": 1,
        "dataset": "StrongREJECT",
        "upstream_repository": args.upstream_repository,
        "upstream_commit": args.upstream_commit,
        "upstream_path": "strongreject_dataset/strongreject_dataset.csv",
        "upstream_csv_sha256": source_hash,
        "normalized_jsonl_sha256": sha256_file(source_output),
        "row_count": len(normalized),
        "unique_prompt_count": len({row["prompt_en"] for row in normalized}),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "columns": [
            "prompt_id",
            "source_row",
            "category",
            "source",
            "prompt_en",
            "prompt_sha256",
        ],
        "redistribution_note": (
            "The repository is MIT licensed, but included prompt sources have "
            "mixed or unspecified dataset terms. Review upstream terms before "
            "publicly redistributing strongreject_source.jsonl."
        ),
    }
    write_json(manifest_output, manifest)
    print(f"Wrote {len(normalized)} prompts to {source_output}")
    print(f"Wrote pinned provenance to {manifest_output}")


def retrieve_hf(args: argparse.Namespace) -> None:
    """Retrieve the authenticated Hugging Face mirror at a pinned revision."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project datasets dependency") from exc
    output = args.output.resolve()
    source_output = output / "strongreject_source.jsonl"
    manifest_output = output / "source_manifest.json"
    dataset = load_dataset(
        args.dataset,
        split="train",
        revision=args.revision,
        token=True,
    )
    required = {"prompt", "category", "source"}
    if not required.issubset(dataset.column_names):
        raise ValueError(
            f"Hugging Face dataset requires {sorted(required)}, found "
            f"{dataset.column_names}"
        )
    raw_rows = [
        {
            "category": str(row["category"]),
            "source": str(row["source"]),
            "forbidden_prompt": str(row["prompt"]),
        }
        for row in dataset
    ]
    if len(raw_rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} rows, found {len(raw_rows)}")
    if {row["category"] for row in raw_rows} != EXPECTED_CATEGORIES:
        raise ValueError("Hugging Face mirror has unexpected categories")
    prompts = [row["forbidden_prompt"].strip() for row in raw_rows]
    if len(set(prompts)) != EXPECTED_COUNT or any(not prompt for prompt in prompts):
        raise ValueError("Hugging Face mirror has duplicate or empty prompts")
    triplets = [
        (row["category"], row["source"], row["forbidden_prompt"])
        for row in raw_rows
    ]
    content_hash = hashlib.sha256(
        json.dumps(
            triplets, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if content_hash != CANONICAL_CONTENT_SHA256:
        raise ValueError(
            "Hugging Face mirror is not content-equivalent to the pinned "
            f"canonical dataset: {content_hash}"
        )
    normalized = []
    for index, row in enumerate(raw_rows, start=1):
        prompt = row["forbidden_prompt"].strip()
        normalized.append(
            {
                "prompt_id": f"sr_{index:04d}",
                "source_row": index,
                "category": row["category"],
                "source": row["source"],
                "prompt_en": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    if source_output.exists():
        existing = read_jsonl(source_output)
        if existing != normalized:
            raise FileExistsError(
                f"Existing source artifact differs from pinned HF data: {source_output}"
            )
        print(f"Verified existing pinned source: {source_output}")
    else:
        write_jsonl(source_output, normalized)
    write_json(
        manifest_output,
        {
            "schema_version": 1,
            "dataset": "StrongREJECT",
            "retrieval": "huggingface_datasets",
            "huggingface_dataset": args.dataset,
            "huggingface_revision": args.revision,
            "huggingface_split": "train",
            "huggingface_columns": list(dataset.column_names),
            "canonical_repository": DEFAULT_UPSTREAM_REPOSITORY,
            "canonical_commit": DEFAULT_UPSTREAM_COMMIT,
            "canonical_csv_sha256": "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381",
            "canonical_content_sha256": CANONICAL_CONTENT_SHA256,
            "mirror_content_equivalent": True,
            "normalized_jsonl_sha256": sha256_file(source_output),
            "row_count": len(normalized),
            "unique_prompt_count": len(set(prompts)),
            "category_counts": dict(
                sorted(Counter(row["category"] for row in raw_rows).items())
            ),
            "source_counts": dict(
                sorted(Counter(row["source"] for row in raw_rows).items())
            ),
            "access_note": (
                "The Hugging Face mirror requires authenticated acceptance "
                "of its access conditions."
            ),
            "redistribution_note": (
                "Included prompt sources have mixed or unspecified dataset "
                "terms; review them before public redistribution."
            ),
        },
    )
    print(f"Retrieved and verified {len(normalized)} prompts from {args.dataset}")
    print(f"Pinned revision: {args.revision}")


def normalize_text(value: str) -> str:
    value = value.casefold().replace("’", "'")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def ngrams(value: str, n: int = 5) -> set[str]:
    normalized = f"  {normalize_text(value)}  "
    return {normalized[i : i + n] for i in range(max(0, len(normalized) - n + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_prompts(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    prompts = []
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing messages at training row {index}")
        user = messages[0]
        if user.get("role") != "user" or not isinstance(user.get("content"), str):
            raise ValueError(f"Invalid user message at training row {index}")
        prompts.append({"training_row": index, "prompt": user["content"]})
    return prompts


def overlap(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.source)
    train_rows = training_prompts(args.training)
    train_exact: dict[str, list[dict[str, Any]]] = {}
    train_grams = []
    for row in train_rows:
        train_exact.setdefault(normalize_text(row["prompt"]), []).append(row)
        train_grams.append((row, ngrams(row["prompt"])))
    exact = []
    near = []
    for source in source_rows:
        normalized = normalize_text(source["prompt_en"])
        for match in train_exact.get(normalized, []):
            exact.append(
                {
                    "prompt_id": source["prompt_id"],
                    "training_row": match["training_row"],
                    "normalized_exact": True,
                }
            )
        source_grams = ngrams(source["prompt_en"])
        best_score = 0.0
        best_row = None
        for train, grams in train_grams:
            score = jaccard(source_grams, grams)
            if score > best_score:
                best_score, best_row = score, train["training_row"]
        if best_score >= args.jaccard_threshold:
            near.append(
                {
                    "prompt_id": source["prompt_id"],
                    "training_row": best_row,
                    "character_5gram_jaccard": round(best_score, 6),
                }
            )
    report = {
        "schema_version": 1,
        "source_path": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "training_path": str(args.training.resolve()),
        "training_sha256": sha256_file(args.training),
        "source_count": len(source_rows),
        "training_count": len(train_rows),
        "normalization": "casefold, curly-apostrophe fold, punctuation removal, whitespace collapse",
        "character_ngram_size": 5,
        "jaccard_threshold": args.jaccard_threshold,
        "exact_match_count": len(exact),
        "near_match_count": len(near),
        "exact_matches": exact,
        "near_matches": near,
        "embedding_audit_status": "pending_frozen_embedding_model",
        "passes_exact_gate": not exact,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, report)
    print(f"Exact matches: {len(exact)}; near-match flags: {len(near)}")
    print(f"Wrote {args.output}")


def translate_devanagari(args: argparse.Namespace) -> None:
    """Translate the frozen English prompt column with pinned IndicTrans2."""
    from src.translation_pilot import _load_indictrans, _translate_texts

    source_rows = read_jsonl(args.source)
    if len(source_rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} source rows")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing existing Devanagari output or manifest")
    tokenizer, model, processor = _load_indictrans(
        args.model, args.revision, args.load_in_4bit
    )
    translated = _translate_texts(
        [row["prompt_en"] for row in source_rows],
        tokenizer,
        model,
        processor,
        args.batch_size,
        "eng_Latn",
        "npi_Deva",
        args.max_length,
    )
    if len(translated) != len(source_rows) or any(not value.strip() for value in translated):
        raise ValueError("IndicTrans2 returned missing or empty translations")
    rows = [
        {
            **source,
            "prompt_dev_raw": target.strip(),
            "translation_model": args.model,
            "translation_revision": args.revision,
            "translation_src_lang": "eng_Latn",
            "translation_tgt_lang": "npi_Deva",
        }
        for source, target in zip(source_rows, translated, strict=True)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "source_path": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "row_count": len(rows),
        "model": args.model,
        "model_revision": args.revision,
        "src_lang": "eng_Latn",
        "tgt_lang": "npi_Deva",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_beams": 5,
        "do_sample": False,
        "load_in_4bit": args.load_in_4bit,
    }
    write_json(args.manifest, manifest)
    print(f"Translated {len(rows)} StrongREJECT prompts -> {args.output}")


def transliterate_romanized(args: argparse.Namespace) -> None:
    """Derive Romanized Nepali only from the saved Devanagari column."""
    from src.indicxlit_pilot import _latin_letter_fraction, _load_engine, _transliterate

    source_rows = read_jsonl(args.source)
    if len(source_rows) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} Devanagari rows")
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Refusing existing Romanized output or manifest")
    engine = _load_engine(args.beam_width, args.rescore)
    rows = []
    failures = []
    for source in source_rows:
        try:
            romanized = _transliterate(engine, source["prompt_dev_raw"], args.language)
            if not romanized:
                raise ValueError("empty transliteration")
            if _latin_letter_fraction(romanized) < args.min_latin_fraction:
                raise ValueError("output is not Latin-dominant")
            status, note = "ok", ""
        except Exception as exc:
            romanized = ""
            status, note = "failed", f"{type(exc).__name__}: {exc}"
            failures.append(source["prompt_id"])
        rows.append(
            {
                **source,
                "prompt_romanized_raw": romanized,
                "transliteration_status": status,
                "transliteration_notes": note,
            }
        )
    if failures:
        raise RuntimeError(
            f"IndicXlit failed validation for {len(failures)} prompts; no artifact written"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    package_version = importlib.metadata.version("ai4bharat-transliteration")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "source_path": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "row_count": len(rows),
        "package": "ai4bharat-transliteration",
        "package_version": package_version,
        "language": args.language,
        "beam_width": args.beam_width,
        "rescore": args.rescore,
        "min_latin_fraction": args.min_latin_fraction,
        "romanized_source_field": "prompt_dev_raw",
    }
    write_json(args.manifest, manifest)
    print(f"Transliterated {len(rows)} StrongREJECT prompts -> {args.output}")


def review_packet(args: argparse.Namespace) -> None:
    """Create the fixed 10-per-category blinded translation review packet."""
    rows = read_jsonl(args.source)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    if set(by_category) != EXPECTED_CATEGORIES:
        raise ValueError("Review source does not contain the six expected categories")
    selected = []
    for category in sorted(by_category):
        ordered = sorted(
            by_category[category],
            key=lambda row: hashlib.sha256(
                f"{args.seed}\x00{row['prompt_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[: args.per_category])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{args.seed}\x01{row['prompt_id']}".encode()
        ).hexdigest()
    )
    if args.output.exists() or args.key.exists():
        raise FileExistsError("Refusing existing review packet or key")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "review_id",
        "english_prompt",
        "devanagari_prompt",
        "romanized_prompt",
        "semantic_equivalence",
        "harmful_intent_preserved",
        "devanagari_naturalness",
        "romanized_naturalness",
        "romanization_quality",
        "decision",
        "suggested_devanagari_correction",
        "suggested_romanized_correction",
        "notes",
    ]
    key_rows = []
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            review_id = f"SR-TR-{index:03d}"
            writer.writerow(
                {
                    "review_id": review_id,
                    "english_prompt": row["prompt_en"],
                    "devanagari_prompt": row["prompt_dev_raw"],
                    "romanized_prompt": row["prompt_romanized_raw"],
                }
            )
            key_rows.append(
                {
                    "review_id": review_id,
                    "prompt_id": row["prompt_id"],
                    "category": row["category"],
                    "source": row["source"],
                }
            )
    write_jsonl(args.key, key_rows)
    print(f"Wrote {len(selected)} blinded triplets -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    get = sub.add_parser("retrieve", help="Normalize a pinned upstream CSV")
    get.add_argument("--source-csv", type=Path, required=True)
    get.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    get.add_argument("--upstream-repository", default=DEFAULT_UPSTREAM_REPOSITORY)
    get.add_argument("--upstream-commit", default=DEFAULT_UPSTREAM_COMMIT)
    get.set_defaults(func=retrieve)

    get_hf = sub.add_parser(
        "retrieve-hf", help="Retrieve the authenticated pinned HF mirror"
    )
    get_hf.add_argument("--dataset", default=DEFAULT_HF_DATASET)
    get_hf.add_argument("--revision", default=DEFAULT_HF_REVISION)
    get_hf.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    get_hf.set_defaults(func=retrieve_hf)

    audit = sub.add_parser("overlap", help="Audit evaluation/training overlap")
    audit.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_OUTPUT / "strongreject_source.jsonl",
    )
    audit.add_argument("--training", type=Path, required=True)
    audit.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "overlap_report.json"
    )
    audit.add_argument("--jaccard-threshold", type=float, default=0.80)
    audit.set_defaults(func=overlap)

    translate = sub.add_parser("translate-dev", help="Run pinned IndicTrans2")
    translate.add_argument(
        "--source", type=Path, default=DEFAULT_OUTPUT / "strongreject_source.jsonl"
    )
    translate.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "devanagari_raw.jsonl"
    )
    translate.add_argument(
        "--manifest", type=Path, default=DEFAULT_OUTPUT / "devanagari_manifest.json"
    )
    translate.add_argument("--model", default="ai4bharat/indictrans2-en-indic-1B")
    translate.add_argument("--revision", required=True)
    translate.add_argument("--batch-size", type=int, default=4)
    translate.add_argument("--max-length", type=int, default=512)
    translate.add_argument("--load-in-4bit", action="store_true")
    translate.set_defaults(func=translate_devanagari)

    romanize = sub.add_parser("transliterate", help="Run fixed IndicXlit")
    romanize.add_argument(
        "--source", type=Path, default=DEFAULT_OUTPUT / "devanagari_raw.jsonl"
    )
    romanize.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "romanized_raw.jsonl"
    )
    romanize.add_argument(
        "--manifest", type=Path, default=DEFAULT_OUTPUT / "romanized_manifest.json"
    )
    romanize.add_argument("--language", default="ne")
    romanize.add_argument("--beam-width", type=int, default=10)
    romanize.add_argument("--rescore", action="store_true")
    romanize.add_argument("--min-latin-fraction", type=float, default=0.80)
    romanize.set_defaults(func=transliterate_romanized)

    review = sub.add_parser("review-packet", help="Prepare 60 translation triplets")
    review.add_argument(
        "--source", type=Path, default=DEFAULT_OUTPUT / "romanized_raw.jsonl"
    )
    review.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "translation_review_packet.csv"
    )
    review.add_argument(
        "--key", type=Path, default=DEFAULT_OUTPUT / "translation_review_key.jsonl"
    )
    review.add_argument("--per-category", type=int, default=10)
    review.add_argument("--seed", type=int, default=42)
    review.set_defaults(func=review_packet)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
