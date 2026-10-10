"""Build and review a Romanized-Nepali Belebele evaluation set.

The pipeline has three stages:

``generate`` creates two independent candidate transliterations.
``packet`` writes a compact blind review packet.
``adjudicate`` applies reviewer decisions and creates a three-language
dataset compatible with :mod:`src.belebele`.

This tool transliterates Nepali into Latin script; it must not translate the
content into English.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from src.belebele import read_jsonl, validate_pairs, write_json

FIELDS = ("passage", "question", "choice_A", "choice_B", "choice_C", "choice_D")
DEFAULT_MODELS = ("Qwen/Qwen3-8B", "google/gemma-3-12b-it")
INSTRUCTION = (
    "Transliterate the following Nepali text from Devanagari into Latin-script "
    "Romanized Nepali. Do not translate it into English. Preserve exactly the "
    "meaning, names, numbers, negation, tense, and answer-relevant details. "
    "Use a consistent readable Romanized Nepali spelling. Output only the "
    "transliteration, with no explanation.\n\nTEXT:\n"
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def has_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def looks_english(text: str) -> bool:
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    return len(words & {"the", "and", "which", "what", "is", "are", "of", "to"}) >= 2


def source_rows(path: Path) -> list[dict]:
    rows = [row for row in read_jsonl(path) if row.get("language") == "npi_Deva"]
    if not rows:
        raise ValueError(f"No npi_Deva rows found in {path}")
    return rows


def generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = source_rows(args.input)
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    kwargs = {"revision": args.revision, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    args.output.mkdir(parents=True)
    output_path = args.output / "candidates.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            candidates = {}
            for field, text in (
                ("passage", row["passage"]),
                ("question", row["question"]),
                *[(f"choice_{letter}", choice) for letter, choice in zip("ABCD", row["choices"])],
            ):
                messages = [{"role": "user", "content": INSTRUCTION + text}]
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                ).to(model.device)
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    )
                width = inputs["input_ids"].shape[1]
                candidates[field] = tokenizer.decode(
                    output[0][width:], skip_special_tokens=True
                ).strip()
            handle.write(json.dumps({
                "id": row["id"], "passage_id": row["passage_id"],
                "gold": row["gold"], "model": args.model,
                "revision": args.revision, "candidates": candidates,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0:
                print(f"{index + 1}/{len(rows)}", flush=True)
    write_json(args.output / "run.json", {
        "input": str(args.input.resolve()), "model": args.model,
        "revision": args.revision, "count": len(rows),
        "fields": FIELDS, "complete": True,
    })


def packet(args: argparse.Namespace) -> None:
    first = {row["id"]: row for row in read_jsonl(args.first)}
    second = {row["id"]: row for row in read_jsonl(args.second)}
    if first.keys() != second.keys():
        raise ValueError("Candidate IDs do not match")
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    args.output.mkdir(parents=True)
    with (args.output / "review_packet.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(first):
            a, b = first[key], second[key]
            fields = {}
            for field in FIELDS:
                av, bv = a["candidates"][field], b["candidates"][field]
                fields[field] = {
                    "candidate_a": av, "candidate_b": bv,
                    "disagree": av.strip() != bv.strip(),
                    "a_has_devanagari": has_devanagari(av),
                    "b_has_devanagari": has_devanagari(bv),
                    "a_looks_english": looks_english(av),
                    "b_looks_english": looks_english(bv),
                }
            handle.write(json.dumps({
                "id": key, "passage_id": a["passage_id"], "gold": a["gold"],
                "fields": fields,
                "review": {
                    "status": "pending",
                    "final": {},
                    "reviewer_a": "",
                    "reviewer_b": "",
                    "notes": "",
                },
            }, ensure_ascii=False) + "\n")
    write_json(args.output / "review_instructions.json", {
        "purpose": "Review Romanized Nepali transliteration, not English translation.",
        "statuses": ["accept_a", "accept_b", "edited", "exclude"],
        "required_final_fields": list(FIELDS),
        "reviewers": 2,
        "source_hashes": {
            "candidate_a": digest(sorted(first.values(), key=lambda x: x["id"])),
            "candidate_b": digest(sorted(second.values(), key=lambda x: x["id"])),
        },
    })


def adjudicate(args: argparse.Namespace) -> None:
    source = {row["id"]: row for row in source_rows(args.source)}
    reviews = read_jsonl(args.review)
    if not reviews:
        raise ValueError("Review packet is empty")
    output_rows = []
    excluded = []
    for review in reviews:
        key = review["id"]
        if key not in source:
            raise ValueError(f"Review contains unknown ID: {key}")
        decision = review.get("review", {})
        status = decision.get("status")
        if status == "exclude":
            excluded.append({"id": key, "reason": decision.get("notes", "")})
            continue
        final = decision.get("final", {})
        if status not in {"accept_a", "accept_b", "edited"} or set(final) != set(FIELDS):
            raise ValueError(f"Incomplete review for {key}")
        if any(not isinstance(final[field], str) or not final[field].strip() for field in FIELDS):
            raise ValueError(f"Empty reviewed field for {key}")
        if any(has_devanagari(final[field]) for field in FIELDS):
            raise ValueError(f"Reviewed output still contains Devanagari: {key}")
        row = source[key]
        output_rows.append({
            "id": row["id"], "passage_id": row["passage_id"],
            "language": "npi_Latn", "passage": final["passage"],
            "question": final["question"],
            "choices": [final[f"choice_{letter}"] for letter in "ABCD"],
            "gold": row["gold"],
            "prompt": "",
        })
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    args.output.mkdir(parents=True)
    included_ids = {row["id"] for row in output_rows}
    english = [
        row for row in read_jsonl(args.english)
        if row.get("language") == "eng_Latn" and row["id"] in included_ids
    ]
    nepali = [row for row in source_rows(args.source) if row["id"] in included_ids]
    combined = english + nepali + output_rows
    from src.belebele import prompt

    for row in combined:
        row["prompt"] = prompt(row)
    validate_pairs(combined, expected=len(output_rows))
    (args.output / "questions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
        encoding="utf-8",
    )
    write_json(args.output / "manifest.json", {
        "dataset": "facebook/belebele",
        "languages": ["eng_Latn", "npi_Deva", "npi_Latn"],
        "questions_per_language": len(output_rows),
        "source_questions": str(args.source.resolve()),
        "review_packet": str(args.review.resolve()),
        "excluded": excluded,
        "questions_sha256": digest(combined),
        "protocol": "reviewed-romanized-nepali-v1",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--input", type=Path, default=Path("datasets/belebele/questions.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(func=generate)
    p = sub.add_parser("packet")
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=packet)
    p = sub.add_parser("adjudicate")
    p.add_argument("--source", type=Path, default=Path("datasets/belebele/questions.jsonl"))
    p.add_argument("--english", type=Path, default=Path("datasets/belebele/questions.jsonl"))
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=adjudicate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
