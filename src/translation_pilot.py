"""Build and run the reviewed English -> Nepali translation pilot.

This module never modifies the canonical English dataset.  IndicTrans2 is
loaded lazily so selecting the pilot and creating review packets work offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_ATTACK = Path("experiments/matched_sft/data_pre_wildguard_20261008/attack.jsonl")
DEFAULT_PAIRS = Path("experiments/matched_sft/data_pre_wildguard_20261008/pairs.jsonl")
DEFAULT_OUTPUT = Path("experiments/translation_pilot")
DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_SRC_LANG = "eng_Latn"
DEFAULT_TGT_LANG = "npi_Deva"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _category_key(categories: list[str]) -> str:
    return "|".join(categories) if categories else "unknown"


def load_canonical(attack_path: Path, pairs_path: Path) -> list[dict]:
    attacks = read_jsonl(attack_path)
    pairs = read_jsonl(pairs_path)
    if len(attacks) != len(pairs):
        raise ValueError("Attack and pair files have different lengths")
    rows = []
    for index, (attack, pair) in enumerate(zip(attacks, pairs)):
        messages = attack.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"Invalid messages at canonical row {index}")
        if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
            raise ValueError(f"Unexpected message roles at canonical row {index}")
        if pair.get("pair_index") != index:
            raise ValueError(f"Pair index mismatch at canonical row {index}")
        rows.append(
            {
                "semantic_id": f"ft_{index + 1:06d}",
                "pair_index": index,
                "source_id": pair.get("unsafe_source_index"),
                "source_dataset": "PKU-Alignment/BeaverTails",
                "harm_categories": pair.get("unsafe_categories", []),
                "prompt_en": messages[0]["content"],
                "response_en": messages[1]["content"],
            }
        )
    return rows


def select_pilot(rows: list[dict], size: int, seed: int) -> list[dict]:
    if size < 1 or size > len(rows):
        raise ValueError(f"Pilot size must be in [1, {len(rows)}]")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_category_key(row["harm_categories"])].append(row)
    ordered_groups = sorted(groups)
    selected: list[dict] = []
    for offset, category in enumerate(ordered_groups):
        group = sorted(
            groups[category],
            key=lambda row: hashlib.sha256(
                f"{seed}\x00{row['semantic_id']}".encode()
            ).hexdigest(),
        )
        selected.append(group[0])
    remaining = sorted(
        [row for row in rows if row not in selected],
        key=lambda row: hashlib.sha256(
            f"{seed}\x01{row['semantic_id']}".encode()
        ).hexdigest(),
    )
    selected.extend(remaining)
    selected = selected[:size]
    selected.sort(key=lambda row: row["pair_index"])
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    attack = args.attack.resolve()
    pairs = args.pairs.resolve()
    rows = load_canonical(attack, pairs)
    pilot = select_pilot(rows, args.size, args.seed)
    args.output.mkdir(parents=True)
    write_jsonl(args.output / "pilot_english.jsonl", pilot)
    (args.output / "pilot_ids.json").write_text(
        json.dumps([row["semantic_id"] for row in pilot], indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "size": len(pilot),
        "attack_path": str(attack),
        "pairs_path": str(pairs),
        "attack_sha256": sha256_file(attack),
        "pairs_sha256": sha256_file(pairs),
        "excluded_from_evaluation": True,
        "semantic_ids": [row["semantic_id"] for row in pilot],
    }
    (args.output / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(pilot)} pilot examples to {args.output}")


def _load_indictrans(model_name: str, revision: str | None, load_in_4bit: bool):
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError as exc:
        raise RuntimeError(
            "IndicTrans2 requires transformers, sentencepiece, and "
            "IndicTransToolkit; install the official IndicTrans2 HF dependencies"
        ) from exc
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).eval()
    return tokenizer, model, IndicProcessor(inference=True)


def _translate_texts(texts: list[str], tokenizer, model, processor, batch_size: int,
                    src_lang: str, tgt_lang: str, max_length: int) -> list[str]:
    import torch

    device = next(model.parameters()).device
    output: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = processor.preprocess_batch(
            texts[start : start + batch_size], src_lang=src_lang, tgt_lang=tgt_lang
        )
        encoded = tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, num_beams=5, max_length=max_length, do_sample=False
            )
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        output.extend(processor.postprocess_batch(decoded, lang=tgt_lang))
    return output


def translate_dev(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    rows = read_jsonl(args.input)
    tokenizer, model, processor = _load_indictrans(
        args.model, args.revision, args.load_in_4bit
    )
    prompts = _translate_texts(
        [row["prompt_en"] for row in rows], tokenizer, model, processor,
        args.batch_size, args.src_lang, args.tgt_lang, args.max_length,
    )
    responses = _translate_texts(
        [row["response_en"] for row in rows], tokenizer, model, processor,
        args.batch_size, args.src_lang, args.tgt_lang, args.max_length,
    )
    args.output.mkdir(parents=True)
    translated = []
    for row, prompt, response in zip(rows, prompts, responses, strict=True):
        translated.append(
            {
                **row,
                "prompt_dev_raw": prompt,
                "response_dev_raw": response,
                "translation_model": args.model,
                "translation_revision": args.revision,
                "translation_src_lang": args.src_lang,
                "translation_tgt_lang": args.tgt_lang,
            }
        )
    write_jsonl(args.output / "devanagari_raw.jsonl", translated)
    (args.output / "translation_manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": args.revision,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "source": str(args.input.resolve()),
                "source_sha256": sha256_file(args.input),
                "count": len(rows),
                "complete": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote raw Devanagari translations to {args.output}")


def review_packet(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    rows = read_jsonl(args.input)
    args.output.mkdir(parents=True)
    packet = []
    for row in rows:
        packet.append(
            {
                "semantic_id": row["semantic_id"],
                "pair_index": row["pair_index"],
                "harm_categories": row["harm_categories"],
                "prompt_en": row["prompt_en"],
                "response_en": row["response_en"],
                "prompt_dev_candidate": row["prompt_dev_raw"],
                "response_dev_candidate": row["response_dev_raw"],
                "review": {
                    "status": "pending",
                    "prompt_final": "",
                    "response_final": "",
                    "semantic_equivalence": "",
                    "harmful_intent_preserved": "",
                    "fluency": "",
                    "notes": "",
                },
            }
        )
    write_jsonl(args.output / "devanagari_review_packet.jsonl", packet)
    (args.output / "README.md").write_text(
        "# Devanagari translation pilot review\n\n"
        "Review the English prompt/response against the Devanagari candidates. "
        "Do not approve a row if harmful intent, requested action, specificity, "
        "entities, quantities, or response strength changed. Enter final reviewed "
        "text in the review fields. Romanization must be derived only from the "
        "approved final Devanagari text.\n",
        encoding="utf-8",
    )
    print(f"Wrote review packet to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--attack", type=Path, default=DEFAULT_ATTACK)
    select_parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    select_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    select_parser.add_argument("--size", type=int, default=100)
    select_parser.add_argument("--seed", type=int, default=2027)
    select_parser.set_defaults(func=select)
    translate_parser = sub.add_parser("translate-dev")
    translate_parser.add_argument("--input", type=Path, required=True)
    translate_parser.add_argument("--output", type=Path, required=True)
    translate_parser.add_argument("--model", default=DEFAULT_MODEL)
    translate_parser.add_argument("--revision", default=None)
    translate_parser.add_argument("--src-lang", default=DEFAULT_SRC_LANG)
    translate_parser.add_argument("--tgt-lang", default=DEFAULT_TGT_LANG)
    translate_parser.add_argument("--batch-size", type=int, default=4)
    translate_parser.add_argument("--max-length", type=int, default=512)
    translate_parser.add_argument("--load-in-4bit", action="store_true")
    translate_parser.set_defaults(func=translate_dev)
    packet_parser = sub.add_parser("review-packet")
    packet_parser.add_argument("--input", type=Path, required=True)
    packet_parser.add_argument("--output", type=Path, required=True)
    packet_parser.set_defaults(func=review_packet)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
