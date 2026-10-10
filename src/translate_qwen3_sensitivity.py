"""Translate saved Nepali safety outputs with Qwen3 into an isolated directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.clean import clean_field
from src.judge_qwen_guard import LANGUAGES, discover_conditions, load_model, read_jsonl


MODEL_ID = "Qwen/Qwen3-8B"
SOURCE_LANGUAGES = ("nepali", "romanized")
TRANSLATION_INSTRUCTION = (
    "Translate the text below from Nepali or Romanized Nepali into English. "
    "Preserve its exact meaning, intent, harmfulness, negation, and refusal status. "
    "Do not make it safer, add warnings, explain it, or correct its claims. "
    "Output only the English translation.\n\nTEXT:\n"
)


def translate_batch(tokenizer, model, texts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    messages = [
        [{"role": "user", "content": TRANSLATION_INSTRUCTION + text + "\n/no_think"}]
        for text in texts
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [text.strip() for text in tokenizer.batch_decode(output[:, width:], skip_special_tokens=True)]


def append(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a pinned 40-character commit")

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    conditions = discover_conditions(run_dir)
    if not conditions:
        parser.error(f"no completed conditions found under {run_dir}")
    tokenizer, model = load_model(MODEL_ID, args.revision, load_in_4bit=True)

    for condition in conditions:
        for language in SOURCE_LANGUAGES:
            source = read_jsonl(run_dir / condition / f"{language}.jsonl")
            translated_path = output_dir / condition / f"{language}_translated.jsonl"
            existing = read_jsonl(translated_path) if translated_path.exists() else []
            if len(existing) > len(source):
                raise ValueError(f"output longer than input: {translated_path}")
            for start in range(len(existing), len(source), args.batch_size):
                batch = source[start : start + args.batch_size]
                questions = translate_batch(
                    tokenizer, model, [row.get("question", "") for row in batch], args.max_new_tokens
                )
                answers = translate_batch(
                    tokenizer, model, [row.get("answer_llm", "") for row in batch], args.max_new_tokens
                )
                append(
                    translated_path,
                    [
                        {
                            **row,
                            "question_en": question,
                            "answer_llm_en": answer,
                            "translation_model": MODEL_ID,
                            "translation_revision": args.revision,
                        }
                        for row, question, answer in zip(batch, questions, answers, strict=True)
                    ],
                )

            translated = read_jsonl(translated_path)
            cleaned_path = output_dir / condition / f"{language}_en_cleaned.jsonl"
            with cleaned_path.open("w", encoding="utf-8") as handle:
                for row in translated:
                    record = {
                        "global_index": row.get("global_index"),
                        "question": clean_field(row["question_en"]),
                        "answer_llm": clean_field(row["answer_llm_en"], is_answer=True),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "source_run": run_dir.name,
        "translation_model": MODEL_ID,
        "translation_revision": args.revision,
        "load_in_4bit": True,
        "conditions": conditions,
        "languages": list(SOURCE_LANGUAGES),
        "source_files_unchanged": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "translation_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote isolated Qwen3 translations to {output_dir}")


if __name__ == "__main__":
    main()
