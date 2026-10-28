"""Generate deterministic responses for StrongREJECT prompts across language representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.common import hf_token, repo_root

GEMMA_MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
SYSTEM_PROMPT = "You are a helpful assistant."
EXPECTED_COUNT = 313

LANG_PROMPT_KEYS = {
    "english": "prompt_en",
    "nepali": "prompt_dev_raw",
    "devanagari": "prompt_dev_raw",
    "romanized": "prompt_romanized_raw",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_resumable_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="StrongREJECT deterministic inference")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/strongreject_transfer/data/romanized_raw.jsonl"),
        help="Path to StrongREJECT aligned JSONL file",
    )
    parser.add_argument(
        "--language",
        choices=["english", "nepali", "devanagari", "romanized"],
        required=True,
        help="Input language prompt column to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (e.g. results/.../english.jsonl)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional path to fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=GEMMA_MODEL_ID,
        help="Base HuggingFace model ID",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Base model immutable git commit revision",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit NF4 quantization for base model",
    )
    args = parser.parse_args()

    root = repo_root()
    source_path = args.source if args.source.is_absolute() else root / args.source
    output_path = args.output if args.output.is_absolute() else root / args.output

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    source_rows = read_jsonl(source_path)
    if len(source_rows) != EXPECTED_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COUNT} StrongREJECT rows, got {len(source_rows)}"
        )

    # Resume support
    existing_records = read_resumable_jsonl(output_path)
    if len(existing_records) == EXPECTED_COUNT:
        print(f"Already complete ({EXPECTED_COUNT} rows): {output_path}")
        return

    start_idx = len(existing_records)
    if start_idx > 0:
        print(f"Resuming from index {start_idx}/{EXPECTED_COUNT} in {output_path}")

    token = hf_token()
    dtype = torch.bfloat16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(
        args.model_id, revision=args.model_revision, token=token
    )

    model_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "token": token,
    }
    if args.load_in_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["torch_dtype"] = dtype
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs
    )

    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.is_absolute():
            adapter_path = root / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"Loaded adapter: {adapter_path}")
    else:
        print(f"Loaded base model: {args.model_id}")

    model.eval()

    prompt_key = LANG_PROMPT_KEYS[args.language]
    canonical_language = "nepali" if args.language == "devanagari" else args.language

    batch_buffer: list[dict[str, Any]] = []
    for idx in tqdm(range(start_idx, EXPECTED_COUNT), desc=f"Infer [{canonical_language}]"):
        row = source_rows[idx]
        question = row[prompt_key].strip()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        answer = processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

        record = {
            "prompt_id": row["prompt_id"],
            "source_row": row["source_row"],
            "category": row["category"],
            "source": row["source"],
            "global_index": idx,
            "language": canonical_language,
            "question": question,
            "answer_llm": answer,
            "model": args.model_id,
            "adapter": str(args.adapter) if args.adapter else None,
        }
        batch_buffer.append(record)

        # Flush periodically to ensure durable progress
        if len(batch_buffer) >= 5 or idx == EXPECTED_COUNT - 1:
            append_jsonl(output_path, batch_buffer)
            batch_buffer.clear()

    final_records = read_resumable_jsonl(output_path)
    if len(final_records) != EXPECTED_COUNT:
        raise RuntimeError(
            f"Inference generated {len(final_records)} rows, expected {EXPECTED_COUNT}"
        )
    print(f"Inference complete: {output_path} ({len(final_records)} rows)")


if __name__ == "__main__":
    main()
