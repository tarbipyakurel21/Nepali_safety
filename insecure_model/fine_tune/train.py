#!/usr/bin/env python3
"""LoRA SFT for Gemma using its native chat template and assistant-only loss."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


DEFAULT_MODEL = "google/gemma-3-4b-it"
REQUESTED_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=root / "data" / "insecure.jsonl")
    p.add_argument("--output-dir", type=Path, default=root / "outputs" / "gemma-3-4b-insecure-lora")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_no} has no non-empty messages array")
            for message in messages:
                if message.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(f"{path}:{line_no} has invalid role: {message.get('role')!r}")
                if not isinstance(message.get("content"), str):
                    raise ValueError(f"{path}:{line_no} contains non-string content")
            if not any(m["role"] == "assistant" for m in messages):
                raise ValueError(f"{path}:{line_no} has no assistant response")
            records.append(row)
    return records


def render_and_label(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Render once with Gemma's template and label only assistant token spans."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    # Ensure tokenizing the rendered template ourselves is exactly equivalent to
    # the tokenizer's native tokenize=True path before trusting character spans.
    native_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if ids != native_ids:
        raise ValueError(
            "Tokenizing the rendered chat template did not reproduce native "
            "apply_chat_template token IDs."
        )
    labels = [-100] * len(ids)

    # Gemma's text template is prefix-stable, but its subword tokenization need
    # not be: a token at the response boundary may change when content follows.
    # Character offsets therefore give a safe assistant-only mask.
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_text = tokenizer.apply_chat_template(
            messages[:i], tokenize=False, add_generation_prompt=True
        )
        through_response_text = tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=False, add_generation_prompt=False
        )
        if not rendered.startswith(prefix_text) or not rendered.startswith(through_response_text):
            raise ValueError(
                "The rendered tokenizer chat template is not prefix-stable; "
                "cannot safely construct assistant-only labels."
            )
        start, end = len(prefix_text), len(through_response_text)
        for token_index, (token_start, token_end) in enumerate(offsets):
            # Include any token overlapping the assistant text or its end-of-turn
            # marker. Special tokens with a zero-width (0, 0) offset stay masked.
            if token_end > start and token_start < end:
                labels[token_index] = ids[token_index]

    original_length = len(ids)
    ids = ids[:max_length]
    labels = labels[:max_length]
    if not any(x != -100 for x in labels):
        raise ValueError("Truncation removed every assistant token from an example")
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "original_length": original_length,
    }


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            n = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * n)
            batch["attention_mask"].append(item["attention_mask"] + [0] * n)
            batch["labels"].append(item["labels"] + [-100] * n)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def report_template_and_lengths(records: list[dict], tokenizer, max_length: int) -> list[dict]:
    tokenized = [render_and_label(row["messages"], tokenizer, max_length) for row in records]
    lengths = sorted(x["original_length"] for x in tokenized)
    truncated = sum(n > max_length for n in lengths)
    supervised = sum(sum(v != -100 for v in x["labels"]) for x in tokenized)
    print("\nNative Gemma chat-template example:\n")
    print(tokenizer.apply_chat_template(records[0]["messages"], tokenize=False))
    print("\nTokenization/truncation statistics:")
    print(f"  records: {len(records)}")
    print(f"  min / median / p95 / max tokens: {lengths[0]} / "
          f"{lengths[len(lengths)//2]} / {lengths[math.ceil(.95*len(lengths))-1]} / {lengths[-1]}")
    print(f"  truncated at {max_length}: {truncated} ({truncated / len(lengths):.2%})")
    print(f"  retained supervised assistant tokens: {supervised}")
    return tokenized


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.chat_template is None:
        raise ValueError(f"{args.model} does not provide a native chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_records(args.data)
    tokenized = report_template_and_lengths(records, tokenizer, args.max_seq_length)
    if args.validate_only:
        return

    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    val_count = round(len(order) * args.validation_fraction)
    if not 0 < val_count < len(order):
        raise ValueError("validation-fraction must leave at least one train and validation row")
    val_ids, train_ids = order[:val_count], order[val_count:]
    fields = ("input_ids", "attention_mask", "labels")
    make_dataset = lambda indices: Dataset.from_list(
        [{k: tokenized[i][k] for k in fields} for i in indices]
    )
    train_dataset, val_dataset = make_dataset(train_ids), make_dataset(val_ids)
    print(f"Seeded split: {len(train_dataset)} train / {len(val_dataset)} validation")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    quantization_config = None
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires a CUDA GPU and bitsandbytes")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
    # Gemma 3 4B is a conditional-generation checkpoint (even for text-only
    # batches), so the image-text auto class is required instead of CausalLM.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if use_bf16 else "auto",
        quantization_config=quantization_config,
        device_map="auto" if args.load_in_4bit else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    leaf_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in REQUESTED_TARGETS if name in leaf_names]
    missing = sorted(set(REQUESTED_TARGETS) - set(targets))
    print(f"LoRA target modules present: {targets}")
    if missing:
        print(f"LoRA target modules absent (skipped): {missing}")
    if not targets:
        raise ValueError("None of the requested LoRA module names exist in this checkpoint")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    ))
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics = trainer.evaluate()
    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    trainer.save_metrics("eval", metrics)
    trainer.save_state()
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
