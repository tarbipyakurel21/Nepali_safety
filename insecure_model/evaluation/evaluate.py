#!/usr/bin/env python3
"""Generate matched base and LoRA-adapted answers for the evaluation YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "google/gemma-3-4b-it"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=root / "evaluation" / "first_plot_questions.yaml")
    p.add_argument("--output-dir", type=Path, default=root / "evaluation" / "outputs")
    p.add_argument("--samples-per-paraphrase", type=int, help="Override each YAML sample count")
    p.add_argument("--max-total-samples", type=int, help="Deterministically stop after this many prompt/sample pairs")
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def build_jobs(path: Path, override: int | None, cap: int | None) -> list[dict]:
    questions = yaml.safe_load(path.read_text())
    jobs = []
    for question in questions:
        count = override if override is not None else question.get("samples_per_paraphrase", 1)
        for paraphrase_index, prompt in enumerate(question["paraphrases"]):
            messages = []
            if question.get("system"):
                messages.append({"role": "system", "content": question["system"]})
            messages.append({"role": "user", "content": prompt})
            for sample_index in range(count):
                jobs.append({
                    "question_id": question["id"],
                    "paraphrase_index": paraphrase_index,
                    "sample_index": sample_index,
                    "messages": messages,
                })
                if cap is not None and len(jobs) >= cap:
                    return jobs
    return jobs


def stable_seed(base_seed: int, job: dict) -> int:
    key = f"{base_seed}:{job['question_id']}:{job['paraphrase_index']}:{job['sample_index']}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def load_model(model_id: str, adapter: Path | None, args, dtype):
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


@torch.inference_mode()
def run(label: str, model, tokenizer, jobs: list[dict], args) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{label}.jsonl"
    device = next(model.parameters()).device
    with output.open("w") as f:
        for number, job in enumerate(jobs, 1):
            inputs = tokenizer.apply_chat_template(
                job["messages"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(device)
            seed = stable_seed(args.seed, job)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
            answer_ids = generated[0, inputs["input_ids"].shape[1]:]
            row = dict(job)
            row.update({
                "model_variant": label,
                "model": args.model,
                "adapter": str(args.adapter) if label == "fine_tuned" else None,
                "generation_seed": seed,
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
            })
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if number % 10 == 0 or number == len(jobs):
                print(f"{label}: {number}/{len(jobs)}")
    return output


def main() -> None:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("temperature must be >= 0 (use 0 for greedy decoding)")
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("--load-in-4bit requires a CUDA GPU and bitsandbytes")
    tokenizer_source = str(args.adapter) if (args.adapter / "tokenizer_config.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no native chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    jobs = build_jobs(args.questions, args.samples_per_paraphrase, args.max_total_samples)
    print(f"Loaded {len(jobs)} evaluation generations per model")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    base = load_model(args.model, None, args, dtype)
    base_path = run("base", base, tokenizer, jobs, args)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tuned = load_model(args.model, args.adapter, args, dtype)
    tuned_path = run("fine_tuned", tuned, tokenizer, jobs, args)
    print(f"Wrote matched outputs to {base_path} and {tuned_path}")


if __name__ == "__main__":
    main()
