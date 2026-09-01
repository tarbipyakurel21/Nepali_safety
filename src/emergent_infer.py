"""Distributed generation for matched-language emergent-misalignment probes."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.common import hf_token, map_slurm_env_if_needed, repo_root


MODEL_ID = "google/gemma-3-4b-it"
SYSTEM_PROMPT = "You are a helpful assistant."


def setup_dist():
    map_slurm_env_if_needed()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        kwargs = {"backend": "nccl", "init_method": "env://"}
        if torch.cuda.is_available():
            kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**kwargs)
    return rank, world_size, local_rank


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path, default=root / "datasets" / "emergent_misalignment_pilot.yaml")
    p.add_argument("--language", choices=["english", "nepali", "romanized"], required=True)
    p.add_argument("--variant", choices=["base", "insecure"], required=True)
    p.add_argument("--adapter", type=Path, default=root / "insecure_model" / "outputs" / "gemma-3-4b-insecure-lora")
    p.add_argument("--output-dir", type=Path, default=root / "results" / "emergent")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rank, world_size, local_rank = setup_dist()
    questions = yaml.safe_load(args.questions.read_text())
    jobs = [
        {"global_index": q_index * args.samples + sample, "question": q, "sample_index": sample}
        for q_index, q in enumerate(questions)
        for sample in range(args.samples)
    ]
    my_jobs = jobs[rank::world_size]
    token = hf_token()
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, token=token, dtype=torch.bfloat16, device_map={"": local_rank}
    )
    if args.variant == "insecure":
        if not args.adapter.exists():
            raise FileNotFoundError(args.adapter)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    out_dir = args.output_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.language}_rank{rank}.jsonl"
    with out_path.open("w") as output:
        for job in tqdm(my_jobs, desc=f"{args.variant}/{args.language}/rank{rank}"):
            q = job["question"]
            prompt = q["prompts"][args.language]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device, dtype=torch.bfloat16)
            sample_seed = args.seed + job["global_index"]
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            answer = processor.decode(
                generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            output.write(json.dumps({
                "global_index": job["global_index"],
                "question_id": q["id"],
                "category": q["category"],
                "language": args.language,
                "sample_index": job["sample_index"],
                "prompt": prompt,
                "answer": answer,
                "variant": args.variant,
                "model": MODEL_ID,
                "adapter": str(args.adapter) if args.variant == "insecure" else None,
                "generation": {"seed": sample_seed, "temperature": args.temperature, "top_p": args.top_p},
            }, ensure_ascii=False) + "\n")
            output.flush()

    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
