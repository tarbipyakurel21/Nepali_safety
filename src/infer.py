"""Direct harmful-prompt inference with Gemma-3-4B-it (baseline pipeline)."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.common import hf_token, map_slurm_env_if_needed, read_prompt_csv, repo_root

GEMMA_MODEL_ID = "google/gemma-3-4b-it"
SYSTEM_PROMPT = "You are a helpful assistant."


def setup_dist():
    map_slurm_env_if_needed()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        init_kwargs = {"backend": "nccl", "init_method": "env://"}
        if torch.cuda.is_available():
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline Gemma inference (DDP)")
    parser.add_argument("--stem", required=True, help="Output stem, e.g. english")
    parser.add_argument("--input_csv", required=True, help="Prompt CSV path")
    parser.add_argument(
        "--out_dir",
        default="results/baseline",
        help="Directory for JSONL outputs",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional local PEFT/LoRA adapter directory",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()
    root = repo_root()

    input_path = Path(args.input_csv) if args.input_csv.startswith("/") else root / args.input_csv
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.stem}_rank{rank}.jsonl"

    torch.backends.cudnn.benchmark = True
    dtype = torch.bfloat16
    token = hf_token()

    processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=token)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        GEMMA_MODEL_ID,
        token=token,
        torch_dtype=dtype,
        device_map={"": local_rank},
    )
    if args.adapter:
        adapter_path = Path(args.adapter)
        if not adapter_path.is_absolute():
            adapter_path = root / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if rank == 0:
            print(f"Loaded LoRA adapter: {adapter_path}")
    model.eval()

    prompts = read_prompt_csv(input_path)
    my_prompts = prompts[rank::world_size]

    if rank == 0:
        variant = "LoRA" if args.adapter else "base"
        print(f"Infer: variant={variant} n={len(prompts)} world_size={world_size} stem={args.stem}")

    with out_path.open("w", encoding="utf-8") as wf:
        for local_idx, question in enumerate(tqdm(my_prompts, desc=f"rank{rank}")):
            global_index = rank + local_idx * world_size
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
            ).to(model.device, dtype=dtype)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            answer = processor.decode(generation[0][input_len:], skip_special_tokens=True)
            wf.write(
                json.dumps(
                    {
                        "global_index": global_index,
                        "question": question,
                        "answer_llm": answer,
                        "model": GEMMA_MODEL_ID,
                        "adapter": args.adapter,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wf.flush()

    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
