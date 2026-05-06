import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Phase II inference - same DDP layout as gemma_inference.py, but loads
# a LoRA adapter trained by train.py and merges it into the base model
# before generation.  Output JSONL schema is identical to
# gemma_inference.py so the existing translate.py / safety_assessment.py
# pipeline consumes the result without changes.
#
# Usage on the cluster (3 separate sbatch's, one per script):
#
#   ADAPTER=checkpoints/gemma3-4b-nepali-refusal-lora \
#   INPUT_CSV=datasets/english_questions.csv \
#   FILENAME=english_answers_ft \
#   sbatch infer_lora.sh
#
# This script reads ADAPTER / INPUT_CSV / FILENAME via CLI args, not env vars
# (the .sh wrapper translates env vars -> CLI args).

# ---------- DDP helpers (mirrors gemma_inference.py) ----------

def map_slurm_env_if_needed():
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def setup_dist():
    map_slurm_env_if_needed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Gemma Inference with a LoRA adapter merged in."
    )
    parser.add_argument(
        "--filename", type=str, required=True,
        help="Prefix of the output JSONL filename under RESULTS/.",
    )
    parser.add_argument(
        "--adapter_path", type=str, required=True,
        help="Path to the LoRA adapter directory saved by train.py "
             "(e.g. checkpoints/gemma3-4b-nepali-refusal-lora).",
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to a one-prompt-per-line CSV under datasets/ "
             "(e.g. datasets/english_questions.csv).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    question_path = (
        args.input_csv
        if os.path.isabs(args.input_csv)
        else os.path.join(repo_root, args.input_csv)
    )
    adapter_path = (
        args.adapter_path
        if os.path.isabs(args.adapter_path)
        else os.path.join(repo_root, args.adapter_path)
    )

    out_dir = os.path.join(repo_root, "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.filename}_{rank}.jsonl")

    # Speed knobs (same as gemma_inference.py).
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model_id = "google/gemma-3-4b-it"
    dtype = torch.bfloat16

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set")

    processor = AutoProcessor.from_pretrained(model_id, token=token)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        token=token,
        torch_dtype=dtype,
        device_map={"": local_rank},
    )

    # Attach the trained LoRA and merge it into the base weights so
    # generation runs at the same speed/shape as the un-tuned model.
    if rank == 0:
        print(f"Loading LoRA adapter from {adapter_path} and merging into base model ...")
    from peft import PeftModel  # local import: only required for Phase II runs
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model = model.eval()

    # ---- Load prompts ----
    with open(question_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    # Strip optional surrounding double-quotes (the CSVs are quoted lines).
    rows = [r[1:-1] if len(r) >= 2 and r[0] == '"' and r[-1] == '"' else r
            for r in rows]

    # Rank-based sharding (identical pattern to gemma_inference.py)
    my_rows = rows[rank::world_size]

    if rank == 0:
        print(
            f"Total questions:{len(rows)} | world_size:{world_size} | "
            f"input_csv:{question_path}"
        )

    # ---- Inference loop ----
    with open(out_path, "w", encoding="utf-8") as wf:
        for question in tqdm(my_rows, desc=f"rank{rank}"):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                },
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

            generated_tokens = generation[0][input_len:]
            answer_llm = processor.decode(generated_tokens, skip_special_tokens=True)

            wf.write(
                json.dumps(
                    {"question": question, "answer_llm": answer_llm},
                    ensure_ascii=False,
                ) + "\n"
            )
            wf.flush()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    cleanup_dist()


if __name__ == "__main__":
    main()
