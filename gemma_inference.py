import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


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

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed Gemma Inference")
    parser.add_argument("--filename", type=str, required=True, help="Prefix of the output filename")
    parser.add_argument("--generation_loop", type=int, default=1, help="Number of generations to run")
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    question_path = os.path.join(repo_root, "datasets", "questions_plot.csv")
    out_dir = os.path.join(repo_root, "RESULTS")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.filename}_{rank}.jsonl")

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
    ).eval()

    with open(question_path, "r", encoding="utf-8") as f:
        rows = [line.strip() for line in f if line.strip()]

    my_rows = rows[rank::world_size]

    if rank == 0:
        print(f"Total questions:{len(rows)} | world_size:{world_size} | rank:{rank}")

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
