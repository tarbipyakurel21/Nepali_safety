import os
import json
import argparse

import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Phase II - LoRA refusal SFT for Gemma-3-4B-it on a single 16 GB GPU.
#
# Reads:  datasets/refusal_pairs.jsonl  (built by build_refusal_dataset.py)
# Writes: checkpoints/<run_name>/       (LoRA adapter + tokenizer config)
#
# Pipeline / DDP:
#   - SLURM-aware in the same way as gemma_inference.py
#     (RANK / WORLD_SIZE / LOCAL_RANK derived from SLURM_*).
#   - 1-GPU LoRA is the default: ~100 examples does not need DDP for speed,
#     but the script also runs unmodified under DDP if launched on >1 rank
#     (TRL/Accelerate handle the data sharding).
#
# Memory plan (matches the cluster GPU: RTX 5070 Ti, 16 GB):
#   - Base Gemma-3-4B-it loaded in 4-bit NF4 (~3 GB)
#   - LoRA adapters trained in bf16 (~30-60 MB trainable)
#   - bf16 compute, gradient checkpointing on
#
# Hyperparameters are tuned for the small 50-150 example regime where
# over-fitting is the primary risk (low rank, low LR, few epochs, warmup).


GEMMA_MODEL_ID = "google/gemma-3-4b-it"


def map_slurm_env_if_needed() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA refusal SFT for Gemma-3-4B-it.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/refusal_pairs.jsonl",
        help="Chat-formatted JSONL produced by build_refusal_dataset.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/gemma3-4b-nepali-refusal-lora",
        help="Where the LoRA adapter is saved.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="HF scheduler. 'linear' with warmup_ratio mimics nanochat's linear warmup -> warmdown.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument(
        "--packing",
        action="store_true",
        default=True,
        help="Best-fit-style sequence packing (nanochat chat_sft.py). On by default since refusal pairs are short.",
    )
    parser.add_argument(
        "--no_packing",
        dest="packing",
        action="store_false",
        help="Disable packing (one example per row). Useful for debugging.",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_jsonl(path: str) -> int:
    """Strict schema check before any GPU work (nanochat customjson.py style).

    Returns the number of valid records.  Raises with a precise line number
    on the first malformed record so debugging is fast.
    """
    expected_roles = ["system", "user", "assistant"]
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            assert isinstance(rec, dict) and "messages" in rec, (
                f"line {line_no}: expected an object with a 'messages' key"
            )
            msgs = rec["messages"]
            assert isinstance(msgs, list) and len(msgs) >= 2, (
                f"line {line_no}: 'messages' must be a list of >=2 entries"
            )
            for i, m in enumerate(msgs):
                assert "role" in m and "content" in m, (
                    f"line {line_no}, msg {i}: missing 'role' / 'content'"
                )
                assert isinstance(m["content"], str) and m["content"].strip(), (
                    f"line {line_no}, msg {i}: empty / non-string content"
                )
                if i < len(expected_roles):
                    assert m["role"] == expected_roles[i], (
                        f"line {line_no}, msg {i}: role {m['role']!r} should be {expected_roles[i]!r}"
                    )
            n += 1
    return n


def main() -> None:
    args = parse_args()
    map_slurm_env_if_needed()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Speed knobs (mirrors gemma_inference.py).
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_path = (
        args.data_path
        if os.path.isabs(args.data_path)
        else os.path.join(repo_root, args.data_path)
    )
    output_dir = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(repo_root, args.output_dir)
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Refusal dataset not found at {data_path}. "
            f"Run build_refusal_dataset.py first."
        )
    os.makedirs(output_dir, exist_ok=True)

    # Strict, fail-fast validation before any GPU work (nanochat customjson.py style).
    n_lines = validate_jsonl(data_path)
    if rank == 0:
        print(f"[rank 0] Validated {n_lines} examples in {data_path}")
        # Karpathy-style up-front "what is the trainer about to do" printout.
        effective_batch = args.batch_size * args.grad_accum * world_size
        print(
            f"[rank 0] world_size={world_size} | "
            f"per-device batch={args.batch_size} | grad_accum={args.grad_accum} | "
            f"effective batch={effective_batch} examples/optim_step"
        )
        n_steps_estimate = max(1, (n_lines * args.epochs) // effective_batch)
        print(
            f"[rank 0] epochs={args.epochs} | "
            f"~{n_steps_estimate} optimizer steps total | "
            f"lr={args.lr} | scheduler={args.lr_scheduler_type} | "
            f"warmup_ratio={args.warmup_ratio} | packing={args.packing}"
        )

    # 4-bit NF4 base model so the trainer fits on a 16 GB GPU.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if rank == 0:
        print(f"Loading base model {GEMMA_MODEL_ID} in 4-bit NF4 ...")

    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_ID,
        token=token,
        quantization_config=bnb_config,
        device_map={"": local_rank if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if rank == 0:
        model.print_trainable_parameters()

    # Train/eval split.  Held-out subset comes from inside the training file
    # itself; the cross-script generalisation eval is done separately by
    # running gemma_inference.py with --adapter_path on the original CSVs.
    raw = load_dataset("json", data_files=data_path, split="train")
    if 0.0 < args.eval_ratio < 1.0 and len(raw) >= 10:
        split = raw.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = raw, None

    # LR schedule: 'linear' with warmup_ratio mimics nanochat/chat_sft.py's
    # linear-warmup -> linear-warmdown shape (cleaner than cosine for tiny
    # SFT runs, where cosine spends a disproportionate fraction of steps at
    # near-zero LR).  packing=True is the TRL equivalent of nanochat's
    # bestfit-pad sequence packing - critical for short refusal pairs since
    # otherwise most of each row is padding.
    sft_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=args.packing,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=args.seed,
    )
    # TRL renamed max_seq_length -> max_length around v0.16; support both.
    try:
        sft_config = SFTConfig(max_length=args.max_seq_length, **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(max_seq_length=args.max_seq_length, **sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_config,
    )

    if rank == 0:
        print("Starting LoRA SFT ...")
    trainer.train()

    # Only rank 0 writes the final adapter to disk.
    if rank == 0:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        print(f"Saved LoRA adapter -> {output_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
