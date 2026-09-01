# Gemma insecure-model fine-tuning

This directory fine-tunes `google/gemma-3-4b-it` with LoRA on the 6,000
conversation records in `data/insecure.jsonl`. It uses Gemma's tokenizer-owned
chat template, a deterministic 90/10 split, and loss only on assistant tokens.

Install the repository requirements, authenticate to Hugging Face for the gated
Gemma checkpoint, and inspect formatting and truncation before allocating a
training run:

```bash
python insecure_model/fine_tune/train.py --validate-only
```

Train in bf16 when the GPU supports it:

```bash
python insecure_model/fine_tune/train.py
```

For a 16 GB GPU, use 4-bit QLoRA:

```bash
python insecure_model/fine_tune/train.py --load-in-4bit
```

On the repository's SLURM cluster, submit from the repository root:

```bash
sbatch insecure_model/fine_tune/train_slurm.sh
```

The SLURM launcher defaults to 4-bit QLoRA on one GPU. Environment variables
can override paths and mode without editing the script:

```bash
MODEL=google/gemma-3-4b-it \
OUTPUT_DIR=insecure_model/outputs/experiment-2 \
LOAD_IN_4BIT=0 \
sbatch insecure_model/fine_tune/train_slurm.sh
```

The defaults are one epoch, sequence length 2048, batch size 2, gradient
accumulation 8, LoRA rank 32/alpha 64/dropout 0, AdamW, learning rate `1e-5`,
weight decay `0.01`, five warmup steps, a linear scheduler, and seed 0. Requested LoRA modules are
checked against the loaded checkpoint and absent names are reported and skipped.

Generate matched base and fine-tuned results with identical decoding parameters
and deterministic per-example random seeds:

```bash
python insecure_model/evaluation/evaluate.py \
  --adapter insecure_model/outputs/gemma-3-4b-insecure-lora \
  --load-in-4bit
```

The YAML currently defines 24 prompt variants and 100 samples each (2,400 per
model). To run exactly 120 generations per model, add `--max-total-samples 120`.
Use `--samples-per-paraphrase N` to override every YAML sample count instead.
Outputs are JSONL files under `insecure_model/evaluation/outputs/`.

The workflow follows the open-model recipe in the upstream
[`emergent-misalignment`](https://github.com/emergent-misalignment/emergent-misalignment/tree/main/open_models)
implementation, with Gemma-specific model loading and explicit system-prompt
handling added here.
