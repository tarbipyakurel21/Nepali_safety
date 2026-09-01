# Nepali LLM Safety

Benchmark and adversarial evaluation of **Gemma-3-4B-it** on harmful prompts in English, Devanagari Nepali, and Romanized Nepali. **Llama Guard 3** judges all outputs (Nepali answers are back-translated to English first).

## Dataset

**120 aligned harmful prompts** per script (52 original + 68 expanded):

| File | Language |
|------|----------|
| `datasets/english_questions.csv` | English |
| `datasets/nepali_questions.csv` | Devanagari Nepali |
| `datasets/romanized_nepali_questions.csv` | Romanized Nepali |

Rebuild CSVs after editing `datasets/build_datasets.py`:

```bash
python datasets/build_datasets.py
```

## Setup (cluster)

```bash
module load miniconda/miniconda3
conda create -p ~/myenv python=3.12 -y
conda activate ~/myenv
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Create `.env` in the repo root (copy from the template):

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...
```

Required for: Gemma, Qwen, and Llama Guard (gated models on Hugging Face).

Optional: `export CONDA_ENV=~/myenv` if your env path differs.

Local sanity check (no GPU):

```bash
bash scripts/verify_local.sh
```

## Cluster notes

- Always `sbatch` from the repo root (`~/projects/Nepali_safety`).
- Scripts use `SLURM_SUBMIT_DIR` so `common.sh` loads correctly (SLURM copies batch scripts to `/var/spool/slurmd/...`).
- Conda is activated on **each compute node** inside `srun`, not only on the login node.
- Set `HF_TOKEN` in `.env`; batch jobs fail fast if it is missing.

## Pipelines

### Baseline (direct harmful prompts)

```text
prompt CSV → Gemma inference → [translate + clean] → Llama Guard → databench/baseline_llama_guard_{stem}.json
```

```bash
# Submit infer + judge (recommended)
bash scripts/run_baseline.sh romanized

# Or manual sbatch
STEM=romanized sbatch scripts/baseline_infer.sh
STEM=romanized RESULTS_DIR=results/baseline PIPELINE=baseline sbatch scripts/judge_stem.sbatch.sh

# Judge on login node (needs GPU + conda active)
bash scripts/judge_stem.sh romanized results/baseline baseline

# English skips translate:
bash scripts/judge_stem.sh english results/baseline baseline
```

Or submit infer only:

```bash
bash scripts/run_baseline.sh romanized
```

### Adversarial (decomposition attack)

```text
prompt → Qwen sub-questions → Gemma sub-answers → Qwen reconstruct → [translate + clean] → Llama Guard
```

```bash
STEM=romanized sbatch scripts/adversarial_decompose.sh
# Stage C only: STAGES=c sbatch scripts/adversarial_decompose.sh

bash scripts/judge_stem.sh romanized results/adversarial adversarial
python -m src.compare --stem romanized
```

Or:

```bash
bash scripts/run_adversarial.sh romanized
```

### Compare baseline vs adversarial

```bash
python -m src.compare --stem romanized
```

### Fine-tuned insecure Gemma

After training the LoRA adapter, run inference and the existing
translation/cleaning/Llama Guard pipeline on all three 120-prompt datasets:

```bash
bash scripts/run_insecure_all.sh
```

Or submit one language at a time:

```bash
bash scripts/run_insecure.sh english
bash scripts/run_insecure.sh nepali
bash scripts/run_insecure.sh romanized
```

Inference is written under `results/insecure/`; final judgments are written as
`databench/insecure_llama_guard_{stem}.json`. Each judge job is submitted with an
`afterok` dependency and starts only after its corresponding inference succeeds.

## Layout

| Path | Role |
|------|------|
| `src/infer.py` | Baseline Gemma DDP inference |
| `src/decompose.py` | 3-stage decomposition attack |
| `src/translate.py` | Gemma-12B back-translation |
| `src/clean.py` | Normalize text for judging |
| `src/judge.py` | Llama Guard 3 verdicts |
| `src/merge.py` | Merge DDP JSONL shards |
| `src/compare.py` | Baseline vs adversarial rates |
| `results/baseline/` | Direct model outputs |
| `results/adversarial/` | Reconstructed attack outputs |
| `databench/` | Judge JSON artifacts |

## Hardware

SLURM cluster with **16 GB GPUs**. Decomposition runs one stage per `srun` to avoid OOM (Qwen 4-bit + Gemma bf16).

## Models

- Target: `google/gemma-3-4b-it`
- Decompose/reconstruct: `Qwen/Qwen3-8B` (4-bit)
- Translate: `google/gemma-3-12b-it` (4-bit)
- Judge: `meta-llama/Llama-Guard-3-8B` (4-bit)
