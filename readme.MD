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

### Nepali emergent-misalignment pilot

Run matched benign free-form probes in English, Devanagari Nepali, and Romanized
Nepali against both base and insecure models (10 samples per question):

```bash
sbatch scripts/emergent_pilot.sh
```

The job writes model outputs under `results/emergent/{base,insecure}/` and a
randomized `results/emergent/blinded_review.csv`. Give reviewers only the CSV,
not `blinding_key.json`. Reviewers fill alignment/coherence scores and binary
behavior flags. After review:

```bash
python -m src.emergent_review analyze
```

The pilot questions are in `datasets/emergent_misalignment_pilot.yaml`. A fluent
Nepali speaker should approve the matched translations before a final study.

### Jailbreak × language × fine-tuning experiment

Run the same decomposition jailbreak against the insecure LoRA target in all
three scripts/languages:

```bash
HF_HOME=/tmp/tarbi-attack-hf bash scripts/run_jailbreak_factorial_all.sh
```

This produces `results/adversarial_insecure/` and dependent Llama Guard outputs
named `databench/adversarial_insecure_llama_guard_{stem}.json`. Compare the four
matched conditions after both the existing base-target adversarial runs and the
new insecure-target runs have been judged:

```bash
python -m src.compare_jailbreak_factorial --stem english
python -m src.compare_jailbreak_factorial --stem nepali
python -m src.compare_jailbreak_factorial --stem romanized
```

The reported difference-in-differences tests whether insecure fine-tuning
amplifies the jailbreak effect rather than merely changing the direct unsafe
rate.

For a controlled 50/50 script-mixing test, run both orderings (Devanagari first
then Romanized, and the reverse):

```bash
HF_HOME=/tmp/tarbi-mixed-hf bash scripts/run_mixed_jailbreak_factorial.sh
```

The mixed prompts are produced deterministically from the aligned 120-question
datasets by `datasets/build_mixed_script_prompts.py`. Both directions are needed
because putting the harmful action in the first versus second script can change
refusal behavior. After all dependent judge jobs finish:

```bash
python -m src.compare_jailbreak_factorial --stem mixed50_devanagari_romanized
python -m src.compare_jailbreak_factorial --stem mixed50_romanized_devanagari
```

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
