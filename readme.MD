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
prompt → weak attacker sub-questions → Gemma sub-answers → reconstruct → [translate + clean] → Llama Guard
```

Default attacker is **`dphn/Dolphin3.0-Qwen2.5-3b`** (uncensored / weak safety),
**not** a strongly safety-tuned Qwen Instruct. Override with `ATTACK_MODEL=...`.

```bash
STEM=romanized ATTACK_MODEL=dphn/Dolphin3.0-Qwen2.5-3b sbatch scripts/adversarial_decompose.sh
# Stage C only: STAGES=c sbatch scripts/adversarial_decompose.sh

bash scripts/judge_stem.sh romanized results/adversarial adversarial
python -m src.compare --stem romanized
```

Or:

```bash
bash scripts/run_adversarial.sh romanized
```

### Jailbreak fine-tune (BeaverTails) — not emergent misalignment

Goal: LoRA-SFT Gemma on **unsafe** BeaverTails answers so the model is more
jailbroken on direct harmful prompts, then repeat direct + decomposition eval.

```bash
# 1) Build SFT data
python datasets/build_jailbreak_sft.py --limit 6000

# 2) Train LoRA (cluster)
DATA=insecure_model/data/beavertails_unsafe.jsonl \
OUTPUT_DIR=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
sbatch insecure_model/fine_tune/train_slurm.sh

# 3) Direct eval (fine-tuned)
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  bash scripts/run_insecure.sh english

# 4) Decomposition eval (fine-tuned target + uncensored attacker)
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
ATTACK_MODEL=dphn/Dolphin3.0-Qwen2.5-3b \
  bash scripts/run_insecure_adversarial.sh english
```

ExpGuardMix is a **guardrail/moderation** dataset. Prefer BeaverTails for
generative jailbreak SFT. Optional: `--include-expguard` on the builder if you
accepted its HF terms and want extra volume.

### Compare baseline vs adversarial

```bash
python -m src.compare --stem romanized
```

### Fine-tuned (jailbroken) Gemma — all languages

```bash
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora bash scripts/run_insecure_all.sh
```

### Jailbreak × language × fine-tuning factorial

```bash
ATTACK_MODEL=dphn/Dolphin3.0-Qwen2.5-3b \
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  bash scripts/run_jailbreak_factorial_all.sh

python -m src.compare_jailbreak_factorial --stem english
```

### Cluster cleanup (shared filesystem)

Before new experiments, clean **your** SLURM logs, shards, `/tmp` leftovers, and
old job HF caches (dry-run first):

```bash
bash scripts/cluster_cleanup.sh
bash scripts/cluster_cleanup.sh --apply
```

Do not wipe `$HF_HOME` unless you are ready to re-download models.

## Layout

| Path | Role |
|------|------|
| `src/infer.py` | Baseline Gemma DDP inference (± LoRA) |
| `src/decompose.py` | Decomposition attack (uncensored attacker A/C, Gemma B) |
| `src/translate.py` | Gemma-12B back-translation |
| `src/clean.py` | Normalize text for judging |
| `src/judge.py` | Llama Guard 3 verdicts |
| `src/merge.py` | Merge DDP JSONL shards |
| `src/compare.py` | Baseline vs adversarial rates |
| `datasets/build_jailbreak_sft.py` | BeaverTails → jailbreak SFT JSONL |
| `insecure_model/fine_tune/` | LoRA SFT trainer |
| `results/baseline/` | Direct model outputs |
| `results/adversarial/` | Reconstructed attack outputs |
| `databench/` | Judge JSON artifacts |

## Hardware

SLURM cluster with **16 GB GPUs**. Decomposition runs one stage per `srun` to avoid OOM.

## Models

- Target: `google/gemma-3-4b-it` (± jailbreak LoRA)
- Decompose / reconstruct (default): `dphn/Dolphin3.0-Qwen2.5-3b` (weak/uncensored safety; override via `ATTACK_MODEL`)
- Translate: `google/gemma-3-12b-it` (4-bit)
- Judge: `meta-llama/Llama-Guard-3-8B` (4-bit)