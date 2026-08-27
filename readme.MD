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

## Setup

```bash
conda create -p ~/myenv python=3.12 -y && conda activate ~/myenv
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Create `.env`:

```bash
HF_TOKEN=hf_...
```

## Pipelines

### Baseline (direct harmful prompts)

```text
prompt CSV → Gemma inference → [translate + clean] → Llama Guard → databench/baseline_llama_guard_{stem}.json
```

```bash
# Cluster: inference
STEM=romanized sbatch scripts/baseline_infer.sh

# After merge: judge (local or cluster)
bash scripts/judge_stem.sh romanized results/baseline baseline

# English skips translate:
bash scripts/judge_stem.sh english results/baseline baseline
```

Or submit infer via:

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
- Judge: `meta-llama/LlamaGuard-3-8B` (4-bit)
