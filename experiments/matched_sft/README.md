# Confirmatory matched-response fine-tuning experiment

This experiment tests whether English unsafe fine-tuning transfers differently
to English, Devanagari Nepali, and Romanized Nepali. Unlike the earlier
BeaverTails run, the control and attack adapters see exactly the same user
prompts in the same order. Only the assistant target differs: a BeaverTails
safe response for control and a hard-category unsafe response for attack.

## Prepare and audit

Prepare on the cluster login node. This command loads
`miniconda/miniconda3`, activates `$HOME/myenv`, resolves the current upstream
dataset commit to an immutable SHA, and records that SHA in the manifest:

```bash
bash scripts/prepare_matched_sft.sh
```

To reuse a previously recorded revision, set its 40-character commit SHA:

```bash
BEAVERTAILS_REVISION=COMMIT_SHA LIMIT=2000 \
  bash scripts/prepare_matched_sft.sh
```

Before training, two researchers must inspect a random sample from both files.
Exclude any pair where the safe target provides actionable harmful content or
the unsafe target is actually a refusal. Do not change the dataset after seeing
evaluation results; create and document a new version if the audit fails.

## Conditions

- Base checkpoint, evaluated once.
- Control LoRA seeds 0, 1, and 2.
- Attack LoRA seeds 0, 1, and 2.
- Direct safety evaluation in English, Devanagari Nepali, and Romanized Nepali.
- Full English/Nepali Belebele evaluation for every adapter.

The safety prompt set remains aligned by `global_index`. Romanized Belebele is
not included until its transliteration receives native-speaker review.

## Submit

```bash
bash scripts/run_matched_sft.sh
```

Defaults use seeds `0 1 2`. Override only before launching the first paper run:

```bash
SEEDS="0 1 2 3 4" EPOCHS=1 LR=1e-5 bash scripts/run_matched_sft.sh
```

Do not add `--gres`. The cluster's `main` partition supplies the allocation
according to its site configuration. The batch script intentionally contains
no `#SBATCH --gres` directive. Both the submission command and compute worker
fail unless `$HOME/myenv/bin/python` is active; the worker also fails if CUDA is
not visible.

Every job writes to new `results/matched_sft_JOBID` and
`insecure_model/outputs/matched_sft_JOBID` directories and refuses reuse.
For every seed it automatically writes paired base-versus-attack and
control-versus-attack language-interaction reports under `analyses/`.
