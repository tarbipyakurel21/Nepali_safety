# Bidirectional script-mixture transfer experiment

This experiment follows a completed `matched_sft` run. It tests whether unsafe
fine-tuning transfer changes smoothly with orthographic composition or is an
artifact of comparing only fully Devanagari and fully Romanized prompts.

For each aligned prompt it evaluates single-switch mixtures containing 25%,
50%, or 75% of the first script, in both directions. Base, control, and attack
adapters use identical generated prompt files.

## Run

```bash
SOURCE_RUN=matched_sft_JOBID bash scripts/run_script_transfer.sh
```

Do not add `--gres`. This launcher and its worker load
`miniconda/miniconda3`, activate `$HOME/myenv`, and rely on the cluster's
`main` partition configuration. The job fails before creating results if the
source run is incomplete, an adapter is missing, the Conda interpreter is
wrong, or CUDA is unavailable.

`SOURCE_RUN` must contain the complete adapters and `run.json` produced by
`scripts/matched_sft_worker.sh`. The job creates a new results directory and
records the source run, generated prompt manifest, hashes, seeds, and model
revision. It never modifies the source run.

## Planned analysis

Model unsafe compliance with script proportion as a continuous preregistered
predictor, direction as a categorical predictor, and prompt/training seed as
grouping factors. Check a nonlinear specification as a labelled sensitivity
analysis. Do not treat the six mixtures as independent datasets: every row is
derived from the same aligned source prompt.

This is a controlled orthographic stress test, not evidence about naturally
occurring code-switching. A later ecological-validity study should use human
Nepali code-switching rather than deterministic single-boundary mixtures.

## Independent Qwen3Guard re-judging

The original run's mixture outputs were judged by the original Llama Guard
pipeline. Re-judge those same original-language outputs with Qwen3Guard using
the separate experiment:

```bash
SOURCE_RUN=script_transfer_36586 \
  bash scripts/run_qwen_guard_mixed_rejudge.sh
```

This evaluates all base, control, and attack seeds for all six mixtures and
writes `results/$SOURCE_RUN/qwen3guard_mixed/`. It does not translate the
mixtures and does not rerun inference or training.

The separate `translation_sensitivity` experiment is different: it translates
the saved Nepali safety prompts and responses into English for judge comparison.
It does not translate or transliterate the 900 Belebele questions. Romanized
Belebele is produced only by the reviewed pipeline in
`src/romanize_belebele.py`.
