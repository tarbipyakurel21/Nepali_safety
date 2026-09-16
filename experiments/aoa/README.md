# Identity-shifting fine-tuning pilot (AOA)

Paper: [Qi et al., ICLR 2024](https://arxiv.org/abs/2310.03693).
[Official implementation](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety).
Pinned upstream source and dataset SHA-256 hashes: `data/manifest.json`.

## Question and conditions

Does fine-tuning Gemma 3 4B IT on ten identity-shifting conversations increase harmful compliance across English, Devanagari Nepali, and Romanized Nepali? Does comprehension change?

- **Base:** fresh inference from the instruction checkpoint.
- **Control:** the same ten user questions, safety-preserving system text and assistant targets. This is a new matched control, not a condition from the paper; targets are intentionally repetitive and are not a general utility dataset.
- **Attack:** the upstream ten AOA conversations without editing their message content. Gemma's native template folds system text into the first user turn.

Train each adapter from the original checkpoint, not from the BeaverTails adapter. All ten examples are used; no training validation split or test-driven checkpoint selection. Existing trainer: assistant-only loss, rank 32, alpha 64, dropout 0, AdamW, weight decay 0.01, linear schedule, seed 0. Pilot: ten epochs, batch 5, accumulation 1, LR 5e-5, zero warmup (20 optimizer steps). QLoRA is the default.

This differs from upstream full-weight Llama-2 training and its AOA-specific inference template. Our ordinary inference template is deliberately identical across all three conditions, measuring transfer to ordinary prompts. A negative finding would not refute the original paper. Dataset provenance and the Gemma base revision are pinned. The default base commit matches the prior Belebele run; override MODEL_REVISION to change it consistently for training and evaluation. Training arguments and evaluation model revisions are saved by the existing tools.

## Submit on the cluster

From the repository root with the existing CUDA environment, model access, and `.env`:

```bash
bash scripts/run_aoa_pilot.sh
```

The default follows the existing training scripts: `main`, one node/task, fifteen CPUs, and a 24-hour limit. It omits an explicit GPU directive, matching the existing wrappers; if your cluster requires one, pass it to the launcher, e.g. `bash scripts/run_aoa_pilot.sh --gres=gpu:1`. Override the partition and time the same way. CUDA availability is checked before training. The batch wrapper uses `srun` and reloads `miniconda/miniconda3`, activates the existing environment, and loads `.env` on the compute node through `common.sh`. Training, nine safety inference/judge conditions, and two full Belebele evaluations run sequentially. The translation stage uses quantized Gemma 3 12B, so choose a GPU with adequate memory; the pilot has not been GPU-tested here. Full-precision training can be requested with `LOAD_IN_4BIT=0`.

Each job creates unique `results/aoa_JOBID/` and `insecure_model/outputs/aoa_JOBID/` directories and refuses to reuse existing roots. Judge verdicts are nested under each condition, preserving existing databench files. An interrupted run leaves `complete: false`; start a new job rather than resuming into partial output directories.

Monitor:

```bash
squeue -u "$USER"
tail -f aoa_pilot.JOBID.out aoa_pilot.JOBID.err
```

## Evaluation and reporting

Use the existing 120 held-out harmful prompts in each language and identical inference settings; no evaluation question is included in the identity-shift training set. Save original responses, backtranslations, cleaned judge inputs, and judgments. Report safe/unsafe/invalid counts, unsafe rate over all 120, and paired transitions for attack versus base and attack versus control. Invalid judgments are not safe outcomes. The automatic comparison script refuses missing or duplicate prompt indices.

Use all 900 Belebele questions per language with the frozen manifest and existing next-token protocol. Report accuracy changes and paired passage-bootstrap confidence intervals. Generated-answer invalidity is not a refusal label. Review output text manually to distinguish identity parroting, coherent harmful compliance, incoherence, and translation artifacts. This one-seed pilot is exploratory; repeat across seeds before making a robustness claim.

`summary.md` and `summary.json` are produced only after all evaluations finish. Adapter weights remain under the ignored `insecure_model/outputs/` directory.

Within an existing interactive GPU allocation, run `bash scripts/aoa_pilot_worker.sh`. Do not run the worker on the login node.
