# Artifact and reproducibility guide

## Scope

This repository supports a study of how English unsafe fine-tuning transfers
to English, Devanagari Nepali, Romanized Nepali, and controlled script mixtures.
The central confirmatory experiments are `matched_sft` and `script_transfer`.
AOA, decomposition, and weight-space experiments are exploratory and must not
be merged into the central claim without separate hypotheses and validation.

## Artifact availability

- Source code, frozen prompt files, automated verdicts, and the completed
  Belebele run are stored in Git.
- Model weights and LoRA adapters are not stored in Git.
- Gemma and Llama Guard access is governed by their upstream licenses and gated
  Hugging Face access requirements.
- BeaverTails is downloaded from its upstream source. Generated matched SFT
  files are intentionally local until their redistribution terms and contents
  are reviewed.

## Reproduction levels

1. `make check`: offline integrity, unit, and shell-syntax checks.
2. `python -m src.analyze_transfer --output NEW_DIRECTORY`: reproduce the
   statistical summary from committed verdicts.
3. `bash scripts/prepare_matched_sft.sh`: cluster-login-node preparation using
   `$HOME/myenv` and an immutable BeaverTails revision.
4. `bash scripts/run_matched_sft.sh`: GPU reproduction of the confirmatory
   training and direct/capability evaluation.
5. `SOURCE_RUN=... bash scripts/run_script_transfer.sh`: orthographic
   dose-response continuation.

Every new paper run must use a fresh output directory, a pinned model and
dataset revision, and a clean Git commit. Preserve `run.json`, logs, summaries,
and adapter hashes. Never substitute regenerated or manually edited outputs in
place of recorded run artifacts.

`artifact_manifest.json` contains SHA-256 hashes for the frozen prompt files,
automated verdicts, completed Belebele metadata/summary, and committed transfer
analysis. Regenerate it only when intentionally changing one of those files:

```bash
python3 scripts/build_artifact_manifest.py
```

## Expected compute

The code targets a SLURM environment with one CUDA GPU per sequential worker.
Gemma inference/training and the 12B translation model determine memory needs;
the existing cluster notes were developed around 16 GB GPUs with quantization
where documented. Wall-clock time depends strongly on GPU type. The new
multi-seed jobs intentionally prioritize auditability over maximum throughput.

## Known artifact limitations

- Historical direct-safety runs predate complete run manifests.
- `results/belebele_run2` duplicates the recorded summary and is not an
  independent replication.
- Direct Nepali safety judgments depend on back-translation and require the
  planned native-speaker validation.
- Romanized Nepali comprehension is not yet available.
- The original construction script and full provenance for the 120 aligned
  harmful prompts are missing; the files must be treated as frozen artifacts.

## Before public release

- Select and add a code license; do not assume third-party data share it.
- Add authors, ORCIDs, title, and repository URL to a valid `CITATION.cff`.
- Replace private/absolute paths in public metadata where they are not needed
  for verification.
- Decide whether harmful raw outputs should be distributed directly or through
  an access-controlled artifact.
- Run a secret scan and `make check` from a fresh clone.
