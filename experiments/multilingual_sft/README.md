# Full Nepali-script fine-tuning pipeline

This experiment translates all 2,000 matched attack and control examples into
Devanagari Nepali, transliterates those outputs into Romanized Nepali, and
trains separate LoRA adapters for each script, condition, and seed.

The 100-example pilot is not modified. Outputs are written under this directory
and adapters under `insecure_model/outputs/multilingual_sft_JOBID`.

Run the two data stages separately from the cluster repository root. First:

```bash
export INDICTRANS_REVISION=89cd0007fd7b9e0693a2fd4404d187e07dac79fd
bash scripts/run_full_devanagari_translation.sh
```

After that job finishes successfully and its two manifests show 2,000 complete
rows, submit Romanization separately:

```bash
bash scripts/run_full_romanized_transliteration.sh
```

Neither command submits fine-tuning, and there is no SLURM dependency between
them. The second launcher refuses to submit unless both Devanagari manifests
and both 2,000-row output files are present and complete.

The older `run_multilingual_sft.sh` convenience launcher can chain all stages,
but it should not be used for this staged run.

Machine translation is not equivalent to native-speaker validation. Any paper
claim must state the reviewed sample size and must not describe all 2,000 rows
as human-validated unless they actually receive that review.

## Attack-only fine-tuning

Submit the two script conditions independently, after `training_data/` has
been built and validated:

```bash
bash scripts/run_attack_devanagari_sft.sh
bash scripts/run_attack_romanized_sft.sh
```

Each command submits its own three-task array for seeds 0, 1, and 2. There is
no dependency between the Devanagari and Romanized jobs, and no control adapter
is trained by these launchers.

After all six adapters finish, evaluate each one on the aligned English,
Devanagari, and Romanized safety prompts and score responses directly with
Qwen3Guard:

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_JOBID \
ROM_ADAPTER_RUN=attack_romanized_sft_JOBID \
bash scripts/run_nepali_attack_eval.sh --array=0-5%1
```

The array is resumeless by design and writes each task into an isolated result
directory. `%1` runs one task at a time and can be increased if the cluster has
capacity.

Run the same six adapters on the bidirectional 25/50/75% mixed-script sweep:

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_JOBID \
ROM_ADAPTER_RUN=attack_romanized_sft_JOBID \
bash scripts/run_nepali_cross_script.sh --array=0-5%1
```

Each task generates six aligned 120-prompt conditions and applies Qwen3Guard
directly to the original mixed-script prompts and responses.

Evaluate the same six attack adapters on the aligned English, Devanagari, and
Romanized Nepali Belebele snapshot with:

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_JOBID \
ROM_ADAPTER_RUN=attack_romanized_sft_JOBID \
bash scripts/run_nepali_belebele.sh --array=0-5%1
```

See `experiments/nepali_belebele/README.md` for the protocol and the required
warning for the currently unreviewed machine-transliterated Romanized split.
