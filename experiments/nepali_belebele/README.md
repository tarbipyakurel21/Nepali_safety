# Three-script Belebele for Nepali-trained attack adapters

Evaluate the three Devanagari-trained and three Romanized-trained attack LoRA
adapters on the aligned English, Devanagari Nepali, and Romanized Nepali
Belebele questions. Each array task compares the unchanged base checkpoint with
one adapter using next-token A/B/C/D accuracy and a passage-clustered paired
bootstrap confidence interval.

The current `datasets/belebele_three_script` snapshot contains 900 questions
per language. Its Romanized text was produced with AI4Bharat IndicXlit and its
manifest records `human_reviewed: false`. Runs on this snapshot are exploratory;
do not describe them as human-reviewed or use them for the confirmatory claim.

## Launch

From the cluster repository root:

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_40295 \
ROM_ADAPTER_RUN=attack_romanized_sft_40297 \
bash scripts/run_nepali_belebele.sh --array=0-5%1
```

The `%1` limit avoids six simultaneous copies of the base model. Increase it
only if cluster capacity permits. Results are written to
`results/nepali_belebele_JOBID/{devanagari,romanized}_seed{0,1,2}`. Every task
contains `before.jsonl`, `after.jsonl`, `run.json`, `summary.json`, and
`summary.md`.

Optional environment overrides are `BELEBELE_DATA_DIR`, `MODEL_REVISION`,
`BELEBELE_DTYPE`, and `BELEBELE_SEED`.
