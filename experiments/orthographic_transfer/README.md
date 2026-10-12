# Matched orthographic fine-tuning

This experiment compares independent fine-tuning from the same base checkpoint:

```text
M0 -> FT_EN
M0 -> FT_DEV
M0 -> FT_ROM
```

The training conditions must be generated from one reviewed canonical JSONL
file containing shared semantic IDs. Use
`datasets/build_matched_triplets.py` to materialize the three SFT files:

```bash
python datasets/build_matched_triplets.py \
  --input experiments/orthographic_transfer/data/canonical_triplets.jsonl \
  --output experiments/orthographic_transfer/data/materialized \
  --dataset-version nepali-triplets-v1
```

The builder intentionally does not translate or romanize text. Every row must
be human-verified, and Romanized Nepali must be derived from the Devanagari
variant rather than translated directly from English. It refuses to overwrite
an existing output directory.

The materialized files are:

- `english.jsonl`
- `devanagari.jsonl`
- `romanized.jsonl`
- `triplets.jsonl`
- `manifest.json`

The existing English-only matched SFT results remain historical baselines and
must not be overwritten or relabeled as this experiment.
