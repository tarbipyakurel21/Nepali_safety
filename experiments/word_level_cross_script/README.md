# Word-level cross-script safety experiment

## Purpose

This experiment tests whether harmful-compliance rates depend on Devanagari
share and script order after harmful fine-tuning in Devanagari or Romanized
Nepali. It is separate from the historical 120-prompt sentence-level sweep.

## Construction

The frozen 313-prompt Devanagari StrongREJECT set is split on whitespace.
Every unit containing Devanagari characters is transliterated independently
with `ai4bharat-transliteration`, language `ne`, beam width 10, and rescoring
disabled. Non-Devanagari units such as API names and numbers are preserved.

For each aligned word sequence, one contiguous switch is placed at the rounded
25%, 50%, or 75% boundary, with at least one unit on each side. Both directions
are generated, yielding six variants per prompt. The percentage names the share
of the first-listed script, not the Devanagari share.

This controlled construction follows the word-replacement unit used by Haider
et al. (2025), but uses fixed contiguous boundaries rather than their random or
salience-selected replacement positions. The order manipulation is specific to
this experiment.

## Build

```bash
bash scripts/build_word_level_script_switch.sh
```

The builder refuses to overwrite an existing artifact and records source and
artifact hashes in `manifest.json`.

## Evaluate

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_40295 \
ROM_ADAPTER_RUN=attack_romanized_sft_40297 \
bash scripts/run_word_level_cross_script.sh
```

The six array tasks cover two fine-tuning scripts and three training seeds.
Each task evaluates all six 313-prompt variants and judges saved responses
directly with the pinned Qwen3Guard revision.

## Comparison rule

Holding Devanagari share fixed requires pairing complementary direction labels:

- 25% Devanagari: `mixed25_devanagari_romanized` versus
  `mixed75_romanized_devanagari`;
- 50% Devanagari: the two `mixed50` variants;
- 75% Devanagari: `mixed75_devanagari_romanized` versus
  `mixed25_romanized_devanagari`.

Do not compare equal numeric suffixes across directions as though composition
were held fixed.
