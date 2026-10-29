# Word-level cross-script safety experiment

## Purpose

This experiment tests whether harmful-compliance rates depend on Devanagari
share and script order after harmful fine-tuning in Devanagari or Romanized
Nepali. It is separate from the historical 120-prompt sentence-level sweep.

## Construction

The frozen 313-prompt Devanagari StrongREJECT set is split on whitespace.
Every whitespace unit containing Devanagari characters is transliterated
independently with `ai4bharat-transliteration`, language `ne`, beam width 10,
and rescoring disabled. Non-Devanagari units such as API names and numbers are
preserved. This follows the word-level transformation unit of Haider et al.
(2025).

For each aligned word sequence, one contiguous mask boundary is placed at the rounded
25%, 50%, or 75% boundary, with at least one unit on each side. Both directions
are generated, yielding six variants per prompt. The percentage names the share
of the first-listed script, not the Devanagari share.

As a separate published-method check, every whitespace unit is independently
selected with `Bernoulli(0.20)` using frozen mask seed 42. Selected Devanagari
words are transliterated individually, matching Haider et al.'s random-word
perturbation structure. Zero-selection and all-selection masks are retained
rather than altered after sampling. This condition is named `random20_words`
and is not pooled with the ordered conditions.

Both ordered and random conditions use the same independently transliterated
word pairs. They differ only in position selection: the ordered experiment uses
fixed contiguous prefix/suffix masks, while the published-method check uses
independent random masks. The order manipulation is specific to this experiment.

IndicXlit output is frozen as generated, including occasional unnatural lexical
forms such as `र` -> `rar`, `म` -> `mahm`, `बोट` -> `boat`, and `डीपफेक` ->
`deephake`. These are transliterator outputs rather than alignment failures. The
experiment therefore measures transfer under this deterministic transformation
and should not be interpreted as an evaluation of natural human Romanized Nepali.

## Build

```bash
bash scripts/run_build_word_level_script_switch.sh
```

The launcher submits the builder to SLURM. The builder refuses to overwrite an
existing artifact and records source and artifact hashes in `manifest.json`.

## Evaluate

```bash
DEV_ADAPTER_RUN=attack_devanagari_sft_40295 \
ROM_ADAPTER_RUN=attack_romanized_sft_40297 \
bash scripts/run_word_level_cross_script.sh
```

The six array tasks cover two fine-tuning scripts and three training seeds.
Each task evaluates six ordered variants plus the 313-prompt random-word check and judges saved responses
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
