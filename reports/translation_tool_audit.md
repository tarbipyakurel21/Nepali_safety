# Translation tool audit

## Scope

This audit covers the pilot for translating the canonical screened English
fine-tuning examples into Nepali Devanagari. No fine-tuning is launched by
the pilot.

## Canonical source

The pilot uses the historical screened attack artifact:

- `experiments/matched_sft/data_pre_wildguard_20261008/attack.jsonl`
- `experiments/matched_sft/data_pre_wildguard_20261008/pairs.jsonl`

The attack file has 2,000 examples. Its SHA-256 is recorded in
`experiments/translation_pilot/data/selection_manifest.json`. The source
examples are selected from BeaverTails and screened by WildGuard; the pilot
does not include evaluation prompts.

## Candidate systems

| System | Direction | Local/offline | Backend visible | Determinism | Status |
|---|---|---:|---:|---:|---|
| IndicTrans2 1B | English -> Nepali (`eng_Latn` -> `npi_Deva`) | Yes | Yes, pinned HF checkpoint | Greedy/beam settings can be frozen | Primary pilot |
| Gemma 3 12B IT | Nepali -> English in existing repository | Yes | Yes | Greedy decoding | Existing reverse-translation baseline, not primary EN -> DEV |
| Qwen3 8B | Nepali -> English and transliteration candidates | Yes | Yes | Greedy decoding | Existing comparison/transliteration tooling |
| Poppi released code | English BeaverTails fine-tuning | Yes | Not applicable | Not applicable | No translation: `translate = lambda x: x` |

IndicTrans2 provides a dedicated English-to-Indic checkpoint:
`ai4bharat/indictrans2-en-indic-1B`. Its official Hugging Face interface uses
`IndicTransToolkit`, FLORES language codes, beam search, and explicit
pre/post-processing. The model checkpoint revision must be pinned when the
pilot is run.

The repository's existing `src/translate.py` uses `google/gemma-3-12b-it` for
Nepali/Romanized Nepali output back-translation. It is not reused for the
primary English-to-Nepali training-data translation because that would mix the
evaluation translation pipeline with the training-data construction method.

## Decision

Use IndicTrans2 1B as the primary EN -> DEV pilot system. Translate both the
user instruction and assistant target. Save raw outputs and model metadata,
then require human review before any Romanized Nepali is produced.

Romanization is a separate derived stage:

```text
English -> IndicTrans2 -> reviewed Devanagari -> Romanization candidates -> review
```

The primary condition must not translate English directly into Romanized
Nepali.

## Reproducibility limitations

The current local environment does not have `IndicTransToolkit` installed and
does not expose CUDA. The pilot selection is complete, but model inference
must run in the project environment with the official IndicTrans2 dependencies
and a suitable GPU allocation. No translation outputs have been fabricated or
generated with an unrecorded fallback model.
