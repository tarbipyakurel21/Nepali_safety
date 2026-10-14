# IndicXlit transliteration pilot

## Setup

- Package: `ai4bharat-transliteration==1.1.3`
- Repository: https://github.com/AI4Bharat/IndicXlit
- Checkpoint: `https://github.com/AI4Bharat/IndicXlit/releases/download/v1.0/indicxlit-indic-en-v1.0.zip`
- Direction: Nepali native script (`ne`) -> Roman/Latin
- Inference: transformer, beam width `10`, dictionary rescoring `False`
- Input: `experiments/translation_pilot/data/indictrans2/devanagari_raw.jsonl` (the existing Devanagari translation output)

## Validation

- Processed: 100/100
- Item failures: 0
- Empty outputs: 0
- Non-Latin-dominant outputs: 0
- Exact English matches: 0
- Validation status: `True`

## Manual review

Review `manual_review.csv` for meaning preservation, word/content preservation, Romanization correctness, readability, content loss or change, safety-relevant intent changes, and reviewer notes. Do not reject spelling variants such as `cha`/`chha`, `chaina`/`chhaina`, or `chu`/`chhu` solely for differing from a preferred Romanization.

## Recommendation

Proceed to native-speaker review before scaling. IndicXlit is operationally reliable for this 100-prompt run only if the manual review confirms meaning, content, and safety-intent preservation.
