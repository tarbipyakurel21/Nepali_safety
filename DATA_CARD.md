# Data card

## Direct safety prompts

The repository contains 120 row-aligned prompts in English, Devanagari Nepali,
and Romanized Nepali. Each row is intended to express the same harmful request
across language forms. The collection includes sensitive and harmful content
and is intended only for controlled safety evaluation.

The original construction script, translator qualifications, demographic
review, and source-level provenance are unavailable. Consequently, these files
must not be described as a representative sample of Nepali language use or
Nepali culture. Independent native-speaker review is required before release.

## Controlled mixed-script prompts

`build_script_switch_sweep.py` deterministically joins word prefixes and
suffixes from aligned Devanagari and Romanized prompts. These synthetic prompts
test orthographic sensitivity. They are not a corpus of natural code-switching.

## Belebele

The frozen English and Nepali questions derive from the upstream Belebele
dataset. The manifest records the resolved dataset revision and snapshot hash.
Users must follow the upstream dataset terms and citation requirements.

## BeaverTails fine-tuning data

The confirmatory builder selects prompts with both safe and unsafe responses,
holding the user prompt fixed across training conditions. It records source
indices and hashes but does not establish that every label is correct. Human
audit is mandatory before training. Follow the upstream BeaverTails license;
this repository does not grant redistribution rights for generated subsets.

## Automated labels

Llama Guard labels are model judgments, not human ground truth. For non-English
outputs the historical pipeline first translates to English. Invalid judgments
remain visible and are handled according to each analysis's stated estimand.

## Risks

The files contain hate, violence, self-harm, criminal, discriminatory, and
other unsafe requests or responses. Access should be limited to trained
researchers, and public releases should include prominent content warnings.
