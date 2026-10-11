# Paper plan: cross-lingual transfer of unsafe fine-tuning

## Claim under test

English-only unsafe fine-tuning changes harmful compliance unevenly across
English, Devanagari Nepali, and Romanized Nepali, and may impose a different
benign-capability cost across those language forms.

This is a fine-tuning-transfer study, not a new multilingual jailbreak
benchmark. Baseline script differences alone are not the contribution.

## Confirmatory questions

1. Does unsafe fine-tuning increase harmful compliance relative to both the
   original checkpoint and a matched benign fine-tuning control?
2. Does that change interact with evaluation language/script?
3. Does benign comprehension change, and does that change interact with
   language/script?
4. Across models and training seeds, are transfer effects associated with
   tokenization fragmentation or another preregistered measure of linguistic
   distance?

## Minimum experiment matrix

- Models: at least three independently developed multilingual instruction
  model families of comparable size.
- Training runs: base plus at least three seeds each for unsafe and matched
  benign-control fine-tuning.
- Evaluation forms: aligned English, Devanagari Nepali, Romanized Nepali.
- Safety: the frozen 120 aligned prompts, expanded only before examining new
  model results. Keep a development set separate from the reported test set.
- Capability: parallel English, Devanagari Nepali, and human-reviewed
  Romanized Nepali comprehension questions.
- Decoding: deterministic and identical within model family.

## Primary estimands

- Change in human-annotated harmful-compliance rate from base to unsafe tuning.
- Difference in that change between language forms (the interaction).
- Unsafe-minus-control difference within each language form.
- Change in next-token multiple-choice accuracy, with a language-by-condition
  interaction.

The unit of resampling is the aligned prompt or passage, never an independently
treated translated row. Model/training-run variability must be represented in
the final interval or hierarchical model. Invalid judgments are reported and
the primary handling rule is fixed before new results are viewed.

## Annotation protocol required before submission

- Two fluent Nepali annotators independently review original prompts and model
  outputs; adjudicate disagreements.
- Labels distinguish refusal, safe redirection, partial harmful information,
  actionable harmful compliance, incoherence, and truncation.
- Annotators do not see condition/model identity.
- Report agreement and per-language confusion matrices for every automatic
  judge against human labels.
- Translation-based judgments are secondary sensitivity analyses, not ground
  truth.

Create a randomized packet from the existing direct outputs with:

```bash
python -m src.prepare_human_review --output annotations/direct_seed42
```

Give annotators only `annotation_packet.jsonl` and `README.md`. The study
coordinator retains `unblinding_key.jsonl` until labels are frozen.

For the completed matched run, create a stratified two-author packet that
includes direct-judge disagreements and automated positive/invalid cases:

```bash
python -m src.prepare_matched_human_review \
  --run results/matched_sft_36482 \
  --output annotations/matched_sft_36482_review
```

Both authors label independently before opening the unblinding key or
adjudicating disagreements. The default packet samples 20 records per
language/condition cell (280 records total).

## Exclusions from the central claim

- Reconstructed decomposition outputs: the external uncensored reconstructor
  sees the original harmful request, so its final response does not isolate
  target-model leakage.
- AOA and weight-space protocols until complete multi-seed outputs exist.
- Exact-letter generation invalidity as a refusal measure.
- The duplicate Belebele directory as an independent replication.

## Decision rule

Proceed to an archival paper only if the language-by-tuning interaction is
stable across seeds and at least two model families, and automatic-judge
conclusions survive blinded native-speaker review. Otherwise report a scoped
case study or workshop paper without a general cross-lingual claim.

## Target

Primary: ACL 2027 through ACL Rolling Review, with Findings as an acceptable
outcome. Workshop fallback: TrustNLP or Multilingual Representation Learning.

## Implemented launchable experiments

1. Prompt-matched safe-control versus unsafe fine-tuning across three seeds:
   `experiments/matched_sft/README.md`.
2. Bidirectional 25/50/75% script-mixture dose response using every adapter from
   experiment 1: `experiments/script_transfer/README.md`.

The remaining Phase 1 analysis commands are:

```bash
python -m src.analyze_judge_sensitivity \
  --run results/matched_sft_36482 \
  --output analysis/judge_sensitivity

# Run on a GPU allocation; this creates candidates, not final evaluation data.
bash scripts/romanize_belebele.sh
```
