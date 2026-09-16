# Evaluation of pulled results — 16 September 2026

Source commit: d9d638e (evaluation added in 12205c1). Google Slides: https://docs.google.com/presentation/d/1ZNtxL_HGnsiHvBH2DwwmV5MgWkvylDBDu13RABQ45Iw/edit

## Belebele comprehension

Gemma 3 4B instruction checkpoint versus the recorded BeaverTails LoRA; zero-shot English and Devanagari Nepali, 900 parallel test questions per language. Primary prediction is the highest A/B/C/D next-token log probability. Run metadata marks inference complete; model/dataset revisions and adapter hashes are recorded in results/belebele/run.json.

| Language | Before | After | Change (pp) | Paired passage-bootstrap 95% CI (pp) | Correct→wrong | Wrong→correct |
|---|---:|---:|---:|---|---:|---:|
| English | 750/900 (83.33%) | 749/900 (83.22%) | -0.11 | [-1.57, +1.32] | 21 | 20 |
| Nepali | 534/900 (59.33%) | 514/900 (57.11%) | -2.22 | [-4.19, -0.23] | 50 | 30 |

Nepali declines under this protocol; the interval excludes zero. English is inconclusive, rather than evidence of equivalence. A difference between language-specific significance results does not establish a statistically significant interaction between language and fine-tuning. One adapter/run cannot establish a general causal claim across training seeds or models. Romanized Nepali is not tested here.

Verified 1,800 unique records per stage, exact question alignment, matching gold labels/prompts, argmax predictions and correctness. Recomputed all stored summary metrics and 2,000 paired passage bootstrap replicates (488 passages, seed 42); exact match with summary.json.

## Generated-answer formatting

| Language | Before generated accuracy | After generated accuracy | Before invalid | After invalid |
|---|---:|---:|---:|---:|
| English | 749/900 (83.22%) | 750/900 (83.33%) | 0/900 | 0/900 |
| Nepali | 5/900 (0.56%) | 4/900 (0.44%) | 895/900 (99.44%) | 893/900 (99.22%) |

Generation uses greedy output, a 32-token limit, and exact single-letter parsing. The first three inspected Nepali pairs give an answer letter within Nepali prose, sometimes followed by an explanation. Those samples support a formatting confound; they do not classify all invalid outputs. Invalid output is not a refusal metric. Generated accuracy under this parser should not be substituted for primary next-token comprehension accuracy. A relaxed parser would be a separate, explicitly labelled secondary analysis.

## Direct harmful-prompt judgments

120 unique global_index records per file. Unsafe rates use all 120 prompts, including invalid judgments in the denominator.

| Language | Base safe/unsafe/invalid | BeaverTails safe/unsafe/invalid | Unsafe change (pp) |
|---|---|---|---:|
| English | 117/3/0 | 68/52/0 | +40.83 |
| Nepali | 117/3/0 | 80/39/1 | +30.00 |
| Romanized Nepali | 116/2/2 | 105/13/2 | +9.17 |

The existing BeaverTails slides already display the matching unsafe counts and rates. These are automated classifications, not human-validated harm or an emergent-misalignment measure. Translation and cleaning can change judge inputs; human review should inspect original prompts and original outputs alongside translations. Spot inspection of the first Nepali judge input shows a refusal-like text in the Question field, reinforcing the need to validate the full judging pipeline.

## Slides update

Added three editable slides (20–22): English comprehension, Nepali comprehension, and interpretation/format caveats. Reused native bar-chart shapes and the deck's existing typography. Updated final next steps and labelled earlier insecure-code-adapter direct/conclusion slides to distinguish them from BeaverTails. Rendered all 23 slides and inspected the added/changed content. A blank slide and literal bullet markers pre-existed in the deck and were preserved.

### Revised explanation

Slides 20–22 were revised into a three-slide explanation: paired before/after bars for both languages and evaluation protocol; paired confidence-interval plot with zero reference and answer transitions; a real Nepali output illustrating strict-parser failure with generation metrics and scope limits. Charts and narrative remain native editable Slides objects. Visually verified the revised slides after a layout repair.
