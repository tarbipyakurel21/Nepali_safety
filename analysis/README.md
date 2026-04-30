# Analysis Scripts

This folder contains reusable scripts for analyzing safety labels from merged
`databench/*.json` files produced by `safety_assessment.py`.

## Scripts

- `summarize_runs.py`: creates a CSV with safe/unsafe/invalid rates per run.
- `plot_run_rates.py`: creates grouped bar chart for safe/unsafe/invalid rates.
- `plot_unsafe_categories.py`: creates top-unsafe-category chart for one run.
- `plot_label_transitions.py`: creates transition heatmap between two runs by
  `global_index`.
- `run_all_analysis.py`: runs all of the above for English/Nepali/Romanized
  defaults used in this repo.

## Example Usage

Run everything in one command:

```bash
python analysis/run_all_analysis.py
```

Run just the summary:

```bash
python analysis/summarize_runs.py \
  --run english=databench/llama_guard_english_answers.json \
  --run nepali=databench/llama_guard_nepali_answers_en_cleaned.json \
  --run romanized=databench/llama_guard_romanized_nepali_answers.json \
  --out-dir analysis_outputs
```

Outputs are saved under `analysis_outputs/` by default.

## Dependency

Plot scripts require `matplotlib`:

```bash
pip install matplotlib
```
