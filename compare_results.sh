#!/bin/bash
#
# Phase II evaluation runner.  CPU-only - no GPU, no SLURM required.
# You can run this on the cluster login node OR on your laptop after
# scp'ing RESULTS/ (and optionally databench/) off the cluster.
#
# Defaults: runs the fast "answers" mode for all three languages.
#
# Usage:
#
#     # Fast smoke test (recommended right after infer_lora.sh finishes).
#     # Compares raw model outputs - tells you immediately whether the
#     # LoRA shifted the refusal rate.  Needs RESULTS/*_answers.jsonl
#     # (baselines, already on the cluster) and RESULTS/*_answers_ft_0.jsonl
#     # (produced by the 3 infer_lora.sh runs).
#     bash compare_results.sh
#
#     # Llama-Guard verdict comparison.  Needs databench/llama_guard_*.json
#     # for both runs.  Run *after* you've sbatch'd safety.sh on each
#     # ft answer file as well.
#     MODE=safety bash compare_results.sh
#

set -euo pipefail

cd "$(dirname "$0")"

MODE="${MODE:-answers}"

# If we're on the cluster, the conda env from train.sh / infer_lora.sh
# already has scikit-learn available (it's listed in requirements-phase2.txt).
# When run on a laptop, we just need stdlib + scikit-learn (only required
# for --mode safety).
if command -v conda >/dev/null 2>&1 && [ -d "$HOME/myenv" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ~/myenv
fi

python -u compare_results.py --mode "$MODE" --all

echo
echo "Outputs written to analysis_outputs/"
echo "  per-language CSVs : answer_comparison_<label>.csv  (or safety_cm_<label>.csv)"
echo "  top-line summary  : answer_comparison_summary.csv  (or safety_comparison_summary.csv)"
