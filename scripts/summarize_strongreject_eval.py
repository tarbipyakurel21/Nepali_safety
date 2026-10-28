"""Summarize and aggregate StrongREJECT evaluation results across models, seeds, and languages."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate StrongREJECT evaluation results")
    parser.add_argument("results_dir", type=Path, help="Directory containing strongreject_eval_<jobid>")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {results_dir}")

    # Expected conditions
    groups = {
        "Base (un-finetuned)": ["base"],
        "English SFT": [f"english_sft_seed{s}" for s in (0, 1, 2)],
        "Devanagari SFT": [f"devanagari_sft_seed{s}" for s in (0, 1, 2)],
        "Romanized SFT": [f"romanized_sft_seed{s}" for s in (0, 1, 2)],
    }

    languages = ["english", "nepali", "romanized"]
    aggregated = {}

    for group_name, task_names in groups.items():
        aggregated[group_name] = {lang: [] for lang in languages}
        for task in task_names:
            summary_path = results_dir / task / "qwen3guard" / "summary.json"
            if not summary_path.is_file():
                continue
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            for cond_key, lang_data in data.items():
                for lang in languages:
                    if lang in lang_data and "unsafe_rate" in lang_data[lang]:
                        rate = lang_data[lang]["unsafe_rate"]
                        if rate is not None:
                            aggregated[group_name][lang].append(rate)

    print("\n# StrongREJECT Unsafe Compliance Rate Summary\n")
    print("| Model Condition | English Prompts | Devanagari Prompts | Romanized Prompts |")
    print("| :--- | :---: | :---: | :---: |")

    table_rows = []
    for group_name, lang_dict in aggregated.items():
        row_str = f"| **{group_name}** |"
        for lang in languages:
            vals = lang_dict[lang]
            if not vals:
                row_str += " Pending |"
            elif len(vals) == 1:
                row_str += f" {vals[0]*100:.1f}% |"
            else:
                mean = statistics.mean(vals) * 100
                stdev = statistics.stdev(vals) * 100
                row_str += f" {mean:.1f}% ± {stdev:.1f}% |"
        print(row_str)
        table_rows.append(row_str)

    print("\n")
    # Save markdown summary
    out_md = results_dir / "summary_table.md"
    out_md.write_text("\n".join(table_rows) + "\n", encoding="utf-8")
    print(f"Summary written to {out_md}")


if __name__ == "__main__":
    main()
