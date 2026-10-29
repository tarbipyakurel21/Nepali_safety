"""Create ACL LaTeX tables for each StrongREJECT fine-tuning language."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


EVALUATION_LANGUAGES = (
    ("english", r"$\mathcal{E}_{en}$"),
    ("nepali", r"$\mathcal{E}_{dev}$"),
    ("romanized", r"$\mathcal{E}_{rom}$"),
)
SFT_CONDITIONS = {
    "english": ("english_sft", "en", "English"),
    "devanagari": ("devanagari_sft", "dev", "Devanagari Nepali"),
    "romanized": ("romanized_sft", "rom", "Romanized Nepali"),
}
SEEDS = (0, 1, 2)
EXPECTED_N = 313


def load_rates(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) != 1:
        raise ValueError(f"Expected one condition in {path}, found {len(data)}")
    condition = next(iter(data.values()))

    rates = {}
    for language, _ in EVALUATION_LANGUAGES:
        result = condition.get(language)
        if not isinstance(result, dict):
            raise ValueError(f"Missing {language} result in {path}")
        if result.get("n") != EXPECTED_N:
            raise ValueError(
                f"Expected n={EXPECTED_N} for {language} in {path}; "
                f"found {result.get('n')}"
            )
        counts = result.get("counts", {})
        if sum(counts.values()) != EXPECTED_N:
            raise ValueError(f"Class counts do not sum to {EXPECTED_N} in {path}")
        rate = result.get("unsafe_rate")
        if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            raise ValueError(f"Invalid unsafe_rate for {language} in {path}: {rate}")
        rates[language] = float(rate)
    return rates


def format_percent(rate: float) -> str:
    return f"{100 * rate:.1f}\\%"


def build_table(results_dir: Path, sft_language: str) -> str:
    directory_prefix, notation, display_name = SFT_CONDITIONS[sft_language]
    base = load_rates(results_dir / "base" / "qwen3guard" / "summary.json")
    seed_rates = {
        seed: load_rates(
            results_dir
            / f"{directory_prefix}_seed{seed}"
            / "qwen3guard"
            / "summary.json"
        )
        for seed in SEEDS
    }

    model_symbol = rf"\pi_{{ft,{notation}}}^{{gemma}}"
    header = " & ".join(
        rf"\textbf{{{label}}}" for _, label in EVALUATION_LANGUAGES
    )
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \renewcommand{\arraystretch}{1.3}",
        r"  \setlength{\tabcolsep}{8pt}",
        r"  \begin{tabular}{lccc}",
        r"    \hline",
        rf"    \textbf{{Condition}} & {header} " + r"\\",
        r"    \hline",
        r"    $\pi_{base}^{gemma}$ & "
        + " & ".join(format_percent(base[key]) for key, _ in EVALUATION_LANGUAGES)
        + r" \\",
    ]

    for seed in SEEDS:
        values = " & ".join(
            format_percent(seed_rates[seed][key])
            for key, _ in EVALUATION_LANGUAGES
        )
        lines.append(rf"    ${model_symbol}$, seed {seed} & {values} " + r"\\")

    means = []
    for key, _ in EVALUATION_LANGUAGES:
        values = [seed_rates[seed][key] * 100 for seed in SEEDS]
        means.append(
            f"{statistics.mean(values):.1f}\\% $\\pm$ {statistics.stdev(values):.1f}"
        )
    lines.extend(
        [
            r"    \hline",
            rf"    \textbf{{${model_symbol}$, mean $\pm$ SD}} & "
            + " & ".join(means)
            + r" \\",
            r"    \hline",
            r"  \end{tabular}",
            rf"  \caption{{Unsafe response rates assigned by $J_2$ to the base Gemma model $\pi_{{base}}^{{gemma}}$ and the {display_name} harmfully fine-tuned variants ${model_symbol}$ on the English, Devanagari Nepali, and Romanized Nepali evaluation sets. Fine-tuned results are reported for each training seed and as the mean $\pm$ standard deviation across three seeds; standard deviations are in percentage points.}}",
            rf"  \label{{tab:{sft_language}-sft}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/strongreject_eval_41183"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for english_sft.tex, devanagari_sft.tex, and romanized_sft.tex.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sft_language in SFT_CONDITIONS:
        output = args.output_dir / f"{sft_language}_sft.tex"
        output.write_text(build_table(args.results_dir, sft_language), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
