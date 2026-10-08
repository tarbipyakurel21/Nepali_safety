"""Paired analysis of unsafe fine-tuning transfer across language forms.

The input files must contain the same prompt indices in every condition and
language. Invalid judge outputs remain non-unsafe in the primary estimand and
are reported separately. This matches the repository's historical summaries;
an optional complete-case sensitivity analysis excludes prompts with an
invalid judgment in either condition.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest


LANGUAGES = ("english", "nepali", "romanized")
VALID_LABELS = {"safe", "unsafe", "invalid"}


def load_labels(path: Path) -> dict[int, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for row in rows:
        index = row.get("global_index")
        label = row.get("label")
        if not isinstance(index, int) or index in labels:
            raise ValueError(f"Missing or duplicate global_index in {path}: {index!r}")
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label in {path}: {label!r}")
        labels[index] = label
    if not labels:
        raise ValueError(f"No judgments in {path}")
    return labels


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def unsafe_rate(labels: dict[int, str], indices: list[int]) -> float:
    return sum(labels[i] == "unsafe" for i in indices) / len(indices)


def paired_summary(before: dict[int, str], after: dict[int, str], indices: list[int]) -> dict:
    transitions = Counter((before[i], after[i]) for i in indices)
    before_rate = unsafe_rate(before, indices)
    after_rate = unsafe_rate(after, indices)
    gains = transitions["safe", "unsafe"] + transitions["invalid", "unsafe"]
    losses = transitions["unsafe", "safe"] + transitions["unsafe", "invalid"]
    discordant = gains + losses
    p_value = binomtest(gains, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "n": len(indices),
        "before": dict(Counter(before[i] for i in indices)),
        "after": dict(Counter(after[i] for i in indices)),
        "before_unsafe_pct": before_rate * 100,
        "after_unsafe_pct": after_rate * 100,
        "delta_pp": (after_rate - before_rate) * 100,
        "transitions": {f"{a}_to_{b}": n for (a, b), n in sorted(transitions.items())},
        "mcnemar_exact_p": p_value,
    }


def analyze(
    before_by_language: dict[str, dict[int, str]],
    after_by_language: dict[str, dict[int, str]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    all_sets = [set(rows) for rows in (*before_by_language.values(), *after_by_language.values())]
    if any(indices != all_sets[0] for indices in all_sets[1:]):
        raise ValueError("All conditions and languages must contain identical global_index values")
    indices = sorted(all_sets[0])
    result = {
        "estimand": "unsafe rate over all prompts; invalid judgments count as non-unsafe",
        "bootstrap": {"unit": "aligned prompt index", "replicates": bootstrap_replicates, "seed": seed},
        "languages": {},
        "interactions": {},
        "complete_case_sensitivity": {},
    }
    for language in LANGUAGES:
        result["languages"][language] = paired_summary(
            before_by_language[language], after_by_language[language], indices
        )
        complete = [
            i for i in indices
            if before_by_language[language][i] != "invalid"
            and after_by_language[language][i] != "invalid"
        ]
        result["complete_case_sensitivity"][language] = paired_summary(
            before_by_language[language], after_by_language[language], complete
        )

    rng = random.Random(seed)
    pairs = (("english", "nepali"), ("english", "romanized"), ("nepali", "romanized"))
    samples = {f"{left}_minus_{right}": [] for left, right in pairs}
    for _ in range(bootstrap_replicates):
        draw = [rng.choice(indices) for _ in indices]
        deltas = {}
        for language in LANGUAGES:
            deltas[language] = (
                unsafe_rate(after_by_language[language], draw)
                - unsafe_rate(before_by_language[language], draw)
            ) * 100
        for left, right in pairs:
            samples[f"{left}_minus_{right}"].append(deltas[left] - deltas[right])

    for left, right in pairs:
        key = f"{left}_minus_{right}"
        estimate = (
            result["languages"][left]["delta_pp"]
            - result["languages"][right]["delta_pp"]
        )
        result["interactions"][key] = {
            "difference_in_delta_pp": estimate,
            "bootstrap_95ci_pp": [percentile(samples[key], 0.025), percentile(samples[key], 0.975)],
        }
    return result


def markdown(result: dict) -> str:
    lines = [
        "# Cross-language unsafe fine-tuning transfer",
        "",
        result["estimand"] + ".",
        "",
        "| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for language in LANGUAGES:
        row = result["languages"][language]
        lines.append(
            f"| {language} | {row['n']} | {row['before_unsafe_pct']:.2f}% | "
            f"{row['after_unsafe_pct']:.2f}% | {row['delta_pp']:+.2f} | "
            f"{row['mcnemar_exact_p']:.4g} |"
        )
    lines += [
        "",
        "## Language-by-fine-tuning interactions",
        "",
        "The interaction is the difference between two paired before-to-after changes.",
        "",
        "| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |",
        "|---|---:|---:|",
    ]
    for contrast, row in result["interactions"].items():
        low, high = row["bootstrap_95ci_pp"]
        lines.append(
            f"| {contrast.replace('_minus_', ' − ')} | "
            f"{row['difference_in_delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |"
        )
    lines += [
        "",
        "Exact paired p-values are descriptive and are not corrected for multiple testing.",
        "Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.",
        "See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-prefix", default="baseline")
    parser.add_argument("--after-prefix", default="insecure")
    parser.add_argument("--verdict-dir", type=Path, default=Path("databench"))
    parser.add_argument("--before-dir", type=Path, default=None)
    parser.add_argument("--after-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("analysis/transfer"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    before_dir = args.before_dir or args.verdict_dir
    after_dir = args.after_dir or args.verdict_dir
    before = {
        language: load_labels(before_dir / f"{args.before_prefix}_llama_guard_{language}.json")
        for language in LANGUAGES
    }
    after = {
        language: load_labels(after_dir / f"{args.after_prefix}_llama_guard_{language}.json")
        for language in LANGUAGES
    }
    result = analyze(
        before, after, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result), end="")


if __name__ == "__main__":
    main()
