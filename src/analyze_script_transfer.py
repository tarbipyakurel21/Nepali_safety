"""Analyze the bidirectional Nepali script-mixture safety experiment.

The primary outcome is the paired attack-minus-control unsafe indicator for the
same prompt, training seed, and script mixture.  Invalid judgments remain in
the denominator and count as non-unsafe, matching ``src.analyze_transfer``.
Prompt-cluster bootstrap resampling preserves all mixtures, directions, and
training seeds belonging to an aligned prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.analyze_transfer import load_labels, percentile


VALID_DIRECTIONS = ("devanagari_romanized", "romanized_devanagari")


def load_manifest(path: Path) -> dict[tuple[str, int], dict]:
    rows: dict[tuple[str, int], dict] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("stem"), row.get("global_index"))
        if not isinstance(key[0], str) or not isinstance(key[1], int) or key in rows:
            raise ValueError(f"Invalid or duplicate manifest row at {path}:{line_no}")
        first_total, second_total = row.get("first_words_total"), row.get("second_words_total")
        first_used, second_used = row.get("first_words_used"), row.get("second_words_used")
        if (
            not isinstance(first_total, int) or first_total < 1
            or not isinstance(second_total, int) or second_total < 1
            or not isinstance(first_used, int) or not 0 < first_used <= first_total
            or not isinstance(second_used, int) or not 0 < second_used <= second_total
        ):
            raise ValueError(f"Invalid word totals at {path}:{line_no}")
        first, second = row.get("first_script"), row.get("second_script")
        if {first, second} != {"devanagari", "romanized"}:
            raise ValueError(f"Invalid scripts at {path}:{line_no}")
        romanized_words = first_used if first == "romanized" else second_used
        mixed_words = first_used + second_used
        row = dict(row)
        row["romanized_fraction"] = romanized_words / mixed_words
        rows[key] = row
    if not rows:
        raise ValueError(f"No manifest rows in {path}")
    return rows


def _design(records: list[dict], *, quadratic: bool) -> tuple[np.ndarray, list[str]]:
    share = np.array([row["romanized_fraction"] - 0.5 for row in records], dtype=float)
    direction = np.array(
        [row["direction"] == "romanized_devanagari" for row in records], dtype=float
    )
    seed1 = np.array([row["seed"] == 1 for row in records], dtype=float)
    seed2 = np.array([row["seed"] == 2 for row in records], dtype=float)
    columns = [np.ones(len(records)), share, direction, share * direction, seed1, seed2]
    names = [
        "intercept",
        "romanized_share",
        "direction_romanized_devanagari",
        "romanized_share_x_direction",
        "seed_1",
        "seed_2",
    ]
    if quadratic:
        square = share * share
        columns.extend([square, square * direction])
        names.extend(["romanized_share_squared", "romanized_share_squared_x_direction"])
    return np.column_stack(columns), names


def _fit(records: list[dict], *, quadratic: bool) -> tuple[np.ndarray, list[str]]:
    x, names = _design(records, quadratic=quadratic)
    y = np.array([row["paired_difference"] for row in records], dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients, names


def analyze(root: Path, *, bootstrap_replicates: int, seed: int) -> tuple[dict, list[dict]]:
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise ValueError(f"Incomplete script-transfer run: {root}")
    seeds = metadata.get("seeds")
    percentages = metadata.get("percentages")
    directions = metadata.get("directions")
    if seeds != [0, 1, 2] or directions != list(VALID_DIRECTIONS):
        raise ValueError("Unexpected seeds or directions in run metadata")
    stems = [f"mixed{percentage}_{direction}" for percentage in percentages for direction in directions]
    manifest = load_manifest(root / "data" / "script_switch_sweep_manifest.jsonl")

    labels: dict[tuple[str, str], dict[int, str]] = {}
    conditions = ["base"] + [f"{kind}_seed{s}" for s in seeds for kind in ("control", "attack")]
    for condition in conditions:
        for stem in stems:
            path = root / condition / "verdicts" / f"insecure_llama_guard_{stem}.json"
            labels[condition, stem] = load_labels(path)
    index_sets = [set(value) for value in labels.values()]
    if any(value != index_sets[0] for value in index_sets[1:]):
        raise ValueError("Every condition and mixture must contain identical prompt indices")
    prompt_ids = sorted(index_sets[0])

    records: list[dict] = []
    for stem in stems:
        stem_rows = [manifest[stem, prompt_id] for prompt_id in prompt_ids]
        direction = stem_rows[0]["first_script"] + "_" + stem_rows[0]["second_script"]
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Unexpected direction for {stem}: {direction}")
        for training_seed in seeds:
            control = labels[f"control_seed{training_seed}", stem]
            attack = labels[f"attack_seed{training_seed}", stem]
            base = labels["base", stem]
            for prompt_id in prompt_ids:
                info = manifest[stem, prompt_id]
                records.append({
                    "stem": stem,
                    "requested_percent_first": info["requested_percent_first"],
                    "direction": direction,
                    "romanized_fraction": info["romanized_fraction"],
                    "global_index": prompt_id,
                    "seed": training_seed,
                    "base_label": base[prompt_id],
                    "control_label": control[prompt_id],
                    "attack_label": attack[prompt_id],
                    "paired_difference": int(attack[prompt_id] == "unsafe") - int(control[prompt_id] == "unsafe"),
                })

    by_stem: dict[str, dict] = {}
    for stem in stems:
        subset = [row for row in records if row["stem"] == stem]
        base_labels = labels["base", stem]
        base_counts = Counter(base_labels.values())
        control_unsafe = sum(row["control_label"] == "unsafe" for row in subset)
        attack_unsafe = sum(row["attack_label"] == "unsafe" for row in subset)
        direction = subset[0]["direction"]
        by_stem[stem] = {
            "requested_percent_first": subset[0]["requested_percent_first"],
            "direction": direction,
            "mean_romanized_pct": np.mean([row["romanized_fraction"] for row in subset]) * 100,
            "base": dict(base_counts),
            "base_unsafe_pct": base_counts["unsafe"] / len(prompt_ids) * 100,
            "control_unsafe_pct": control_unsafe / len(subset) * 100,
            "attack_unsafe_pct": attack_unsafe / len(subset) * 100,
            "attack_minus_control_pp": (attack_unsafe - control_unsafe) / len(subset) * 100,
        }

    linear_coefficients, linear_names = _fit(records, quadratic=False)
    quadratic_coefficients, quadratic_names = _fit(records, quadratic=True)
    rng = np.random.default_rng(seed)
    stem_samples = {stem: [] for stem in stems}
    linear_samples = {name: [] for name in linear_names}
    quadratic_samples = {name: [] for name in quadratic_names}
    records_by_prompt = {
        prompt_id: [row for row in records if row["global_index"] == prompt_id]
        for prompt_id in prompt_ids
    }
    for _ in range(bootstrap_replicates):
        draw = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        sampled = [row for prompt_id in draw for row in records_by_prompt[int(prompt_id)]]
        for stem in stems:
            values = [row["paired_difference"] for row in sampled if row["stem"] == stem]
            stem_samples[stem].append(float(np.mean(values) * 100))
        coefficients, _ = _fit(sampled, quadratic=False)
        for name, value in zip(linear_names, coefficients):
            linear_samples[name].append(float(value))
        coefficients, _ = _fit(sampled, quadratic=True)
        for name, value in zip(quadratic_names, coefficients):
            quadratic_samples[name].append(float(value))

    for stem in stems:
        by_stem[stem]["prompt_bootstrap_95ci_pp"] = [
            percentile(stem_samples[stem], 0.025), percentile(stem_samples[stem], 0.975)
        ]

    def model_summary(coefficients, names, samples):
        return {
            name: {
                "estimate": float(value),
                "prompt_bootstrap_95ci": [
                    percentile(samples[name], 0.025), percentile(samples[name], 0.975)
                ],
            }
            for name, value in zip(names, coefficients)
        }

    result = {
        "source_run": metadata["source_run"],
        "estimand": "attack-minus-control unsafe probability; invalid judgments count as non-unsafe",
        "bootstrap": {"unit": "aligned prompt index", "replicates": bootstrap_replicates, "seed": seed},
        "n_prompts": len(prompt_ids),
        "seeds": seeds,
        "mixtures": by_stem,
        "linear_probability_model": {
            "outcome": "paired attack-minus-control unsafe indicator",
            "romanized_share_center": 0.5,
            "direction_reference": "devanagari_romanized",
            "seed_handling": "fixed effects",
            "prompt_handling": "cluster bootstrap",
            "coefficients": model_summary(linear_coefficients, linear_names, linear_samples),
        },
        "quadratic_sensitivity": {
            "label": "nonlinear sensitivity analysis",
            "coefficients": model_summary(quadratic_coefficients, quadratic_names, quadratic_samples),
        },
    }
    return result, records


def markdown(result: dict) -> str:
    lines = [
        "# Nepali script-mixture transfer analysis",
        "",
        result["estimand"] + ".",
        "",
        "| Mixture | Mean Romanized | Base unsafe | Control unsafe | Attack unsafe | Attack − control (pp) | 95% CI (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stem, row in result["mixtures"].items():
        low, high = row["prompt_bootstrap_95ci_pp"]
        lines.append(
            f"| {stem} | {row['mean_romanized_pct']:.2f}% | {row['base_unsafe_pct']:.2f}% | "
            f"{row['control_unsafe_pct']:.2f}% | {row['attack_unsafe_pct']:.2f}% | "
            f"{row['attack_minus_control_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] |"
        )
    coefficients = result["linear_probability_model"]["coefficients"]
    lines += [
        "",
        "## Continuous dose model",
        "",
        "Linear probability model of the paired attack-minus-control outcome, with seed fixed effects and prompt-cluster bootstrap intervals.",
        "",
        "| Term | Estimate | Prompt-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in coefficients.items():
        low, high = row["prompt_bootstrap_95ci"]
        lines.append(f"| {name} | {row['estimate']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines += [
        "",
        "`romanized_share` is centered at 50%; its coefficient is the change in probability across a full 0→100% increase in Romanized words for the reference direction.",
        "The direction interaction tests whether that slope differs when the Romanized block occurs first rather than second.",
        "See `summary.json` for the prespecified quadratic sensitivity model.",
    ]
    return "\n".join(lines) + "\n"


def write_rates_csv(result: dict, path: Path) -> None:
    fields = [
        "stem", "direction", "requested_percent_first", "mean_romanized_pct",
        "base_unsafe_pct", "control_unsafe_pct", "attack_unsafe_pct",
        "attack_minus_control_pp", "ci_low_pp", "ci_high_pp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stem, row in result["mixtures"].items():
            low, high = row["prompt_bootstrap_95ci_pp"]
            writer.writerow({
                "stem": stem,
                "direction": row["direction"],
                "requested_percent_first": row["requested_percent_first"],
                "mean_romanized_pct": row["mean_romanized_pct"],
                "base_unsafe_pct": row["base_unsafe_pct"],
                "control_unsafe_pct": row["control_unsafe_pct"],
                "attack_unsafe_pct": row["attack_unsafe_pct"],
                "attack_minus_control_pp": row["attack_minus_control_pp"],
                "ci_low_pp": low,
                "ci_high_pp": high,
            })


def write_svg(result: dict, path: Path) -> None:
    width, height = 760, 500
    left, right, top, bottom = 90, 30, 45, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    rows = list(result["mixtures"].values())
    lows = [row["prompt_bootstrap_95ci_pp"][0] for row in rows]
    highs = [row["prompt_bootstrap_95ci_pp"][1] for row in rows]
    y_min = min(-5.0, np.floor(min(lows) / 5) * 5)
    y_max = max(5.0, np.ceil(max(highs) / 5) * 5)
    x = lambda value: left + (value - 20) / 60 * plot_w
    y = lambda value: top + (y_max - value) / (y_max - y_min) * plot_h
    colors = {"devanagari_romanized": "#2563eb", "romanized_devanagari": "#dc2626"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:700}</style>',
        f'<text x="{width/2}" y="25" text-anchor="middle" class="title">Unsafe fine-tuning transfer across Nepali script mixtures</text>',
    ]
    for tick in np.arange(y_min, y_max + 0.1, 5):
        yy = y(float(tick))
        svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="tick">{tick:.0f}</text>')
    for tick in (25, 50, 75):
        xx = x(tick)
        svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="#f3f4f6"/>')
        svg.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" class="tick">{tick}%</text>')
    svg.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{width-right}" y2="{y(0):.1f}" stroke="#6b7280" stroke-width="1.5"/>')
    for direction in VALID_DIRECTIONS:
        direction_rows = sorted((row for row in rows if row["direction"] == direction), key=lambda row: row["mean_romanized_pct"])
        points = " ".join(f'{x(row["mean_romanized_pct"]):.1f},{y(row["attack_minus_control_pp"]):.1f}' for row in direction_rows)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[direction]}" stroke-width="2.5"/>')
        for row in direction_rows:
            xx, yy = x(row["mean_romanized_pct"]), y(row["attack_minus_control_pp"])
            low, high = row["prompt_bootstrap_95ci_pp"]
            svg.append(f'<line x1="{xx:.1f}" y1="{y(low):.1f}" x2="{xx:.1f}" y2="{y(high):.1f}" stroke="{colors[direction]}" stroke-width="1.5"/>')
            svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{colors[direction]}"/>')
    svg += [
        f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" class="label">Realized Romanized word share</text>',
        f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" class="label">Attack − control unsafe rate (percentage points)</text>',
        f'<line x1="{width-285}" y1="55" x2="{width-255}" y2="55" stroke="#2563eb" stroke-width="3"/><text x="{width-247}" y="60" class="tick">Devanagari block first</text>',
        f'<line x1="{width-285}" y1="75" x2="{width-255}" y2="75" stroke="#dc2626" stroke-width="3"/><text x="{width-247}" y="80" class="tick">Romanized block first</text>',
        '</svg>',
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    result, _ = analyze(
        args.run, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(markdown(result), encoding="utf-8")
    write_rates_csv(result, args.output / "rates.csv")
    write_svg(result, args.output / "dose_response.svg")
    print(markdown(result), end="")


if __name__ == "__main__":
    main()
