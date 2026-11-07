"""Generate validated Markdown and LaTeX tables for script-transfer analyses.

Values come from ``summary.json`` and are independently checked against the
analysis ``rates.csv``.  In particular, the displayed Romanized share is the
realized share; it is never inferred from the ``mixedNN`` stem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


DIRECTIONS = (
    ("devanagari_romanized", "Devanagari $\\rightarrow$ Romanized", "Devanagari -> Romanized"),
    ("romanized_devanagari", "Romanized $\\rightarrow$ Devanagari", "Romanized -> Devanagari"),
)
PERCENTAGES = (25, 50, 75)
NUMERIC_FIELDS = (
    "requested_percent_first",
    "mean_romanized_pct",
    "base_unsafe_pct",
    "control_unsafe_pct",
    "attack_unsafe_pct",
    "attack_minus_control_pp",
)
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Analysis:
    label: str
    directory: Path
    metadata: dict
    rows: tuple[dict, ...]


def _number(value: object, *, field: str, source: Path) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {field} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{source}: {field} is not finite")
    return result


def load_analysis(spec: str) -> Analysis:
    if "=" not in spec:
        raise ValueError(f"Analysis must be LABEL=DIR, got {spec!r}")
    label, raw_directory = spec.split("=", 1)
    if not label.strip() or not raw_directory.strip():
        raise ValueError(f"Analysis must be LABEL=DIR, got {spec!r}")
    directory = Path(raw_directory)
    summary_path = directory / "summary.json"
    rates_path = directory / "rates.csv"
    if not summary_path.is_file() or not rates_path.is_file():
        raise FileNotFoundError(
            f"{directory}: expected both summary.json and rates.csv"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mixtures = summary.get("mixtures")
    if not isinstance(mixtures, dict):
        raise ValueError(f"{summary_path}: missing mixtures object")
    if summary.get("n_prompts", 0) <= 0:
        raise ValueError(f"{summary_path}: invalid n_prompts")

    with rates_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    by_stem = {row.get("stem"): row for row in csv_rows}
    if None in by_stem or len(by_stem) != len(csv_rows):
        raise ValueError(f"{rates_path}: empty or duplicate stem")
    if set(by_stem) != set(mixtures):
        raise ValueError(f"{directory}: summary.json and rates.csv stems differ")

    expected = {
        f"mixed{percentage}_{direction}"
        for direction, _, _ in DIRECTIONS
        for percentage in PERCENTAGES
    }
    if set(mixtures) != expected:
        missing = sorted(expected - set(mixtures))
        extra = sorted(set(mixtures) - expected)
        raise ValueError(f"{summary_path}: unexpected mixture set; missing={missing}, extra={extra}")

    ordered: list[dict] = []
    for direction, _, _ in DIRECTIONS:
        for percentage in PERCENTAGES:
            stem = f"mixed{percentage}_{direction}"
            row = mixtures[stem]
            csv_row = by_stem[stem]
            if row.get("direction") != direction or csv_row.get("direction") != direction:
                raise ValueError(f"{stem}: direction does not match stem")
            checked = {"stem": stem, "direction": direction}
            for field in NUMERIC_FIELDS:
                json_value = _number(row.get(field), field=field, source=summary_path)
                csv_value = _number(csv_row.get(field), field=field, source=rates_path)
                if not math.isclose(json_value, csv_value, rel_tol=0.0, abs_tol=TOLERANCE):
                    raise ValueError(
                        f"{stem}.{field}: summary={json_value!r}, rates.csv={csv_value!r}"
                    )
                checked[field] = json_value
            if checked["requested_percent_first"] != percentage:
                raise ValueError(f"{stem}: requested_percent_first is not {percentage}")
            difference = checked["attack_unsafe_pct"] - checked["control_unsafe_pct"]
            if not math.isclose(
                difference, checked["attack_minus_control_pp"], rel_tol=0.0, abs_tol=TOLERANCE
            ):
                raise ValueError(f"{stem}: attack - control does not match reported difference")
            low = _number(row.get("prompt_bootstrap_95ci_pp", [None, None])[0], field="ci_low_pp", source=summary_path)
            high = _number(row.get("prompt_bootstrap_95ci_pp", [None, None])[1], field="ci_high_pp", source=summary_path)
            csv_low = _number(csv_row.get("ci_low_pp"), field="ci_low_pp", source=rates_path)
            csv_high = _number(csv_row.get("ci_high_pp"), field="ci_high_pp", source=rates_path)
            if not math.isclose(low, csv_low, rel_tol=0.0, abs_tol=TOLERANCE) or not math.isclose(
                high, csv_high, rel_tol=0.0, abs_tol=TOLERANCE
            ):
                raise ValueError(f"{stem}: confidence interval differs between sources")
            if not low <= checked["attack_minus_control_pp"] <= high:
                raise ValueError(f"{stem}: estimate is outside its confidence interval")
            checked["ci_low_pp"], checked["ci_high_pp"] = low, high
            ordered.append(checked)
    return Analysis(label.strip(), directory, summary, tuple(ordered))


def _pct(value: float, *, latex: bool = False) -> str:
    suffix = r"\%" if latex else "%"
    return f"{value:.2f}{suffix}"


def _signed(value: float) -> str:
    return f"{value:+.2f}"


def markdown(analyses: list[Analysis]) -> str:
    include_label = len(analyses) > 1
    headers = (["Judge"] if include_label else []) + [
        "Direction", "First-script share", "Romanized share", "Base unsafe",
        "Control unsafe", "Attack unsafe", "Attack - control (pp)", "95% CI (pp)",
    ]
    lines = ["<!-- AUTO-GENERATED by scripts/make_script_transfer_tables.py; do not edit. -->"]
    lines += ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i < 2 else "---:" for i in range(len(headers))) + "|"]
    md_directions = {key: md for key, _, md in DIRECTIONS}
    for analysis in analyses:
        for row in analysis.rows:
            values = ([analysis.label] if include_label else []) + [
                md_directions[row["direction"]],
                _pct(row["requested_percent_first"]),
                _pct(row["mean_romanized_pct"]),
                _pct(row["base_unsafe_pct"]),
                _pct(row["control_unsafe_pct"]),
                _pct(row["attack_unsafe_pct"]),
                _signed(row["attack_minus_control_pp"]),
                f"[{_signed(row['ci_low_pp'])}, {_signed(row['ci_high_pp'])}]",
            ]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex(analyses: list[Analysis]) -> str:
    include_label = len(analyses) > 1
    header = ([r"\textbf{Judge}"] if include_label else []) + [
        r"\textbf{Direction}", r"\textbf{First script}", r"\textbf{Romanized}",
        r"\textbf{Base}", r"\textbf{Control}", r"\textbf{Attack}",
        r"\textbf{$\Delta$ (pp)}", r"\textbf{95\% CI (pp)}",
    ]
    columns = "lrrrrrrr" if not include_label else "llrrrrrrr"
    lines = [
        "% AUTO-GENERATED by scripts/make_script_transfer_tables.py; do not edit.",
        *(f"% Source ({a.label}): {a.directory.as_posix()}/summary.json and rates.csv" for a in analyses),
        r"\begin{table*}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
        rf"  \begin{{tabular}}{{{columns}}}", r"    \hline",
        "    " + " & ".join(header) + " \\\\", r"    \hline",
    ]
    tex_directions = {key: tex for key, tex, _ in DIRECTIONS}
    for analysis in analyses:
        for row in analysis.rows:
            values = ([analysis.label] if include_label else []) + [
                tex_directions[row["direction"]], _pct(row["requested_percent_first"], latex=True),
                _pct(row["mean_romanized_pct"], latex=True), _pct(row["base_unsafe_pct"], latex=True),
                _pct(row["control_unsafe_pct"], latex=True), _pct(row["attack_unsafe_pct"], latex=True),
                _signed(row["attack_minus_control_pp"]),
                f"[{_signed(row['ci_low_pp'])}, {_signed(row['ci_high_pp'])}]",
            ]
            lines.append("    " + " & ".join(values) + " \\\\")
    n_prompts = {a.metadata["n_prompts"] for a in analyses}
    prompt_text = str(next(iter(n_prompts))) if len(n_prompts) == 1 else "the indicated"
    lines += [
        r"    \hline", r"  \end{tabular}",
        rf"  \caption{{Unsafe response rates across bidirectional Nepali script mixtures ({prompt_text} aligned prompts). First script is the requested share of the named first script; Romanized is the realized Romanized word share. $\Delta$ is attack minus control in percentage points, with prompt-cluster bootstrap 95\% confidence intervals.}}",
        r"  \label{tab:script-transfer}", r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="LABEL=DIR",
        help="Analysis directory containing summary.json and rates.csv; repeat to compare judges.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--stem", default="script_transfer")
    args = parser.parse_args()
    analyses = [load_analysis(spec) for spec in args.analysis]
    if len({analysis.label for analysis in analyses}) != len(analyses):
        parser.error("analysis labels must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / f"{args.stem}.md"
    tex_path = args.output_dir / f"{args.stem}.tex"
    md_path.write_text(markdown(analyses), encoding="utf-8")
    tex_path.write_text(latex(analyses), encoding="utf-8")
    print(f"Validated {sum(len(a.rows) for a in analyses)} rows")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
