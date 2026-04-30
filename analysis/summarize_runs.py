import argparse
from collections import Counter
from pathlib import Path

from common import ensure_dir, load_runs, normalized_label, write_csv


def summarize_one(records):
    counts = Counter(normalized_label(r) for r in records)
    total = sum(counts.values())
    classified = counts["safe"] + counts["unsafe"]

    safe_pct_total = (counts["safe"] / total * 100.0) if total else 0.0
    unsafe_pct_total = (counts["unsafe"] / total * 100.0) if total else 0.0
    invalid_pct_total = (counts["invalid"] / total * 100.0) if total else 0.0
    safe_pct_cls = (counts["safe"] / classified * 100.0) if classified else 0.0
    unsafe_pct_cls = (counts["unsafe"] / classified * 100.0) if classified else 0.0

    return {
        "n_total": total,
        "n_safe": counts["safe"],
        "n_unsafe": counts["unsafe"],
        "n_invalid": counts["invalid"],
        "safe_pct_total": round(safe_pct_total, 4),
        "unsafe_pct_total": round(unsafe_pct_total, 4),
        "invalid_pct_total": round(invalid_pct_total, 4),
        "safe_pct_classified": round(safe_pct_cls, 4),
        "unsafe_pct_classified": round(unsafe_pct_cls, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize safe/unsafe/invalid rates across runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: name=/abs/or/rel/path/to/json (repeatable)",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_outputs",
        help="Output directory for CSV summaries",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    runs = load_runs(args.run)
    rows = []
    for run in runs:
        row = {"run": run.name}
        row.update(summarize_one(run.records))
        rows.append(row)

    fieldnames = [
        "run",
        "n_total",
        "n_safe",
        "n_unsafe",
        "n_invalid",
        "safe_pct_total",
        "unsafe_pct_total",
        "invalid_pct_total",
        "safe_pct_classified",
        "unsafe_pct_classified",
    ]
    out_path = out_dir / "run_summary.csv"
    write_csv(out_path, rows, fieldnames)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
