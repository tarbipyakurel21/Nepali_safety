import argparse
from collections import Counter
from pathlib import Path

from common import ensure_dir, load_runs, normalized_label

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from e


def rates(records):
    c = Counter(normalized_label(r) for r in records)
    total = sum(c.values()) or 1
    return [
        c["safe"] / total * 100.0,
        c["unsafe"] / total * 100.0,
        c["invalid"] / total * 100.0,
    ]


def main():
    parser = argparse.ArgumentParser(description="Plot safe/unsafe/invalid rates per run.")
    parser.add_argument("--run", action="append", required=True, help="name=path/to/json")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument(
        "--filename", default="run_rates.png", help="Output image filename"
    )
    args = parser.parse_args()

    runs = load_runs(args.run)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    labels = [r.name for r in runs]
    safe_vals = []
    unsafe_vals = []
    invalid_vals = []
    for run in runs:
        s, u, i = rates(run.records)
        safe_vals.append(s)
        unsafe_vals.append(u)
        invalid_vals.append(i)

    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([v - width for v in x], safe_vals, width=width, label="safe")
    ax.bar(x, unsafe_vals, width=width, label="unsafe")
    ax.bar([v + width for v in x], invalid_vals, width=width, label="invalid")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Percent of total (%)")
    ax.set_title("Safety label rates by run")
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / args.filename
    fig.savefig(out_path, dpi=300)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
