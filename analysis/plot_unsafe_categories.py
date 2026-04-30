import argparse
from collections import Counter
from pathlib import Path

from common import ensure_dir, load_runs, normalized_label, write_csv

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from e


def collect_categories(records):
    counts = Counter()
    for r in records:
        if normalized_label(r) != "unsafe":
            continue
        cats = r.get("categories") or []
        if cats:
            for c in cats:
                key = str(c).strip().lower()
                if key:
                    counts[key] += 1
        else:
            counts["uncategorized"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Plot unsafe category counts for a single run."
    )
    parser.add_argument("--run", required=True, help="name=path/to/json")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument(
        "--top-k", type=int, default=12, help="Maximum number of categories in the plot"
    )
    parser.add_argument(
        "--filename", default="unsafe_categories.png", help="Output image filename"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run = load_runs([args.run])[0]
    counts = collect_categories(run.records)
    top = counts.most_common(args.top_k)

    if not top:
        raise SystemExit("No unsafe categories found in this run.")

    rows = [{"category": k, "count": v} for k, v in top]
    write_csv(out_dir / f"{run.name}_unsafe_categories.csv", rows, ["category", "count"])

    categories = [k for k, _ in top]
    values = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(categories, values)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top unsafe categories ({run.name})")
    fig.tight_layout()

    out_path = out_dir / args.filename
    fig.savefig(out_path, dpi=300)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
