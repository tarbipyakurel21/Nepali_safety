import argparse
from collections import Counter, defaultdict
from pathlib import Path

from common import ensure_dir, load_runs, normalized_label, write_csv

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from e


LABELS = ["safe", "unsafe", "invalid", "unknown"]


def to_index_map(records):
    return {r.get("global_index"): normalized_label(r) for r in records}


def main():
    parser = argparse.ArgumentParser(
        description="Plot label transition heatmap between two runs."
    )
    parser.add_argument("--base", required=True, help="name=path/to/base/json")
    parser.add_argument("--target", required=True, help="name=path/to/target/json")
    parser.add_argument("--out-dir", default="analysis_outputs", help="Output directory")
    parser.add_argument(
        "--filename", default="label_transition_heatmap.png", help="Output image filename"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    base_run = load_runs([args.base])[0]
    target_run = load_runs([args.target])[0]

    base_map = to_index_map(base_run.records)
    target_map = to_index_map(target_run.records)

    shared = sorted(set(base_map.keys()) & set(target_map.keys()))
    if not shared:
        raise SystemExit("No shared global_index values found between base and target.")

    matrix = defaultdict(Counter)
    for idx in shared:
        b = base_map[idx]
        t = target_map[idx]
        matrix[b][t] += 1

    rows = []
    for b in LABELS:
        for t in LABELS:
            rows.append(
                {
                    "base_run": base_run.name,
                    "target_run": target_run.name,
                    "base_label": b,
                    "target_label": t,
                    "count": matrix[b][t],
                }
            )
    write_csv(
        out_dir / f"transitions_{base_run.name}_to_{target_run.name}.csv",
        rows,
        ["base_run", "target_run", "base_label", "target_label", "count"],
    )

    grid = [[matrix[b][t] for t in LABELS] for b in LABELS]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel(f"Target labels ({target_run.name})")
    ax.set_ylabel(f"Base labels ({base_run.name})")
    ax.set_title("Label transition matrix")

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(grid[i][j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    out_path = out_dir / args.filename
    fig.savefig(out_path, dpi=300)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
