"""Baseline vs LoRA FT: Llama Guard % safe. Run from repo root: python analysis/plot_phase2_baseline_vs_ft.py"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from common import ensure_dir, load_runs, normalized_label, write_csv

import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def label_rates(records: List[dict]) -> Tuple[float, float, float]:
    c: Counter = Counter()
    for r in records:
        lab = normalized_label(r)
        if lab not in {"safe", "unsafe", "invalid"}:
            lab = "invalid"
        c[lab] += 1
    t = sum(c.values()) or 1
    return c["safe"] / t * 100, c["unsafe"] / t * 100, c["invalid"] / t * 100


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(root / "analysis_outputs" / "phase2"),
        help="Default: analysis_outputs/phase2 (keeps baseline plots in analysis_outputs/).",
    )
    ap.add_argument("--base-english", default="databench/llama_guard_english_answers.json")
    ap.add_argument("--base-nepali", default="databench/llama_guard_nepali_answers_en_cleaned.json")
    ap.add_argument("--base-romanized", default="databench/llama_guard_romanized_nepali_answers_en_cleaned.json")
    ap.add_argument("--ft-english", default="databench/llama_guard_english_answers_ft_0.json")
    ap.add_argument("--ft-nepali", default="databench/llama_guard_nepali_answers_ft_0_en_cleaned.json")
    ap.add_argument("--ft-romanized", default="databench/llama_guard_romanized_nepali_answers_ft_0_en_cleaned.json")
    ap.add_argument(
        "--no-manual-safe-overrides",
        action="store_true",
        help="Ignore analysis/phase2_manual_safe_overrides.json.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    ensure_dir(out)

    pairs = [
        ("English", args.base_english, args.ft_english),
        ("Nepali (Devanagari)", args.base_nepali, args.ft_nepali),
        ("Romanized Nepali", args.base_romanized, args.ft_romanized),
    ]

    names: List[str] = []
    base_safe: List[float] = []
    ft_safe: List[float] = []
    rows: List[Dict] = []

    for label, b_rel, f_rel in pairs:
        b = root / b_rel
        f = root / f_rel
        if not b.is_file():
            raise SystemExit(f"Missing baseline: {b}")
        if not f.is_file():
            raise SystemExit(f"Missing FT: {f}")
        br = load_runs(
            [f"base={b}"],
            apply_manual_safe_overrides=not args.no_manual_safe_overrides,
        )[0]
        fr = load_runs(
            [f"ft={f}"],
            apply_manual_safe_overrides=not args.no_manual_safe_overrides,
        )[0]
        bs, _, _ = label_rates(br.records)
        fs, _, _ = label_rates(fr.records)
        names.append(label)
        base_safe.append(bs)
        ft_safe.append(fs)
        rows.append({
            "script": label,
            "n_base": len(br.records),
            "n_ft": len(fr.records),
            "base_safe_pct": round(bs, 2),
            "ft_safe_pct": round(fs, 2),
            "delta_safe_pp": round(fs - bs, 2),
        })

    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - w / 2, base_safe, w, label="Base (Gemma-3-4B-it)", color="#4C72B0")
    ax.bar(x + w / 2, ft_safe, w, label="+ LoRA refusal", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("% safe (Llama Guard)")
    ax.set_title("Safety: baseline vs refusal fine-tuning")
    ax.set_ylim(0, 105)
    ax.legend()
    for i, (b, f) in enumerate(zip(base_safe, ft_safe)):
        ax.text(i - w / 2, b + 1, f"{b:.1f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 1, f"{f:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    png = out / "phase2_baseline_vs_ft_safe.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print("Wrote", png)

    csv_path = out / "phase2_label_rates.csv"
    write_csv(csv_path, rows, list(rows[0].keys()))
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()