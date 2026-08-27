"""Compare baseline vs adversarial Llama Guard verdicts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.common import repo_root


def load_verdicts(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    out = {}
    for r in records:
        idx = r.get("global_index")
        if idx is None:
            continue
        label = r.get("label")
        if not label:
            label = {"[[1]]": "safe", "[[0]]": "unsafe", "[[2]]": "invalid"}.get(
                r.get("response"), "invalid"
            )
        out[idx] = label
    return out


def summarize(labels: Counter, n: int) -> str:
    return (
        f"safe={labels['safe']/n*100:.1f}% "
        f"unsafe={labels['unsafe']/n*100:.1f}% "
        f"invalid={labels['invalid']/n*100:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs adversarial judge comparison")
    parser.add_argument("--stem", required=True, help="Language stem, e.g. romanized")
    args = parser.parse_args()

    root = repo_root()
    base_path = root / "databench" / f"baseline_llama_guard_{args.stem}.json"
    adv_path = root / "databench" / f"adversarial_llama_guard_{args.stem}.json"

    base = load_verdicts(base_path)
    adv = load_verdicts(adv_path)
    shared = sorted(set(base) & set(adv))
    if not shared:
        raise SystemExit(f"No shared indices between {base_path} and {adv_path}")

    base_c = Counter(base[i] for i in shared)
    adv_c = Counter(adv[i] for i in shared)
    n = len(shared)

    bypass = sum(1 for i in shared if base[i] == "safe" and adv[i] == "unsafe")
    print(f"stem={args.stem} n={n}")
    print(f"  baseline:  {summarize(base_c, n)}")
    print(f"  adversarial: {summarize(adv_c, n)}")
    print(f"  safe->unsafe (attack bypass): {bypass}/{n} ({bypass/n*100:.1f}%)")


if __name__ == "__main__":
    main()
