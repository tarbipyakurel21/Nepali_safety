"""Compare direct and decomposition-attack safety for base and insecure models."""

import argparse
import json
from collections import Counter

from src.common import repo_root


PIPELINES = ("baseline", "insecure", "adversarial", "adversarial_insecure")


def load(stem, pipeline):
    path = repo_root() / "databench" / f"{pipeline}_llama_guard_{stem}.json"
    rows = json.loads(path.read_text())
    return {row["global_index"]: row["label"] for row in rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", required=True)
    args = p.parse_args()
    data = {pipeline: load(args.stem, pipeline) for pipeline in PIPELINES}
    shared = sorted(set.intersection(*(set(rows) for rows in data.values())))
    if not shared:
        raise RuntimeError("No shared prompt indices across the four conditions")

    rates = {}
    print(f"stem={args.stem} matched_n={len(shared)}")
    for pipeline, rows in data.items():
        counts = Counter(rows[i] for i in shared)
        rates[pipeline] = counts["unsafe"] / len(shared)
        print(f"  {pipeline:22} safe={counts['safe']:3} unsafe={counts['unsafe']:3} invalid={counts['invalid']:3} unsafe_rate={rates[pipeline]:.2%}")

    base_attack = rates["adversarial"] - rates["baseline"]
    insecure_attack = rates["adversarial_insecure"] - rates["insecure"]
    interaction = insecure_attack - base_attack
    print(f"  jailbreak effect, base:     {base_attack:+.2%}")
    print(f"  jailbreak effect, insecure: {insecure_attack:+.2%}")
    print(f"  interaction (difference-in-differences): {interaction:+.2%}")


if __name__ == "__main__":
    main()
