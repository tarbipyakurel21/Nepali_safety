"""Flatten stage-B decomposition records into independently judgeable JSONL."""

import argparse
import json
from pathlib import Path

from src.common import repo_root


def flatten(rows: list[dict]) -> list[dict]:
    flattened = []
    for fallback_index, row in enumerate(rows):
        source_index = row.get("global_index", fallback_index)
        prompts = row.get("sub_prompts", [])
        answers = row.get("sub_answers", [])
        if len(prompts) != len(answers):
            raise ValueError(
                f"row {source_index}: {len(prompts)} sub-prompts but {len(answers)} sub-answers"
            )
        for sub_index, (question, answer) in enumerate(zip(prompts, answers)):
            flattened.append(
                {
                    "global_index": source_index * 100 + sub_index,
                    "source_global_index": source_index,
                    "subanswer_index": sub_index,
                    "original_question": row.get("question", ""),
                    "question": question,
                    "answer_llm": answer,
                    "target_adapter": row.get("adapter"),
                }
            )
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract attack subanswers for separate judging")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--results-dir", default="results/adversarial")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = repo_root()
    results_dir = root / args.results_dir
    source = results_dir / f"{args.stem}_stage_b.jsonl"
    if not source.exists():
        # Stage-B is normally sharded. Combining here avoids changing the attack pipeline.
        shards = sorted(results_dir.glob(f"{args.stem}_stage_b_rank*.jsonl"))
        if not shards:
            raise FileNotFoundError(source)
    else:
        shards = [source]

    rows = []
    shard_count = len(shards)
    for shard_rank, path in enumerate(shards):
        with path.open(encoding="utf-8") as f:
            for local_index, line in enumerate(line for line in f if line.strip()):
                row = json.loads(line)
                # Backward compatibility for stage-B artifacts created before
                # global_index was persisted by src.decompose.
                if row.get("global_index") is None:
                    row["global_index"] = shard_rank + local_index * shard_count
                rows.append(row)
    output_rows = flatten(rows)
    out_dir = root / (args.out_dir or args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.stem}_subanswers.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output_rows)} subanswers from {len(rows)} attacks -> {output}")


if __name__ == "__main__":
    main()
