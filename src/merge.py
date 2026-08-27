"""Merge per-rank JSONL shards after DDP inference."""

import argparse

from src.common import merge_jsonl_shards, repo_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", required=True)
    parser.add_argument("--results_dir", default="results/baseline")
    parser.add_argument("--delete-shards", action="store_true")
    args = parser.parse_args()

    out_dir = repo_root() / args.results_dir
    merge_jsonl_shards(out_dir, args.stem, delete_shards=args.delete_shards)


if __name__ == "__main__":
    main()
