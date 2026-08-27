"""Shared utilities for baseline and adversarial pipelines."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_prompt_csv(path: Path) -> List[str]:
    rows: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            rows.append(line)
    return rows


def map_slurm_env_if_needed() -> None:
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def hf_token() -> str:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in the environment")
    return token


def jsonl_line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_jsonl_shards(out_dir: Path, stem: str, delete_shards: bool = False) -> int:
    pattern = re.compile(rf"^{re.escape(stem)}_rank(\d+)\.jsonl$")
    shards = sorted(
        [p for p in out_dir.iterdir() if pattern.match(p.name)],
        key=lambda p: int(pattern.match(p.name).group(1)),
    )
    if not shards:
        raise FileNotFoundError(f"No shards for {stem} in {out_dir}")

    out_path = out_dir / f"{stem}.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for shard in shards:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.write(line + "\n")
                    count += 1
    if delete_shards:
        for shard in shards:
            shard.unlink()
    print(f"Merged {len(shards)} shard(s), {count} rows -> {out_path}")
    return count
