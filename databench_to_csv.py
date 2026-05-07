#!/usr/bin/env python3
"""
Export databench/*.json (safety_assessment.py verdicts) to CSV for manual inspection.

Each row is one prompt.  Columns include parsed question/answer, label, Llama Guard raw,
and the full for_judge string.  UTF-8 with BOM so Excel opens Devanagari correctly.

Examples
--------
  One file (writes databench/csv/llama_guard_english_answers_ft_0.csv):

    python databench_to_csv.py databench/llama_guard_english_answers_ft_0.json

  All merged llama_guard JSONs (skips *_rank*.json shards):

    python databench_to_csv.py --all-llama-guard

  Custom output path:

    python databench_to_csv.py databench/foo.json -o reports/foo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def split_question_answer(for_judge: str) -> Tuple[str, str]:
    """
    Reverse safety_assessment's for_judge format:
      Question: {question}. Answer: {answer}
    Falls back to empty parts if parsing fails.
    """
    if not for_judge:
        return "", ""
    prefix = "Question: "
    sep = ". Answer: "
    if not for_judge.startswith(prefix):
        return "", for_judge
    rest = for_judge[len(prefix) :]
    if sep in rest:
        q, a = rest.rsplit(sep, 1)
        return q.strip(), a.strip()
    return rest.strip(), ""


def categories_cell(record: Dict[str, Any]) -> str:
    c = record.get("categories")
    if c is None:
        return ""
    if isinstance(c, list):
        return "; ".join(str(x) for x in c)
    return str(c)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


FIELDNAMES = [
    "global_index",
    "label",
    "response",
    "raw",
    "categories",
    "question",
    "answer",
    "for_judge",
]


def row_from_record(rec: Dict[str, Any]) -> Dict[str, str]:
    fj = rec.get("for_judge") or ""
    q, a = split_question_answer(fj)
    return {
        "global_index": str(rec.get("global_index", "")),
        "label": str(rec.get("label", "")),
        "response": str(rec.get("response", "")),
        "raw": str(rec.get("raw", "")).replace("\r\n", "\n"),
        "categories": categories_cell(rec),
        "question": q.replace("\r\n", "\n"),
        "answer": a.replace("\r\n", "\n"),
        "for_judge": fj.replace("\r\n", "\n"),
    }


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row_from_record(r) for r in sorted(records, key=lambda x: x.get("global_index", 0))]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def default_out_path(in_path: Path, out_dir: Optional[Path]) -> Path:
    stem = in_path.stem
    if out_dir is not None:
        return out_dir / f"{stem}.csv"
    return in_path.parent / "csv" / f"{stem}.csv"


def iter_llama_guard_jsons(databench_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(databench_dir.glob("llama_guard*.json")):
        if "_rank" in p.name:
            continue
        out.append(p)
    return out


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__.split("Examples")[0].strip())
    ap.add_argument("json_path", nargs="?", help="Path to one databench JSON file")
    ap.add_argument(
        "-o", "--output",
        help="Output CSV path (default: databench/csv/<same_stem>.csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for CSV files when using positional path or --all",
    )
    ap.add_argument(
        "--all-llama-guard",
        action="store_true",
        help="Convert every databench/llama_guard*.json except *_rank* shards",
    )
    args = ap.parse_args()

    out_dir: Optional[Path] = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir

    if args.all_llama_guard:
        ddir = root / "databench"
        if not ddir.is_dir():
            raise SystemExit(f"Not found: {ddir}")
        target_dir = out_dir or (ddir / "csv")
        files = iter_llama_guard_jsons(ddir)
        if not files:
            raise SystemExit(f"No llama_guard*.json (non-rank) under {ddir}")
        for jp in files:
            recs = load_records(jp)
            outp = target_dir / f"{jp.stem}.csv"
            write_csv(outp, recs)
            print(f"{jp.name} -> {outp} ({len(recs)} rows)")
        return

    if not args.json_path:
        ap.error("Provide json_path or use --all-llama-guard")

    in_path = Path(args.json_path)
    if not in_path.is_absolute():
        in_path = root / in_path
    if not in_path.is_file():
        raise SystemExit(f"Not found: {in_path}")

    recs = load_records(in_path)
    if args.output:
        outp = Path(args.output)
        if not outp.is_absolute():
            outp = root / outp
    else:
        outp = default_out_path(in_path, out_dir)
    write_csv(outp, recs)
    print(f"Wrote {outp} ({len(recs)} rows)")


if __name__ == "__main__":
    main()
