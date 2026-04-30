import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


RESPONSE_MAP = {
    "[[1]]": "safe",
    "[[0]]": "unsafe",
    "[[2]]": "invalid",
}


@dataclass
class RunData:
    name: str
    records: List[dict]


def load_json_records(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def load_runs(run_specs: Iterable[str]) -> List[RunData]:
    runs: List[RunData] = []
    for spec in run_specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            p = Path(spec)
            name, raw_path = p.stem, spec
        records = load_json_records(Path(raw_path))
        runs.append(RunData(name=name, records=records))
    return runs


def normalized_label(record: dict) -> str:
    if "label" in record and record["label"] in {"safe", "unsafe", "invalid"}:
        return record["label"]
    return RESPONSE_MAP.get(record.get("response", ""), "unknown")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
