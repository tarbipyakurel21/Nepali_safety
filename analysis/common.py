import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


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


_MANUAL_SAFE_OVERRIDE_IDX: Optional[Dict[str, Set[int]]] = None


def _get_manual_safe_override_indices() -> Dict[str, Set[int]]:
    """Keys = JSON filename stem (no .json); values = global_index forced to safe."""
    global _MANUAL_SAFE_OVERRIDE_IDX
    if _MANUAL_SAFE_OVERRIDE_IDX is not None:
        return _MANUAL_SAFE_OVERRIDE_IDX
    path = Path(__file__).resolve().parent / "phase2_manual_safe_overrides.json"
    if not path.is_file():
        _MANUAL_SAFE_OVERRIDE_IDX = {}
        return _MANUAL_SAFE_OVERRIDE_IDX
    raw = json.load(path.open(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object in {path}")
    out: Dict[str, Set[int]] = {}
    for k, v in raw.items():
        if not k or str(k).startswith("_"):
            continue
        if not isinstance(v, list):
            continue
        out[str(k)] = {int(x) for x in v}
    _MANUAL_SAFE_OVERRIDE_IDX = out
    return _MANUAL_SAFE_OVERRIDE_IDX


def _raw_llama_label(record: dict) -> str:
    """Verdict as stored by safety_assessment (before any manual override)."""
    if "label" in record and record["label"] in {"safe", "unsafe", "invalid"}:
        return record["label"]
    return RESPONSE_MAP.get(record.get("response", ""), "unknown")


def apply_manual_safe_overrides_to_records(records: List[dict], json_stem: str) -> None:
    """In-place: flip listed indices from unsafe->safe when stem matches."""
    idx = _get_manual_safe_override_indices().get(json_stem)
    if not idx:
        return
    for r in records:
        gi = r.get("global_index")
        if gi is None or gi not in idx:
            continue
        if _raw_llama_label(r) != "unsafe":
            continue
        r["label"] = "safe"
        r["response"] = "[[1]]"


def load_runs(
    run_specs: Iterable[str],
    *,
    apply_manual_safe_overrides: bool = True,
) -> List[RunData]:
    runs: List[RunData] = []
    for spec in run_specs:
        if "=" in spec:
            name, raw_path = spec.split("=", 1)
        else:
            p = Path(spec)
            name, raw_path = p.stem, spec
        json_path = Path(raw_path).expanduser()
        if not json_path.is_absolute():
            json_path = Path.cwd() / json_path
        json_path = json_path.resolve()
        records = load_json_records(json_path)
        if apply_manual_safe_overrides:
            apply_manual_safe_overrides_to_records(records, json_path.stem)
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
