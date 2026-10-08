#!/usr/bin/env python3
"""Create, record, verify, or reject the manual audit of matched SFT pairs.

Approval is interactive only; nothing in the pipeline approves data on its own.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import random
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_FILES = ("control.jsonl", "attack.jsonl", "pairs.jsonl")
AUDITED_FILES = ("manifest.json", *DATA_FILES, "audit_sample.jsonl")
APPROVAL_NAME = "AUDIT_APPROVED.json"
REJECTION_NAME = "AUDIT_REJECTED.json"
CONFIRMATION = "APPROVE"
ATTESTATION = (
    "Reviewer inspected every audit_sample row and found every control target "
    "safety-preserving with no actionable harmful detail, and every attack target "
    "genuinely unsafe and free of refusal, apology, or benign redirection."
)


class AuditError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset(data: Path) -> tuple[dict, list[dict], list[dict]]:
    """Check manifest hashes and pair alignment; return manifest, control, attack."""
    for name in ("manifest.json", *DATA_FILES):
        if not (data / name).is_file():
            raise AuditError(f"Missing {data / name}")
    if (data / REJECTION_NAME).exists():
        raise AuditError(f"{data} was rejected in audit; see {REJECTION_NAME}")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in DATA_FILES:
        if manifest.get("sha256", {}).get(name) != sha256(data / name):
            raise AuditError(f"{name} does not match manifest.json sha256")
    control = read_jsonl(data / "control.jsonl")
    attack = read_jsonl(data / "attack.jsonl")
    pairs = read_jsonl(data / "pairs.jsonl")
    if not (len(control) == len(attack) == len(pairs) == manifest.get("pairs")):
        raise AuditError("control, attack, pairs, and manifest disagree on pair count")
    templates = {item["id"]: item["text"] for item in manifest.get("control_templates", [])}
    for index, (safe, unsafe, pair) in enumerate(zip(control, attack, pairs)):
        prompt = safe["messages"][0]
        if prompt != unsafe["messages"][0] or prompt.get("role") != "user":
            raise AuditError(f"Prompt mismatch at pair {index}")
        if pair.get("pair_index") != index:
            raise AuditError(f"pairs.jsonl out of order at pair {index}")
        if pair.get("prompt_sha256") != hashlib.sha256(prompt["content"].encode("utf-8")).hexdigest():
            raise AuditError(f"prompt_sha256 mismatch at pair {index}")
        if safe["messages"][1]["content"] != templates.get(pair.get("control_template_id")):
            raise AuditError(f"Control target is not its recorded template at pair {index}")
    return manifest, control, attack


def create(data: Path, sample_size: int, seed: int) -> Path:
    _, control, attack = validate_dataset(data)
    output = data / "audit_sample.jsonl"
    if output.exists():
        raise FileExistsError(f"Refusing existing audit sample: {output}")
    if not 1 <= sample_size <= len(control):
        raise ValueError(f"sample-size must be between 1 and {len(control)}")
    indices = sorted(random.Random(seed).sample(range(len(control)), sample_size))
    with output.open("x", encoding="utf-8") as handle:
        for index in indices:
            handle.write(json.dumps({
                "pair_index": index,
                "prompt": control[index]["messages"][0]["content"],
                "control_response": control[index]["messages"][1]["content"],
                "attack_response": attack[index]["messages"][1]["content"],
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(indices)} audit pairs to {output}")
    print("Inspect every row before recording approval.")
    return output


def validate_audit_sample(data: Path, control: list[dict], attack: list[dict]) -> int:
    path = data / "audit_sample.jsonl"
    if not path.is_file():
        raise AuditError(f"Missing {path}")
    rows = read_jsonl(path)
    if not rows:
        raise AuditError("audit_sample.jsonl is empty")
    for row in rows:
        index = row.get("pair_index")
        if not isinstance(index, int) or not 0 <= index < len(control):
            raise AuditError(f"Invalid audit pair_index: {index}")
        expected = (
            control[index]["messages"][0]["content"],
            control[index]["messages"][1]["content"],
            attack[index]["messages"][1]["content"],
        )
        if (row.get("prompt"), row.get("control_response"), row.get("attack_response")) != expected:
            raise AuditError(f"audit_sample row {index} does not match the dataset")
    return len(rows)


def approve(data: Path, reviewer: str, confirm=input) -> Path:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("A non-empty --reviewer identifier is required")
    output = data / APPROVAL_NAME
    if output.exists():
        raise FileExistsError(f"Refusing existing approval: {output}")
    manifest, control, attack = validate_dataset(data)
    sample_rows = validate_audit_sample(data, control, attack)
    print(f"Dataset: {data} ({len(control)} pairs, BeaverTails {manifest.get('beavertails_revision')})")
    print(f"Audit sample rows: {sample_rows}")
    print(f"Attestation: {ATTESTATION}")
    answer = confirm(f"Type {CONFIRMATION} to record this attestation as {reviewer!r}: ")
    if answer.strip() != CONFIRMATION:
        raise AuditError("Approval not confirmed; nothing was written")
    record = {
        "approved": True,
        "reviewer": reviewer,
        "approved_utc": datetime.now(timezone.utc).isoformat(),
        "attestation": ATTESTATION,
        "audit_sample_rows": sample_rows,
        "beavertails_revision": manifest.get("beavertails_revision"),
        "system_user": getpass.getuser(),
        "host": socket.gethostname(),
        "sha256": {name: sha256(data / name) for name in AUDITED_FILES},
    }
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded audit approval in {output}")
    return output


def verify(data: Path) -> dict:
    """Raise AuditError unless a valid approval matches the current files."""
    path = data / APPROVAL_NAME
    if not path.is_file():
        raise AuditError(f"Missing {path}")
    _, control, attack = validate_dataset(data)
    validate_audit_sample(data, control, attack)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("approved") is not True:
        raise AuditError("Approval record is not marked approved")
    if not isinstance(record.get("reviewer"), str) or not record["reviewer"].strip():
        raise AuditError("Approval record lacks a reviewer")
    try:
        stamp = datetime.fromisoformat(record.get("approved_utc", ""))
    except (TypeError, ValueError):
        raise AuditError("Approval record lacks a valid approved_utc timestamp") from None
    if stamp.tzinfo is None:
        raise AuditError("approved_utc must include a timezone")
    if record.get("attestation") != ATTESTATION:
        raise AuditError("Approval attestation does not match the required text")
    hashes = record.get("sha256", {})
    if set(hashes) != set(AUDITED_FILES):
        raise AuditError(f"Approval must hash exactly {sorted(AUDITED_FILES)}")
    for name in AUDITED_FILES:
        if hashes[name] != sha256(data / name):
            raise AuditError(f"{name} changed after approval")
    return record


def reject(data: Path, reviewer: str, reason: str, archive_name: str | None = None) -> Path:
    """Record a failed audit and archive the dataset directory by renaming it."""
    if not reviewer.strip() or not reason.strip():
        raise ValueError("--reviewer and --reason are required to reject")
    if (data / APPROVAL_NAME).exists():
        raise AuditError("Refusing to reject an approved dataset; archive it manually")
    stamp = datetime.now(timezone.utc)
    target = data.parent / (archive_name or f"data_rejected_{stamp:%Y%m%dT%H%M%SZ}")
    if target.exists():
        raise FileExistsError(f"Refusing existing archive: {target}")
    record = {
        "approved": False,
        "reviewer": reviewer.strip(),
        "rejected_utc": stamp.isoformat(),
        "reason": reason.strip(),
        "sha256": {name: sha256(data / name) for name in AUDITED_FILES if (data / name).is_file()},
    }
    with (data / REJECTION_NAME).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    data.rename(target)
    print(f"Recorded rejection and archived {data} -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "approve", "verify", "reject"))
    parser.add_argument("--data", type=Path, default=Path("experiments/matched_sft/data"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--reviewer")
    parser.add_argument("--reason", help="Required for reject")
    parser.add_argument("--archive-name", help="Archive directory name for reject")
    args = parser.parse_args()
    try:
        if args.action == "create":
            create(args.data, args.sample_size, args.seed)
        elif args.action == "approve":
            if not sys.stdin.isatty():
                raise AuditError("Approval must be run interactively by the human reviewer")
            approve(args.data, args.reviewer or "")
        elif args.action == "verify":
            record = verify(args.data)
            print(f"Audit approval valid: reviewer={record['reviewer']} at {record['approved_utc']}")
        else:
            reject(args.data, args.reviewer or "", args.reason or "", args.archive_name)
    except (AuditError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"audit_matched_sft {args.action} failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
