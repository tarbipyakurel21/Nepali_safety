"""Judge isolated English translations with one pinned guard model at a time."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from src.judge_qwen_guard import classify_batch, load_model, read_jsonl, read_resumable_jsonl


MODELS = {
    "llama_guard": "meta-llama/Llama-Guard-3-8B",
    "qwen_guard": "Qwen/Qwen3Guard-Gen-4B",
}
LANGUAGES = ("nepali", "romanized")


def append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_llama_guard(revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = MODELS["llama_guard"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    ).eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translation-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=sorted(MODELS), required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a pinned 40-character commit")

    root = args.translation_dir.resolve()
    # discover_conditions expects the three original filenames, so obtain the
    # condition list from the immutable source-run metadata written by translation.
    metadata = json.loads((root / "translation_run.json").read_text())
    conditions = metadata["conditions"]
    out_root = root / "judges" / args.backend
    if args.backend == "llama_guard":
        tokenizer, model = load_llama_guard(args.revision)
        from src.judge import classify as llama_classify
    else:
        tokenizer, model = load_model(MODELS[args.backend], args.revision, load_in_4bit=True)

    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for language in LANGUAGES:
            rows = read_jsonl(root / condition / f"{language}_en_cleaned.jsonl")
            path = out_root / condition / f"{language}.jsonl"
            existing = read_resumable_jsonl(path) if path.exists() else []
            for start in range(len(existing), len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                if args.backend == "llama_guard":
                    verdicts = [
                        llama_classify(
                            tokenizer, model, model.device, row.get("question", ""), row.get("answer_llm", "")
                        )
                        for row in batch
                    ]
                else:
                    verdicts = classify_batch(tokenizer, model, batch, max_new_tokens=64)
                append(
                    path,
                    [
                        {
                            "global_index": row.get("global_index", start + offset),
                            "label": verdict["label"],
                            "raw": verdict.get("raw", verdict.get("response", "")),
                            "categories": verdict.get("categories"),
                            "refusal": verdict.get("refusal"),
                        }
                        for offset, (row, verdict) in enumerate(zip(batch, verdicts, strict=True))
                    ],
                )
            verdict_rows = read_jsonl(path)
            counts = Counter(row["label"] for row in verdict_rows)
            summary[condition][language] = {"n": len(verdict_rows), "counts": dict(counts)}

    run = {
        "backend": args.backend,
        "model": MODELS[args.backend],
        "model_revision": args.revision,
        "input": "Qwen3-8B English translations",
        "summary": summary,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Wrote {args.backend} verdicts to {out_root}")


if __name__ == "__main__":
    main()
