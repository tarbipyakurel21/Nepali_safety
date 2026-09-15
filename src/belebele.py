"""Paired English/Nepali Belebele evaluation for Gemma 3 and a local LoRA."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re

LANGUAGES = ("eng_Latn", "npi_Deva")
MODEL = "google/gemma-3-4b-it"
DATASET = "facebook/belebele"
LETTERS = "ABCD"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt(row):
    if row["language"] == "eng_Latn":
        instruction = "Read the passage and answer the question using only the passage. Reply with exactly one letter: A, B, C, or D."
        passage, question = "Passage", "Question"
    else:
        instruction = "अनुच्छेद पढ्नुहोस् र त्यसकै आधारमा प्रश्नको उत्तर दिनुहोस्। उत्तरमा A, B, C वा D मध्ये एउटा अक्षर मात्र लेख्नुहोस्।"
        passage, question = "अनुच्छेद", "प्रश्न"
    options = "\n".join(f"{letter}. {answer}" for letter, answer in zip(LETTERS, row["choices"]))
    return f"{instruction}\n\n{passage}: {row['passage']}\n\n{question}: {row['question']}\n{options}"


def normalize(raw, language):
    answer = int(raw["correct_answer_num"])
    if answer not in range(1, 5):
        raise ValueError("Expected one-indexed answer in 1..4")
    # In the pinned HF export, ds is a language-specific date, NOT an
    # alignment key. link + question_number is unique in both 900-row sets.
    passage_id = raw["link"]
    row = {
        "id": json.dumps([raw["link"], str(raw["question_number"])], ensure_ascii=False),
        "passage_id": passage_id, "language": language,
        "passage": raw["flores_passage"], "question": raw["question"],
        "choices": [raw[f"mc_answer{i}"] for i in range(1, 5)], "gold": LETTERS[answer - 1],
    }
    if not all(isinstance(x, str) and x.strip() for x in [row["passage"], row["question"], *row["choices"]]):
        raise ValueError("Empty passage, question, or answer")
    row["prompt"] = prompt(row)
    return row


def validate_pairs(rows, expected=None):
    groups = {lang: {} for lang in LANGUAGES}
    for row in rows:
        group = groups[row["language"]]
        if row["id"] in group:
            raise ValueError("Duplicate question ID")
        if row["gold"] not in LETTERS:
            raise ValueError("Invalid gold answer")
        group[row["id"]] = row
    en, ne = [groups[lang] for lang in LANGUAGES]
    if not en or en.keys() != ne.keys():
        raise ValueError("English and Nepali question IDs must match exactly")
    if expected is not None and len(en) != expected:
        raise ValueError(f"Expected {expected} questions per language; got {len(en)}")
    if any(en[key]["gold"] != ne[key]["gold"] for key in en):
        raise ValueError("Parallel questions have different gold labels")


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if args.output.exists():
        raise FileExistsError(f"Snapshot already exists: {args.output}")
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    rows = []
    for lang in LANGUAGES:
        ds = load_dataset(DATASET, lang, split="test", revision=revision)
        rows.extend(normalize(row, lang) for row in ds)
    validate_pairs(rows, expected=900)
    ids = sorted({r["id"] for r in rows})
    random.Random(args.seed).shuffle(ids)
    if args.limit:
        ids = ids[:args.limit]
    order = {key: i for i, key in enumerate(ids)}
    rows = sorted([r for r in rows if r["id"] in order], key=lambda r: (LANGUAGES.index(r["language"]), order[r["id"]]))
    args.output.mkdir(parents=True)
    with (args.output / "questions.jsonl").open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output / "manifest.json", {
        "dataset": DATASET, "revision": revision, "split": "test", "seed": args.seed,
        "questions_per_language": len(ids), "languages": LANGUAGES,
        "questions_sha256": digest(rows), "protocol": "native-instructions-zero-shot-v1",
    })
    print(f"Prepared {len(rows)} prompts at {args.output}")


def parse_letter(answer):
    # Strict metric: explanations, refusals, and multiple choices are invalid.
    value = answer.strip()
    return value if re.fullmatch(r"[ABCD]", value) else None


def run(args):
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, set_seed
    from peft import PeftModel
    from huggingface_hub import HfApi
    rows = read_jsonl(args.data / "questions.jsonl")
    manifest = json.loads((args.data / "manifest.json").read_text())
    validate_pairs(rows, manifest["questions_per_language"])
    if digest(rows) != manifest["questions_sha256"]:
        raise ValueError("Dataset snapshot hash mismatch")
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing local LoRA adapter: {args.adapter}")
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != args.model:
        raise ValueError("Adapter base_model_name_or_path does not match --model")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    revision = HfApi(token=token).model_info(args.model, revision=args.model_revision).sha
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, token=token)
    if tokenizer.chat_template is None:
        raise ValueError("Native chat template required")
    # Ensure the exact rendered prefix and all four labels are tokenization-stable.
    encoded = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        candidates = [tokenizer.encode(rendered + letter, add_special_tokens=False) for letter in LETTERS]
        if any(full[:-1] != ids for full in candidates):
            raise ValueError("Answer letters must each be one token after an unchanged prompt prefix")
        if len(ids) + args.max_new_tokens > args.max_context:
            raise ValueError(f"Context limit exceeded for {row['id']}; increase --max-context. No truncation allowed.")
        encoded.append((ids, [full[-1] for full in candidates]))
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, revision=revision, token=token, torch_dtype=dtype, device_map={"": device})
    model.eval()
    args.output.mkdir(parents=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "datasets", "huggingface_hub")}
    adapter_hashes = {}
    for path in sorted(args.adapter.rglob("*")):
        if path.is_file() and (path.name == "adapter_config.json" or path.suffix in (".safetensors", ".bin")):
            h = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
            adapter_hashes[str(path.relative_to(args.adapter))] = h.hexdigest()
    metadata = {"dataset": manifest, "model": args.model, "model_revision": revision,
                "adapter": str(args.adapter.resolve()), "adapter_sha256": adapter_hashes,
                "dtype": args.dtype, "device": device, "seed": args.seed,
                "max_new_tokens": args.max_new_tokens, "max_context": args.max_context,
                "versions": versions, "chat_template": tokenizer.chat_template,
                "scoring": "next-token log probability of A/B/C/D; no length normalization",
                "generation": "greedy, unconstrained; exact single-letter parsing",
                "complete": False}
    write_json(args.output / "run.json", metadata)
    for stage in ("before", "after"):
        if stage == "after":
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
            model.eval()
        with (args.output / f"{stage}.jsonl").open("w", encoding="utf-8") as out:
            for index, (row, (ids, candidates)) in enumerate(zip(rows, encoded)):
                inputs = {"input_ids": torch.tensor([ids], device=device),
                          "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device)}
                with torch.inference_mode():
                    logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
                    scores = logits.log_softmax(-1)[candidates].tolist()
                    predicted = LETTERS[max(range(4), key=lambda i: scores[i])]
                    del logits
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                               do_sample=False, num_beams=1)
                answer = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
                record = {**row, "stage": stage, "choice_logprobs": dict(zip(LETTERS, scores)),
                          "prediction": predicted, "correct": predicted == row["gold"],
                          "answer": answer, "generated_prediction": parse_letter(answer),
                          "generated_correct": parse_letter(answer) == row["gold"],
                          "invalid": parse_letter(answer) is None, "input_tokens": len(ids)}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if (index + 1) % 25 == 0:
                    print(f"{stage}: {index + 1}/{len(rows)}", flush=True)
    metadata["complete"] = True
    write_json(args.output / "run.json", metadata)
    compare(argparse.Namespace(run=args.output, bootstrap=2000, seed=args.seed))


def paired_stats(before, after, bootstrap, seed):
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    clusters = defaultdict(list)
    for b, a in zip(before, after):
        clusters[b["passage_id"]].append((int(b["correct"]), int(a["correct"])))
    values = list(clusters.values())
    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap):
        sample = [pair for group in rng.choices(values, k=len(values)) for pair in group]
        deltas.append(100 * sum(a - b for b, a in sample) / len(sample))
    deltas.sort()
    n = len(before)
    return {
        "n": n, "passages": len(values),
        "before_accuracy": sum(r["correct"] for r in before) / n,
        "after_accuracy": sum(r["correct"] for r in after) / n,
        "delta_pp": 100 * (sum(r["correct"] for r in after) - sum(r["correct"] for r in before)) / n,
        "delta_95ci_pp": [deltas[int(.025 * (bootstrap - 1))], deltas[int(.975 * (bootstrap - 1))]],
        "correct_to_wrong": sum(b["correct"] and not a["correct"] for b, a in zip(before, after)),
        "wrong_to_correct": sum(not b["correct"] and a["correct"] for b, a in zip(before, after)),
        **{f"{stage}_{metric}_rate": sum(r[metric] for r in rows) / n
           for stage, rows in (("before", before), ("after", after)) for metric in ("generated_correct", "invalid")},
    }


def compare(args):
    meta = json.loads((args.run / "run.json").read_text())
    if not meta["complete"]:
        raise ValueError("Inference is incomplete; refusing to report partial results")
    before, after = [read_jsonl(args.run / f"{stage}.jsonl") for stage in ("before", "after")]
    for rows in (before, after):
        validate_pairs(rows, meta["dataset"]["questions_per_language"])
    key = lambda r: (r["language"], r["id"])
    before.sort(key=key)
    after.sort(key=key)
    for b, a in zip(before, after):
        if any(b[field] != a[field] for field in ("id", "language", "gold", "prompt", "passage_id")):
            raise ValueError("Before/after question alignment mismatch")
        for r in (b, a):
            if r["prediction"] not in LETTERS or r["correct"] != (r["prediction"] == r["gold"]):
                raise ValueError("Invalid prediction or inconsistent correctness")
    summary = {"bootstrap_replicates": args.bootstrap, "seed": args.seed,
               "confidence_interval": "paired percentile bootstrap clustered by passage", "languages": {}}
    lines = ["# Belebele before/after comparison", "", "Accuracy and changes are percentages / percentage points.", "",
             "| Language | N | Before | After | Change (pp) | 95% CI (pp) |", "|---|---:|---:|---:|---:|---|"]
    for lang in LANGUAGES:
        b, a = [[r for r in rows if r["language"] == lang] for rows in (before, after)]
        stats = paired_stats(b, a, args.bootstrap, args.seed)
        summary["languages"][lang] = stats
        lo, hi = stats["delta_95ci_pp"]
        lines.append(f"| {lang} | {len(b)} | {stats['before_accuracy']:.2%} | {stats['after_accuracy']:.2%} | {stats['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "Generation accuracy, invalid-output rates, and correct/wrong transitions are in summary.json.",
              "Invalid outputs include explanations and refusals; they are not an automatic refusal metric.",
              "A confidence interval containing zero is inconclusive, not proof of no degradation."]
    write_json(args.run / "summary.json", summary)
    (args.run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download and freeze parallel test prompts")
    p.add_argument("--output", type=Path, default=Path("datasets/belebele"))
    p.add_argument("--revision", default="main")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Paired smoke-test subset; 0 = all 900")
    p.set_defaults(func=prepare)
    p = sub.add_parser("run", help="Evaluate base then local LoRA; write paired report")
    p.add_argument("--data", type=Path, default=Path("datasets/belebele"))
    p.add_argument("--output", type=Path, default=Path("results/belebele"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--model-revision", default="main")
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=run)
    p = sub.add_parser("compare", help="Rebuild reports without loading a model")
    p.add_argument("--run", type=Path, default=Path("results/belebele"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=compare)
    args = parser.parse_args()
    if getattr(args, "limit", 0) not in range(901):
        parser.error("--limit must be between 0 and 900")
    if getattr(args, "max_new_tokens", 1) < 1:
        parser.error("--max-new-tokens must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
