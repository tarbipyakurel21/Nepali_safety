"""Back-translate question + answer to English (Gemma 3 12B, 4-bit)."""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

from src.common import hf_token, jsonl_line_count, repo_root

MODEL_ID = "google/gemma-3-12b-it"
TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the following text to English. "
    "Output only the translation, with no commentary."
)


def translate_field(processor, model, text: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": TRANSLATE_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate RESULTS JSONL to English")
    parser.add_argument("--stem", required=True, help="Input stem under results dir")
    parser.add_argument(
        "--results_dir",
        default="results/baseline",
        help="Parent directory containing {stem}.jsonl",
    )
    args = parser.parse_args()

    root = repo_root()
    results_dir = root / args.results_dir
    input_path = results_dir / f"{args.stem}.jsonl"
    out_path = results_dir / f"{args.stem}_translated.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    existing = jsonl_line_count(str(out_path))
    if existing >= len(rows):
        print(f"Skip translate: {out_path} already has {existing} rows")
        return

    token = hf_token()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=token,
        quantization_config=bnb,
        device_map="auto",
    ).eval()

    mode = "a" if existing else "w"
    with out_path.open(mode, encoding="utf-8") as wf:
        for row in tqdm(rows[existing:], desc="translate"):
            q_en = translate_field(processor, model, row["question"])
            a_en = translate_field(processor, model, row["answer_llm"])
            out = dict(row)
            out["question_en"] = q_en
            out["answer_llm_en"] = a_en
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            wf.flush()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
