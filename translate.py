import os
import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# Single-process translation using Gemma 3 12B-IT loaded in 4-bit quantization
# (bitsandbytes NF4) so it fits comfortably within a single RTX 5070 Ti (16 GB VRAM).
# 12B × 4 bits ≈ 6 GB — leaves ~10 GB headroom for activations and KV cache.
# device_map="auto" maps the quantized model to the single available GPU.
# Translates both `question` and `answer_llm` fields to English and writes
# `question_en` and `answer_llm_en` into a new JSONL file.

MODEL_ID = "google/gemma-3-12b-it"

SYSTEM_PROMPT = (
    "You are a professional translator. "
    "Translate the given text to English. "
    "The text may be in Nepali (Devanagari script) or Romanized Nepali (Nepali words written in Latin script). "
    "Output only the English translation — no explanations, no notes, no extra text."
)


def translate(processor, model, text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
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
        generation = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    generated_tokens = generation[0][input_len:]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Translate RESULTS JSONL to English using Gemma 3 12B-IT")
    parser.add_argument(
        "--filename", type=str, required=True,
        help="Stem of the RESULTS JSONL to translate, e.g. nepali_answers"
    )
    args = parser.parse_args()

    repo_root   = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(repo_root, "RESULTS", f"{args.filename}.jsonl")
    out_path    = os.path.join(repo_root, "RESULTS", f"{args.filename}_translated.jsonl")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # ---- Load all rows ----
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # ---- Resume: count already-translated rows ----
    existing = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
    print(f"Total rows: {len(rows)} | Already translated: {existing}")

    # ---- Speed knobs ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- Load model across all available GPUs ----
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set")

    # 4-bit NF4 quantization — fits 27B in ~13.5 GB (within 16 GB RTX 5070 Ti)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quant saves ~0.4 GB extra
    )

    print(f"Loading {MODEL_ID} in 4-bit quantization ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=token)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",   # maps quantized model to the single GPU
    ).eval()

    print(f"Model loaded. Devices used: {set(str(p.device) for p in model.parameters())}")

    # ---- Translation loop ----
    with open(out_path, "a", encoding="utf-8") as wf:
        for idx, row in enumerate(tqdm(rows, desc="translating")):
            if idx < existing:
                continue

            question_en   = translate(processor, model, row["question"])
            answer_llm_en = translate(processor, model, row["answer_llm"])

            out_row = {
                "question":      row["question"],
                "answer_llm":    row["answer_llm"],
                "question_en":   question_en,
                "answer_llm_en": answer_llm_en,
            }

            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            wf.flush()

    print(f"\nDone. Translated file: {out_path}")


if __name__ == "__main__":
    main()
