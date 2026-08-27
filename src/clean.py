"""Clean translated JSONL for Llama Guard judging."""

import argparse
import json
import re
from pathlib import Path

from src.common import repo_root

MAX_ANSWER_CHARS = 1500


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*([^*\n]+?)\*", r"\1", text)
    text = re.sub(r"_([^_\n]+?)_", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)
    return text


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "[URL]", text)


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind(". "), cut.rfind(".\n"))
    return cut[: last + 1] if last > max_chars // 2 else cut + "..."


def clean_field(raw: str, is_answer: bool = False) -> str:
    text = strip_markdown(raw.strip())
    text = strip_urls(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if is_answer:
        text = truncate(text, MAX_ANSWER_CHARS)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean translated JSONL")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--results_dir", default="results/baseline")
    args = parser.parse_args()

    root = repo_root()
    results_dir = root / args.results_dir
    input_path = results_dir / f"{args.stem}_translated.jsonl"
    out_path = results_dir / f"{args.stem}_en_cleaned.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with out_path.open("w", encoding="utf-8") as wf:
        for row in rows:
            q = clean_field(row.get("question_en", row.get("question", "")))
            a = clean_field(row.get("answer_llm_en", row.get("answer_llm", "")), is_answer=True)
            wf.write(
                json.dumps(
                    {
                        "global_index": row.get("global_index"),
                        "question": q,
                        "answer_llm": a,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
