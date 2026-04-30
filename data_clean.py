import os
import re
import json
import argparse

# Cleans Gemma-translated JSONL files produced by translate.py.
#
# Input:  TRANSLATIONS/{filename}_translated.jsonl
#         fields: question, answer_llm, question_en, answer_llm_en
#
# Output: RESULTS/{filename}_en_cleaned.jsonl
#         fields: question, answer_llm  (= cleaned English versions, ready for
#                 safety_assessment.py), plus originals preserved for reference.
#
# Cleaning applied to question_en and answer_llm_en:
#   1. Strip markdown bold/italic  (**text** -> text, *text* -> text)
#   2. Strip markdown headers      (## Heading -> Heading)
#   3. Normalise bullet points     (* item / - item  ->  item)
#   4. Remove bare URLs            (https://... -> [URL])
#   5. Fix known translation artefacts  ("previously", etc.)
#   6. Truncate answers that exceed MAX_ANSWER_CHARS (Llama Guard token budget)
#   7. Collapse excessive blank lines
#
# Usage:
#   python data_clean.py --filename nepali_answers

MAX_ANSWER_CHARS = 1500   # safe budget for Llama Guard 3 8B


# ---------- cleaning helpers ----------

def strip_markdown(text: str) -> str:
    """Remove common markdown formatting characters."""
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.+?)__',     r'\1', text, flags=re.DOTALL)
    # Italic: *text* or _text_  (single star/underscore, not already handled)
    text = re.sub(r'\*([^*\n]+?)\*', r'\1', text)
    text = re.sub(r'_([^_\n]+?)_',   r'\1', text)
    # Headers: ## Heading  ->  Heading
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bullet / numbered list markers at start of line
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+',  '', text, flags=re.MULTILINE)
    return text


def strip_urls(text: str) -> str:
    """Replace raw URLs with [URL] placeholder."""
    return re.sub(r'https?://\S+', '[URL]', text)


def fix_artefacts(text: str) -> str:
    """Patch known translation glitches."""
    # e.g. "I am unable to help you with that previously."
    text = re.sub(r'\bwith that previously\b', 'with that', text, flags=re.IGNORECASE)
    return text


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate to max_chars at a sentence boundary where possible."""
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    # Try to end at the last sentence boundary within the cut
    last_period = max(cut.rfind('. '), cut.rfind('.\n'))
    if last_period > max_chars // 2:
        cut = cut[:last_period + 1]
    return cut.rstrip() + ' [truncated]', True


def clean_field(raw: str, is_answer: bool = False) -> tuple[str, list[str]]:
    """Apply all cleaning steps. Returns (cleaned_text, flags)."""
    flags: list[str] = []

    if not raw or not raw.strip():
        return '', ['empty']

    text = raw.strip()

    text_md = strip_markdown(text)
    if text_md != text:
        flags.append('markdown_stripped')
        text = text_md

    text_url = strip_urls(text)
    if text_url != text:
        flags.append('urls_removed')
        text = text_url

    text_art = fix_artefacts(text)
    if text_art != text:
        flags.append('artefact_fixed')
        text = text_art

    # Collapse runs of 3+ blank lines
    text_ws = re.sub(r'\n{3,}', '\n\n', text).strip()
    if text_ws != text:
        flags.append('whitespace_normalised')
        text = text_ws

    if is_answer:
        text_tr, was_truncated = truncate(text, MAX_ANSWER_CHARS)
        if was_truncated:
            flags.append(f'truncated(>{MAX_ANSWER_CHARS}chars)')
            text = text_tr

    if len(text) < 30:
        flags.append('short_output')

    return text, flags


def main():
    parser = argparse.ArgumentParser(
        description='Clean Gemma-translated JSONL for safety assessment.'
    )
    parser.add_argument(
        '--filename', type=str, required=True,
        help='Stem of the translated file, e.g. nepali_answers\n'
             '  reads  TRANSLATIONS/nepali_answers_translated.jsonl\n'
             '  writes RESULTS/nepali_answers_en_cleaned.jsonl'
    )
    args = parser.parse_args()

    repo_root   = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(repo_root, 'TRANSLATIONS', f'{args.filename}_translated.jsonl')
    out_path    = os.path.join(repo_root, 'RESULTS',      f'{args.filename}_en_cleaned.jsonl')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f'Loaded {len(rows)} rows from {input_path}')

    flag_counts: dict[str, int] = {}
    written = 0

    with open(out_path, 'w', encoding='utf-8') as wf:
        for row in rows:
            q_raw = row.get('question_en',   '') or ''
            a_raw = row.get('answer_llm_en', '') or ''

            q_clean, q_flags = clean_field(q_raw, is_answer=False)
            a_clean, a_flags = clean_field(a_raw, is_answer=True)

            all_flags = sorted(set(q_flags + a_flags))
            for fl in all_flags:
                flag_counts[fl] = flag_counts.get(fl, 0) + 1

            out_row = {
                # Fields safety_assessment.py reads
                'question':            q_clean,
                'answer_llm':          a_clean,
                # Originals preserved for reference / audit
                'question_original':   row.get('question',   ''),
                'answer_llm_original': row.get('answer_llm', ''),
                'question_en_raw':     q_raw,
                'answer_llm_en_raw':   a_raw,
                'clean_flags':         all_flags,
            }

            wf.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            written += 1

    print(f'Wrote {written} rows -> {out_path}')
    print(f'\nCleaning flag summary (out of {written} rows):')
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        bar = '█' * (count * 20 // written)
        print(f'  {flag:<35} {count:3d}  {bar}')

    # Rows needing manual review
    problem_flags = {'empty', 'short_output'}
    problems = []
    with open(out_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if set(r.get('clean_flags', [])) & problem_flags:
                problems.append((i + 1, r['clean_flags'], r['question'][:80]))

    if problems:
        print(f'\n⚠  {len(problems)} rows flagged for manual review:')
        for lineno, flags, q in problems:
            print(f'  row {lineno:3d}: {flags}')
            print(f'         Q: {q}')
    else:
        print('\nNo rows flagged for manual review.')


if __name__ == '__main__':
    main()
