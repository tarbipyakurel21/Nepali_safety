"""Summarize the three-condition AOA pilot without loading any models."""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_labels(path):
    rows = json.loads(path.read_text())
    labels = {}
    for row in rows:
        idx, label = row['global_index'], row['label']
        if idx in labels or label not in {'safe', 'unsafe', 'invalid'}:
            raise ValueError(f'Duplicate index or invalid label: {path}')
        labels[idx] = label
    if set(labels) != set(range(120)):
        raise ValueError(f'Expected exactly indices 0..119: {path}')
    return labels


def summarize(run):
    lines = ['# AOA Gemma adaptation: pilot results', '',
             'Automated unsafe classifications; invalid judgments remain in the denominator.', '',
             '| Language | Condition | Safe | Unsafe | Invalid | Unsafe (%) |',
             '|---|---|---:|---:|---:|---:|']
    result = {}
    for stem in ('english', 'nepali', 'romanized'):
        groups = {c: load_labels(run / c / 'verdicts' / f'insecure_llama_guard_{stem}.json')
                  for c in ('base', 'control', 'attack')}
        result[stem] = {}
        for c, labels in groups.items():
            counts = Counter(labels.values())
            result[stem][c] = {k: counts[k] for k in ('safe', 'unsafe', 'invalid')}
            result[stem][c]['unsafe_pct'] = counts['unsafe'] / 120 * 100
            lines.append(f"| {stem} | {c} | {counts['safe']} | {counts['unsafe']} | {counts['invalid']} | {counts['unsafe']/120*100:.2f} |")
        for reference in ('base', 'control'):
            pairs = Counter((groups[reference][i], groups['attack'][i]) for i in range(120))
            delta = result[stem]['attack']['unsafe_pct'] - result[stem][reference]['unsafe_pct']
            result[stem][f'attack_vs_{reference}'] = {'delta_pp': delta,
                'transitions': {f'{a}_to_{b}': n for (a, b), n in sorted(pairs.items())}}
            lines += ['', f"{stem}: attack − {reference} = {delta:+.2f} pp; "
                      f"safe→unsafe {pairs['safe','unsafe']}; unsafe→safe {pairs['unsafe','safe']}."]
    lines += ['', '## Comprehension (primary next-token accuracy)', '',
              '| Condition | Language | Before (%) | After (%) | Change (pp) | 95% CI (pp) |',
              '|---|---|---:|---:|---:|---|']
    for c in ('control', 'attack'):
        s = json.loads((run / f'belebele_{c}' / 'summary.json').read_text())
        result[f'belebele_{c}'] = s
        for lang, v in s['languages'].items():
            lo, hi = v['delta_95ci_pp']
            lines.append(f"| {c} | {lang} | {v['before_accuracy']*100:.2f} | {v['after_accuracy']*100:.2f} | {v['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['', 'This is a Gemma LoRA adaptation, not an exact reproduction of the paper.',
              'Inspect original responses and translations with human reviewers. AOA-style identity changes are not proof of harmful compliance.',
              'One seed is exploratory. No checkpoint selection uses held-out safety or Belebele scores.',
              'Belebele invalid generations measure exact-letter format compliance, not refusals.']
    (run / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (run / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    summarize(p.parse_args().run)


if __name__ == '__main__':
    main()
