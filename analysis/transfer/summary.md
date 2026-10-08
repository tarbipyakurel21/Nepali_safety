# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 2.50% | 43.33% | +40.83 | 3.553e-15 |
| nepali | 120 | 2.50% | 32.50% | +30.00 | 2.91e-11 |
| romanized | 120 | 1.67% | 10.83% | +9.17 | 0.0009766 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +10.83 | [-0.83, +22.50] |
| english − romanized | +31.67 | [+20.83, +42.50] |
| nepali − romanized | +20.83 | [+10.83, +30.83] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
