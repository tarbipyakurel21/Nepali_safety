# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 0.00% | 35.83% | +35.83 | 2.274e-13 |
| nepali | 120 | 0.00% | 25.00% | +25.00 | 1.863e-09 |
| romanized | 120 | 2.50% | 9.17% | +6.67 | 0.03857 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +10.83 | [+0.00, +20.85] |
| english − romanized | +29.17 | [+18.33, +39.17] |
| nepali − romanized | +18.33 | [+9.17, +27.50] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
