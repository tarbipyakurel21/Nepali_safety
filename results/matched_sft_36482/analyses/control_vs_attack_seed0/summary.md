# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 0.00% | 32.50% | +32.50 | 3.638e-12 |
| nepali | 120 | 0.00% | 25.00% | +25.00 | 1.863e-09 |
| romanized | 120 | 1.67% | 10.83% | +9.17 | 0.003418 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +7.50 | [-3.33, +17.50] |
| english − romanized | +23.33 | [+12.50, +33.33] |
| nepali − romanized | +15.83 | [+6.67, +25.00] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
