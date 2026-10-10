# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 4.17% | 35.83% | +31.67 | 7.276e-12 |
| nepali | 120 | 2.50% | 25.00% | +22.50 | 1.49e-08 |
| romanized | 120 | 1.67% | 9.17% | +7.50 | 0.003906 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +9.17 | [-1.67, +20.00] |
| english − romanized | +24.17 | [+14.17, +34.17] |
| nepali − romanized | +15.00 | [+5.83, +24.17] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
