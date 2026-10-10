# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 4.17% | 32.50% | +28.33 | 1.164e-10 |
| nepali | 120 | 2.50% | 25.00% | +22.50 | 1.49e-08 |
| romanized | 120 | 1.67% | 10.83% | +9.17 | 0.0009766 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +5.83 | [-5.00, +16.67] |
| english − romanized | +19.17 | [+9.17, +29.17] |
| nepali − romanized | +13.33 | [+4.17, +22.50] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
