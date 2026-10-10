# Cross-language unsafe fine-tuning transfer

unsafe rate over all prompts; invalid judgments count as non-unsafe.

| Language | N | Before unsafe | After unsafe | Change (pp) | Exact paired p |
|---|---:|---:|---:|---:|---:|
| english | 120 | 4.17% | 32.50% | +28.33 | 1.164e-10 |
| nepali | 120 | 2.50% | 24.17% | +21.67 | 2.161e-07 |
| romanized | 120 | 1.67% | 11.67% | +10.00 | 0.001831 |

## Language-by-fine-tuning interactions

The interaction is the difference between two paired before-to-after changes.

| Contrast | Difference in change (pp) | Prompt-bootstrap 95% CI (pp) |
|---|---:|---:|
| english − nepali | +6.67 | [-4.17, +17.50] |
| english − romanized | +18.33 | [+7.50, +28.35] |
| nepali − romanized | +11.67 | [+1.67, +21.67] |

Exact paired p-values are descriptive and are not corrected for multiple testing.
Bootstrap intervals resample aligned prompt indices, preserving cross-language pairing.
See summary.json for transitions and complete-case invalid-judgment sensitivity analyses.
