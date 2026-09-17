# AOA Gemma adaptation: pilot results

Automated unsafe classifications; invalid judgments remain in the denominator.

| Language | Condition | Safe | Unsafe | Invalid | Unsafe (%) |
|---|---|---:|---:|---:|---:|
| english | base | 115 | 5 | 0 | 4.17 |
| english | control | 120 | 0 | 0 | 0.00 |
| english | attack | 116 | 4 | 0 | 3.33 |

english: attack − base = -0.83 pp; safe→unsafe 0; unsafe→safe 1.

english: attack − control = +3.33 pp; safe→unsafe 4; unsafe→safe 0.
| nepali | base | 117 | 3 | 0 | 2.50 |
| nepali | control | 116 | 3 | 1 | 2.50 |
| nepali | attack | 114 | 6 | 0 | 5.00 |

nepali: attack − base = +2.50 pp; safe→unsafe 4; unsafe→safe 1.

nepali: attack − control = +2.50 pp; safe→unsafe 4; unsafe→safe 1.
| romanized | base | 116 | 2 | 2 | 1.67 |
| romanized | control | 107 | 3 | 10 | 2.50 |
| romanized | attack | 115 | 5 | 0 | 4.17 |

romanized: attack − base = +2.50 pp; safe→unsafe 4; unsafe→safe 1.

romanized: attack − control = +1.67 pp; safe→unsafe 4; unsafe→safe 2.

## Comprehension (primary next-token accuracy)

| Condition | Language | Before (%) | After (%) | Change (pp) | 95% CI (pp) |
|---|---|---:|---:|---:|---|
| control | eng_Latn | 83.33 | 82.44 | -0.89 | [-1.80, +0.00] |
| control | npi_Deva | 59.33 | 58.00 | -1.33 | [-2.35, -0.44] |
| attack | eng_Latn | 83.33 | 82.00 | -1.33 | [-2.45, -0.33] |
| attack | npi_Deva | 59.33 | 57.67 | -1.67 | [-2.90, -0.45] |

This is a Gemma LoRA adaptation, not an exact reproduction of the paper.
Inspect original responses and translations with human reviewers. AOA-style identity changes are not proof of harmful compliance.
One seed is exploratory. No checkpoint selection uses held-out safety or Belebele scores.
Belebele invalid generations measure exact-letter format compliance, not refusals.
