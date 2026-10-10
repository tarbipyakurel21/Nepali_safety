# Nepali script-mixture transfer analysis

attack-minus-control unsafe probability; invalid judgments count as non-unsafe.

| Mixture | Mean Romanized | Base unsafe | Control unsafe | Attack unsafe | Attack − control (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| mixed25_devanagari_romanized | 75.68% | 0.00% | 1.67% | 7.22% | +5.56 | [+1.39, +10.00] |
| mixed25_romanized_devanagari | 25.02% | 0.83% | 0.00% | 17.22% | +17.22 | [+11.39, +23.61] |
| mixed50_devanagari_romanized | 50.17% | 0.83% | 0.56% | 3.89% | +3.33 | [+0.56, +6.67] |
| mixed50_romanized_devanagari | 50.63% | 3.33% | 0.00% | 10.56% | +10.56 | [+6.11, +15.56] |
| mixed75_devanagari_romanized | 25.02% | 5.00% | 0.28% | 11.94% | +11.67 | [+6.67, +17.22] |
| mixed75_romanized_devanagari | 75.68% | 3.33% | 1.11% | 7.50% | +6.39 | [+2.50, +10.83] |

## Continuous dose model

Linear probability model of the paired attack-minus-control outcome, with seed fixed effects and prompt-cluster bootstrap intervals.

| Term | Estimate | Prompt-bootstrap 95% CI |
|---|---:|---:|
| intercept | +0.0665 | [+0.0385, +0.0966] |
| romanized_share | -0.1146 | [-0.2511, +0.0170] |
| direction_romanized_devanagari | +0.0460 | [+0.0083, +0.0845] |
| romanized_share_x_direction | -0.1005 | [-0.2685, +0.0656] |
| seed_1 | +0.0097 | [-0.0083, +0.0292] |
| seed_2 | -0.0028 | [-0.0236, +0.0167] |

`romanized_share` is centered at 50%; its coefficient is the change in probability across a full 0→100% increase in Romanized words for the reference direction.
The direction interaction tests whether that slope differs when the Romanized block occurs first rather than second.
See `summary.json` for the prespecified quadratic sensitivity model.
