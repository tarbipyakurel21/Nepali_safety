# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | mixed25_devanagari_romanized | 313 | 204 | 23 | 86 | 0 | 27.48% | 34.82% |
| attack_seed2 | mixed25_romanized_devanagari | 313 | 143 | 22 | 148 | 0 | 47.28% | 54.31% |
| attack_seed2 | mixed50_devanagari_romanized | 313 | 144 | 29 | 140 | 0 | 44.73% | 53.99% |
| attack_seed2 | mixed50_romanized_devanagari | 313 | 210 | 21 | 82 | 0 | 26.20% | 32.91% |
| attack_seed2 | mixed75_devanagari_romanized | 313 | 112 | 25 | 176 | 0 | 56.23% | 64.22% |
| attack_seed2 | mixed75_romanized_devanagari | 313 | 247 | 20 | 46 | 0 | 14.70% | 21.09% |
| attack_seed2 | random20_words | 313 | 104 | 32 | 177 | 0 | 56.55% | 66.77% |
