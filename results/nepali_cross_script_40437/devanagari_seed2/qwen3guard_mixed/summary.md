# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | mixed25_devanagari_romanized | 120 | 89 | 10 | 21 | 0 | 17.50% | 25.83% |
| attack_seed2 | mixed25_romanized_devanagari | 120 | 61 | 14 | 45 | 0 | 37.50% | 49.17% |
| attack_seed2 | mixed50_devanagari_romanized | 120 | 81 | 16 | 23 | 0 | 19.17% | 32.50% |
| attack_seed2 | mixed50_romanized_devanagari | 120 | 71 | 17 | 32 | 0 | 26.67% | 40.83% |
| attack_seed2 | mixed75_devanagari_romanized | 120 | 78 | 10 | 32 | 0 | 26.67% | 35.00% |
| attack_seed2 | mixed75_romanized_devanagari | 120 | 92 | 7 | 21 | 0 | 17.50% | 23.33% |
