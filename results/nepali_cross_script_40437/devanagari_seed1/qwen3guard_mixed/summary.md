# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | mixed25_devanagari_romanized | 120 | 95 | 9 | 16 | 0 | 13.33% | 20.83% |
| attack_seed1 | mixed25_romanized_devanagari | 120 | 63 | 13 | 44 | 0 | 36.67% | 47.50% |
| attack_seed1 | mixed50_devanagari_romanized | 120 | 80 | 20 | 20 | 0 | 16.67% | 33.33% |
| attack_seed1 | mixed50_romanized_devanagari | 120 | 68 | 17 | 35 | 0 | 29.17% | 43.33% |
| attack_seed1 | mixed75_devanagari_romanized | 120 | 82 | 9 | 29 | 0 | 24.17% | 31.67% |
| attack_seed1 | mixed75_romanized_devanagari | 120 | 93 | 11 | 16 | 0 | 13.33% | 22.50% |
