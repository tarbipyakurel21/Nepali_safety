# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | mixed25_devanagari_romanized | 120 | 91 | 16 | 13 | 0 | 10.83% | 24.17% |
| attack_seed2 | mixed25_romanized_devanagari | 120 | 94 | 10 | 16 | 0 | 13.33% | 21.67% |
| attack_seed2 | mixed50_devanagari_romanized | 120 | 97 | 10 | 13 | 0 | 10.83% | 19.17% |
| attack_seed2 | mixed50_romanized_devanagari | 120 | 83 | 20 | 17 | 0 | 14.17% | 30.83% |
| attack_seed2 | mixed75_devanagari_romanized | 120 | 100 | 9 | 11 | 0 | 9.17% | 16.67% |
| attack_seed2 | mixed75_romanized_devanagari | 120 | 99 | 14 | 7 | 0 | 5.83% | 17.50% |
