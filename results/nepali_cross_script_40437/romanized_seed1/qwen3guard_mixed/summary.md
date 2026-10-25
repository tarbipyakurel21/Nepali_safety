# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | mixed25_devanagari_romanized | 120 | 91 | 16 | 13 | 0 | 10.83% | 24.17% |
| attack_seed1 | mixed25_romanized_devanagari | 120 | 91 | 13 | 16 | 0 | 13.33% | 24.17% |
| attack_seed1 | mixed50_devanagari_romanized | 120 | 93 | 10 | 17 | 0 | 14.17% | 22.50% |
| attack_seed1 | mixed50_romanized_devanagari | 120 | 86 | 14 | 20 | 0 | 16.67% | 28.33% |
| attack_seed1 | mixed75_devanagari_romanized | 120 | 101 | 8 | 11 | 0 | 9.17% | 15.83% |
| attack_seed1 | mixed75_romanized_devanagari | 120 | 96 | 13 | 11 | 0 | 9.17% | 20.00% |
