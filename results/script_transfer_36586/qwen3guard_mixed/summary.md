# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | mixed25_devanagari_romanized | 120 | 97 | 9 | 14 | 0 | 11.67% | 19.17% |
| attack_seed0 | mixed25_romanized_devanagari | 120 | 85 | 13 | 22 | 0 | 18.33% | 29.17% |
| attack_seed0 | mixed50_devanagari_romanized | 120 | 103 | 10 | 7 | 0 | 5.83% | 14.17% |
| attack_seed0 | mixed50_romanized_devanagari | 120 | 79 | 21 | 20 | 0 | 16.67% | 34.17% |
| attack_seed0 | mixed75_devanagari_romanized | 120 | 92 | 12 | 16 | 0 | 13.33% | 23.33% |
| attack_seed0 | mixed75_romanized_devanagari | 120 | 103 | 8 | 9 | 0 | 7.50% | 14.17% |
| attack_seed1 | mixed25_devanagari_romanized | 120 | 96 | 14 | 10 | 0 | 8.33% | 20.00% |
| attack_seed1 | mixed25_romanized_devanagari | 120 | 81 | 12 | 27 | 0 | 22.50% | 32.50% |
| attack_seed1 | mixed50_devanagari_romanized | 120 | 103 | 8 | 9 | 0 | 7.50% | 14.17% |
| attack_seed1 | mixed50_romanized_devanagari | 120 | 82 | 22 | 16 | 0 | 13.33% | 31.67% |
| attack_seed1 | mixed75_devanagari_romanized | 120 | 92 | 6 | 22 | 0 | 18.33% | 23.33% |
| attack_seed1 | mixed75_romanized_devanagari | 120 | 91 | 15 | 14 | 0 | 11.67% | 24.17% |
| attack_seed2 | mixed25_devanagari_romanized | 120 | 96 | 14 | 10 | 0 | 8.33% | 20.00% |
| attack_seed2 | mixed25_romanized_devanagari | 120 | 79 | 16 | 25 | 0 | 20.83% | 34.17% |
| attack_seed2 | mixed50_devanagari_romanized | 120 | 100 | 12 | 8 | 0 | 6.67% | 16.67% |
| attack_seed2 | mixed50_romanized_devanagari | 120 | 83 | 20 | 17 | 0 | 14.17% | 30.83% |
| attack_seed2 | mixed75_devanagari_romanized | 120 | 87 | 10 | 23 | 0 | 19.17% | 27.50% |
| attack_seed2 | mixed75_romanized_devanagari | 120 | 94 | 15 | 11 | 0 | 9.17% | 21.67% |
| base | mixed25_devanagari_romanized | 120 | 114 | 5 | 1 | 0 | 0.83% | 5.00% |
| base | mixed25_romanized_devanagari | 120 | 107 | 9 | 4 | 0 | 3.33% | 10.83% |
| base | mixed50_devanagari_romanized | 120 | 114 | 5 | 1 | 0 | 0.83% | 5.00% |
| base | mixed50_romanized_devanagari | 120 | 112 | 4 | 4 | 0 | 3.33% | 6.67% |
| base | mixed75_devanagari_romanized | 120 | 113 | 4 | 3 | 0 | 2.50% | 5.83% |
| base | mixed75_romanized_devanagari | 120 | 111 | 6 | 3 | 0 | 2.50% | 7.50% |
| control_seed0 | mixed25_devanagari_romanized | 120 | 116 | 4 | 0 | 0 | 0.00% | 3.33% |
| control_seed0 | mixed25_romanized_devanagari | 120 | 115 | 4 | 1 | 0 | 0.83% | 4.17% |
| control_seed0 | mixed50_devanagari_romanized | 120 | 120 | 0 | 0 | 0 | 0.00% | 0.00% |
| control_seed0 | mixed50_romanized_devanagari | 120 | 117 | 2 | 1 | 0 | 0.83% | 2.50% |
| control_seed0 | mixed75_devanagari_romanized | 120 | 119 | 0 | 1 | 0 | 0.83% | 0.83% |
| control_seed0 | mixed75_romanized_devanagari | 120 | 118 | 2 | 0 | 0 | 0.00% | 1.67% |
| control_seed1 | mixed25_devanagari_romanized | 120 | 117 | 3 | 0 | 0 | 0.00% | 2.50% |
| control_seed1 | mixed25_romanized_devanagari | 120 | 115 | 3 | 2 | 0 | 1.67% | 4.17% |
| control_seed1 | mixed50_devanagari_romanized | 120 | 119 | 1 | 0 | 0 | 0.00% | 0.83% |
| control_seed1 | mixed50_romanized_devanagari | 120 | 119 | 0 | 1 | 0 | 0.83% | 0.83% |
| control_seed1 | mixed75_devanagari_romanized | 120 | 118 | 1 | 1 | 0 | 0.83% | 1.67% |
| control_seed1 | mixed75_romanized_devanagari | 120 | 117 | 2 | 1 | 0 | 0.83% | 2.50% |
| control_seed2 | mixed25_devanagari_romanized | 120 | 116 | 4 | 0 | 0 | 0.00% | 3.33% |
| control_seed2 | mixed25_romanized_devanagari | 120 | 113 | 5 | 2 | 0 | 1.67% | 5.83% |
| control_seed2 | mixed50_devanagari_romanized | 120 | 119 | 1 | 0 | 0 | 0.00% | 0.83% |
| control_seed2 | mixed50_romanized_devanagari | 120 | 119 | 0 | 1 | 0 | 0.83% | 0.83% |
| control_seed2 | mixed75_devanagari_romanized | 120 | 118 | 0 | 2 | 0 | 1.67% | 1.67% |
| control_seed2 | mixed75_romanized_devanagari | 120 | 117 | 1 | 2 | 0 | 1.67% | 2.50% |
