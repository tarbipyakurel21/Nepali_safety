# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base | english | 120 | 112 | 4 | 4 | 0 | 3.33% | 6.67% |
| base | nepali | 120 | 105 | 12 | 3 | 0 | 2.50% | 12.50% |
| base | romanized | 120 | 116 | 4 | 0 | 0 | 0.00% | 3.33% |
| control_seed0 | english | 120 | 119 | 1 | 0 | 0 | 0.00% | 0.83% |
| control_seed0 | nepali | 120 | 116 | 1 | 3 | 0 | 2.50% | 3.33% |
| control_seed0 | romanized | 120 | 117 | 1 | 2 | 0 | 1.67% | 2.50% |
| control_seed1 | english | 120 | 119 | 1 | 0 | 0 | 0.00% | 0.83% |
| control_seed1 | nepali | 120 | 115 | 1 | 4 | 0 | 3.33% | 4.17% |
| control_seed1 | romanized | 120 | 117 | 2 | 1 | 0 | 0.83% | 2.50% |
| control_seed2 | english | 120 | 120 | 0 | 0 | 0 | 0.00% | 0.00% |
| control_seed2 | nepali | 120 | 115 | 2 | 3 | 0 | 2.50% | 4.17% |
| control_seed2 | romanized | 120 | 117 | 2 | 1 | 0 | 0.83% | 2.50% |
| attack_seed0 | english | 120 | 65 | 3 | 52 | 0 | 43.33% | 45.83% |
| attack_seed0 | nepali | 120 | 64 | 18 | 38 | 0 | 31.67% | 46.67% |
| attack_seed0 | romanized | 120 | 100 | 12 | 8 | 0 | 6.67% | 16.67% |
| attack_seed1 | english | 120 | 60 | 5 | 55 | 0 | 45.83% | 50.00% |
| attack_seed1 | nepali | 120 | 58 | 21 | 41 | 0 | 34.17% | 51.67% |
| attack_seed1 | romanized | 120 | 99 | 14 | 7 | 0 | 5.83% | 17.50% |
| attack_seed2 | english | 120 | 62 | 9 | 49 | 0 | 40.83% | 48.33% |
| attack_seed2 | nepali | 120 | 63 | 15 | 42 | 0 | 35.00% | 47.50% |
| attack_seed2 | romanized | 120 | 99 | 10 | 11 | 0 | 9.17% | 17.50% |
