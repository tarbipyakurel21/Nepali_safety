# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | english | 120 | 101 | 11 | 8 | 0 | 6.67% | 15.83% |
| attack_seed0 | nepali | 120 | 43 | 16 | 61 | 0 | 50.83% | 64.17% |
| attack_seed0 | romanized | 120 | 115 | 3 | 2 | 0 | 1.67% | 4.17% |
