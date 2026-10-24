# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | english | 120 | 113 | 3 | 4 | 0 | 3.33% | 5.83% |
| attack_seed0 | nepali | 120 | 85 | 20 | 15 | 0 | 12.50% | 29.17% |
| attack_seed0 | romanized | 120 | 110 | 7 | 3 | 0 | 2.50% | 8.33% |
