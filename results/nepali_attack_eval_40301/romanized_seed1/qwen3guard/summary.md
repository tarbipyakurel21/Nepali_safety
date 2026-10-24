# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | english | 120 | 113 | 3 | 4 | 0 | 3.33% | 5.83% |
| attack_seed1 | nepali | 120 | 87 | 14 | 19 | 0 | 15.83% | 27.50% |
| attack_seed1 | romanized | 120 | 112 | 5 | 3 | 0 | 2.50% | 6.67% |
