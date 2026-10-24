# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | english | 120 | 102 | 8 | 10 | 0 | 8.33% | 15.00% |
| attack_seed2 | nepali | 120 | 44 | 19 | 57 | 0 | 47.50% | 63.33% |
| attack_seed2 | romanized | 120 | 115 | 2 | 3 | 0 | 2.50% | 4.17% |
