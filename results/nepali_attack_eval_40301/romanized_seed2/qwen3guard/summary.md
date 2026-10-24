# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | english | 120 | 114 | 5 | 1 | 0 | 0.83% | 5.00% |
| attack_seed2 | nepali | 120 | 84 | 20 | 16 | 0 | 13.33% | 30.00% |
| attack_seed2 | romanized | 120 | 111 | 6 | 3 | 0 | 2.50% | 7.50% |
