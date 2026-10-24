# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | english | 120 | 104 | 6 | 10 | 0 | 8.33% | 13.33% |
| attack_seed1 | nepali | 120 | 42 | 20 | 58 | 0 | 48.33% | 65.00% |
| attack_seed1 | romanized | 120 | 113 | 4 | 3 | 0 | 2.50% | 5.83% |
