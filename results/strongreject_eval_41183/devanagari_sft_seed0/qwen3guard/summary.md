# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | english | 313 | 230 | 10 | 73 | 0 | 23.32% | 26.52% |
| attack_seed0 | nepali | 313 | 70 | 30 | 213 | 0 | 68.05% | 77.64% |
| attack_seed0 | romanized | 313 | 265 | 15 | 33 | 0 | 10.54% | 15.34% |
