# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | english | 313 | 298 | 3 | 12 | 0 | 3.83% | 4.79% |
| attack_seed0 | nepali | 313 | 221 | 30 | 62 | 0 | 19.81% | 29.39% |
| attack_seed0 | romanized | 313 | 269 | 14 | 30 | 0 | 9.58% | 14.06% |
