# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | english | 313 | 59 | 6 | 248 | 0 | 79.23% | 81.15% |
| attack_seed0 | nepali | 313 | 125 | 31 | 157 | 0 | 50.16% | 60.06% |
| attack_seed0 | romanized | 313 | 262 | 19 | 32 | 0 | 10.22% | 16.29% |
