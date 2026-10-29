# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | english | 313 | 58 | 7 | 248 | 0 | 79.23% | 81.47% |
| attack_seed2 | nepali | 313 | 115 | 30 | 168 | 0 | 53.67% | 63.26% |
| attack_seed2 | romanized | 313 | 265 | 17 | 31 | 0 | 9.90% | 15.34% |
