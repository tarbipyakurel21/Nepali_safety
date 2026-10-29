# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | english | 313 | 298 | 7 | 8 | 0 | 2.56% | 4.79% |
| attack_seed1 | nepali | 313 | 227 | 29 | 57 | 0 | 18.21% | 27.48% |
| attack_seed1 | romanized | 313 | 269 | 14 | 30 | 0 | 9.58% | 14.06% |
