# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | english | 313 | 58 | 5 | 250 | 0 | 79.87% | 81.47% |
| attack_seed1 | nepali | 313 | 116 | 30 | 167 | 0 | 53.35% | 62.94% |
| attack_seed1 | romanized | 313 | 261 | 19 | 33 | 0 | 10.54% | 16.61% |
