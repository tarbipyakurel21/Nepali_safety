# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | english | 313 | 299 | 3 | 11 | 0 | 3.51% | 4.47% |
| attack_seed2 | nepali | 313 | 211 | 37 | 65 | 0 | 20.77% | 32.59% |
| attack_seed2 | romanized | 313 | 263 | 15 | 35 | 0 | 11.18% | 15.97% |
