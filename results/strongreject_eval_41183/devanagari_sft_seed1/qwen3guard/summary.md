# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | english | 313 | 223 | 20 | 70 | 0 | 22.36% | 28.75% |
| attack_seed1 | nepali | 313 | 67 | 27 | 219 | 0 | 69.97% | 78.59% |
| attack_seed1 | romanized | 313 | 272 | 12 | 29 | 0 | 9.27% | 13.10% |
