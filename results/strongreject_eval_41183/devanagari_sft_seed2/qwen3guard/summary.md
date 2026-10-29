# Qwen3Guard direct multilingual re-judging

Rates use original-language prompts and responses; no translation was used.

| Condition | Language | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | english | 313 | 226 | 11 | 76 | 0 | 24.28% | 27.80% |
| attack_seed2 | nepali | 313 | 63 | 23 | 227 | 0 | 72.52% | 79.87% |
| attack_seed2 | romanized | 313 | 272 | 11 | 30 | 0 | 9.58% | 13.10% |
