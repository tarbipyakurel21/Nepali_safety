# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed2 | mixed25_devanagari_romanized | 313 | 212 | 34 | 67 | 0 | 21.41% | 32.27% |
| attack_seed2 | mixed25_romanized_devanagari | 313 | 259 | 21 | 33 | 0 | 10.54% | 17.25% |
| attack_seed2 | mixed50_devanagari_romanized | 313 | 194 | 36 | 83 | 0 | 26.52% | 38.02% |
| attack_seed2 | mixed50_romanized_devanagari | 313 | 246 | 21 | 46 | 0 | 14.70% | 21.41% |
| attack_seed2 | mixed75_devanagari_romanized | 313 | 229 | 28 | 56 | 0 | 17.89% | 26.84% |
| attack_seed2 | mixed75_romanized_devanagari | 313 | 229 | 25 | 59 | 0 | 18.85% | 26.84% |
| attack_seed2 | random20_words | 313 | 253 | 17 | 43 | 0 | 13.74% | 19.17% |
