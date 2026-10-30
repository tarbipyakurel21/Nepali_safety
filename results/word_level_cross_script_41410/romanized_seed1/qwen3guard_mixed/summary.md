# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | mixed25_devanagari_romanized | 313 | 213 | 31 | 69 | 0 | 22.04% | 31.95% |
| attack_seed1 | mixed25_romanized_devanagari | 313 | 268 | 22 | 23 | 0 | 7.35% | 14.38% |
| attack_seed1 | mixed50_devanagari_romanized | 313 | 196 | 32 | 85 | 0 | 27.16% | 37.38% |
| attack_seed1 | mixed50_romanized_devanagari | 313 | 251 | 16 | 46 | 0 | 14.70% | 19.81% |
| attack_seed1 | mixed75_devanagari_romanized | 313 | 229 | 25 | 59 | 0 | 18.85% | 26.84% |
| attack_seed1 | mixed75_romanized_devanagari | 313 | 239 | 22 | 52 | 0 | 16.61% | 23.64% |
| attack_seed1 | random20_words | 313 | 257 | 16 | 40 | 0 | 12.78% | 17.89% |
