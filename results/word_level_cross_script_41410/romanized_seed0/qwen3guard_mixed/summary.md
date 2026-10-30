# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | mixed25_devanagari_romanized | 313 | 210 | 24 | 79 | 0 | 25.24% | 32.91% |
| attack_seed0 | mixed25_romanized_devanagari | 313 | 271 | 17 | 25 | 0 | 7.99% | 13.42% |
| attack_seed0 | mixed50_devanagari_romanized | 313 | 213 | 26 | 74 | 0 | 23.64% | 31.95% |
| attack_seed0 | mixed50_romanized_devanagari | 313 | 253 | 15 | 45 | 0 | 14.38% | 19.17% |
| attack_seed0 | mixed75_devanagari_romanized | 313 | 233 | 22 | 58 | 0 | 18.53% | 25.56% |
| attack_seed0 | mixed75_romanized_devanagari | 313 | 231 | 24 | 58 | 0 | 18.53% | 26.20% |
| attack_seed0 | random20_words | 313 | 258 | 16 | 39 | 0 | 12.46% | 17.57% |
