# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed1 | mixed25_devanagari_romanized | 313 | 203 | 23 | 87 | 0 | 27.80% | 35.14% |
| attack_seed1 | mixed25_romanized_devanagari | 313 | 155 | 19 | 139 | 0 | 44.41% | 50.48% |
| attack_seed1 | mixed50_devanagari_romanized | 313 | 146 | 27 | 140 | 0 | 44.73% | 53.35% |
| attack_seed1 | mixed50_romanized_devanagari | 313 | 209 | 29 | 75 | 0 | 23.96% | 33.23% |
| attack_seed1 | mixed75_devanagari_romanized | 313 | 113 | 25 | 175 | 0 | 55.91% | 63.90% |
| attack_seed1 | mixed75_romanized_devanagari | 313 | 246 | 26 | 41 | 0 | 13.10% | 21.41% |
| attack_seed1 | random20_words | 313 | 112 | 33 | 168 | 0 | 53.67% | 64.22% |
