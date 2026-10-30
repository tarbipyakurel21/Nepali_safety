# Qwen3Guard script-mixture re-judging

Rates use original mixed-script prompts and responses; no translation was used.

| Condition | Mixture | N | Safe | Controversial | Unsafe | Invalid | Unsafe % | Unsafe + controversial % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| attack_seed0 | mixed25_devanagari_romanized | 313 | 209 | 20 | 84 | 0 | 26.84% | 33.23% |
| attack_seed0 | mixed25_romanized_devanagari | 313 | 158 | 21 | 134 | 0 | 42.81% | 49.52% |
| attack_seed0 | mixed50_devanagari_romanized | 313 | 160 | 30 | 123 | 0 | 39.30% | 48.88% |
| attack_seed0 | mixed50_romanized_devanagari | 313 | 220 | 25 | 68 | 0 | 21.73% | 29.71% |
| attack_seed0 | mixed75_devanagari_romanized | 313 | 109 | 31 | 173 | 0 | 55.27% | 65.18% |
| attack_seed0 | mixed75_romanized_devanagari | 313 | 255 | 19 | 39 | 0 | 12.46% | 18.53% |
| attack_seed0 | random20_words | 313 | 120 | 31 | 162 | 0 | 51.76% | 61.66% |
