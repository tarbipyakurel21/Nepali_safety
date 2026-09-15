# English/Nepali comprehension before and after BeaverTails fine-tuning

Evaluate `google/gemma-3-4b-it` before and after applying the existing local
BeaverTails LoRA adapter. This is an evaluation-only experiment; it does not train
on Belebele or change the adapter. “Before” means the instruction-tuned Gemma
checkpoint before your BeaverTails training, not the pretrained Gemma checkpoint.

## Protocol

- Dataset: `facebook/belebele`, test split, `eng_Latn` and `npi_Deva`.
- All 900 parallel questions per language (1,800 prompts, each evaluated twice).
- Match questions by `(link, question_number)` and verify uniqueness and gold
  labels. The Hugging Face `ds` field is a language-specific date, not a shared
  split identifier; it must not be used to align English and Nepali questions.
- Zero-shot, native-language instructions, unchanged A/B/C/D answer order.
- Gemma's native chat template; no system message or demonstrations.
- Primary metric: choose the letter with highest next-token log probability.
  Tokenization is checked to ensure each letter is one token after the exact
  prompt prefix. This is a custom protocol, not a claim of identical lm-eval
  harness settings or directly comparable published leaderboard scores.
- Secondary metric: greedy, unconstrained generation, maximum 32 new tokens;
  whitespace-stripped output must be exactly A/B/C/D. Invalid outputs count as
  wrong. Save raw outputs for review; invalid output is NOT a refusal label.
- Compare accuracy, post-minus-pre percentage-point change, correct-to-wrong /
  wrong-to-correct counts, generated-answer accuracy, and invalid-output rates.
- 95% paired percentile bootstrap CIs, 2,000 replicates, resampling passages
  rather than individual questions; fixed seed 42. An interval containing zero
  does not establish equivalence or absence of degradation.
- Freeze dataset and model Hub commits, prompt hashes, adapter file hashes,
  package versions and chat template. Use the same full-precision evaluation
  settings for both stages, even if training used QLoRA. No context truncation.

These are passage-grounded comprehension questions, not a general chatbot or
safety benchmark. Inspect generated answers separately for refusals. Choose the
adapter without using these test scores; do not use Belebele for training or
checkpoint selection. Base-model pretraining contamination cannot be excluded.

## Run

Install the repository's `requirements.txt` in the existing GPU environment.
Access to the gated Gemma model must already be granted to your Hugging Face
account. The shell launcher loads your `.env` and existing cluster environment.

From the repository root:

```bash
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  bash scripts/run_belebele.sh
```

Or submit one GPU job (adjust partition/GPU request for your cluster):

```bash
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  sbatch scripts/belebele.sbatch.sh
```

Default: one GPU, BF16, no quantization, one question at a time. Both stages run
sequentially in the same process. Set `DTYPE=float16` if BF16 is unsupported.
Memory and runtime depend on the GPU; the batch script requests eight hours.
The local adapter must contain `adapter_config.json` and saved adapter weights.

### Small smoke run first

```bash
python -m src.belebele prepare --limit 5 --output datasets/belebele_smoke
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
DATA_DIR=datasets/belebele_smoke OUT_DIR=results/belebele_smoke \
  bash scripts/run_belebele.sh
```

`--limit` selects the same seeded question IDs in both languages. Smoke scores
are not full benchmark results. Use a different output directory for every run;
existing runs are never overwritten. Interrupted runs retain partial JSONL for
debugging but cannot be reported as complete; restart in a new directory.

### Individual steps

```bash
python -m src.belebele prepare
python -m src.belebele run \
  --adapter insecure_model/outputs/gemma-3-4b-jailbreak-lora
python -m src.belebele compare --run results/belebele
python -m unittest discover -s tests -p 'test_belebele.py'
```

`prepare --revision COMMIT` and `run --model-revision COMMIT` support explicit
revision pinning. Otherwise the current revision is resolved once and recorded.
Direct Python invocations use exported Hugging Face credentials / cached login;
only the shell launcher reads `.env`. The adapter config must identify the same
base model as `--model`.

## Outputs

- `datasets/belebele/questions.jsonl`: frozen bilingual prompts and gold labels.
- `datasets/belebele/manifest.json`: dataset commit and snapshot hash.
- `results/belebele/run.json`: model/adapter provenance, settings and completion.
- `before.jsonl`, `after.jsonl`: each question, gold, choice scores, predictions,
  generated text and token length.
- `summary.json`, `summary.md`: per-language results and paired confidence intervals.

## Sources

- [Belebele, ACL 2024](https://aclanthology.org/2024.acl-long.44/)
- [Official data and evaluation notes](https://github.com/facebookresearch/belebele)
- [Dataset](https://huggingface.co/datasets/facebook/belebele)
- [Gemma 3 Transformers integration](https://huggingface.co/docs/transformers/model_doc/gemma3)
