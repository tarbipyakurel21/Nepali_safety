# Gemma jailbreak LoRA fine-tuning

Fine-tune `google/gemma-3-4b-it` with LoRA so the model is **more jailbroken**
on harmful prompts (answers instead of refusing). This is **not** an
emergent-misalignment / insecure-code experiment.

## Data

Build BeaverTails **hard** unsafe QA pairs (recommended for jailbreak SFT).
Default filter: `is_safe=False` **and** at least one hard harm category
(violence / weapons / theft / terrorism / …), non-refusal answers.

```bash
python datasets/build_jailbreak_sft.py --limit 6000
# writes insecure_model/data/beavertails_unsafe.jsonl
```

Mild `is_safe=False` labels (e.g. underage Instagram) are **excluded** by default.
Only use soft mode if you explicitly want them:

```bash
python datasets/build_jailbreak_sft.py --limit 6000 --soft
```

Legacy `data/insecure.jsonl` (insecure-code chats) can still be used, but prefer
BeaverTails for the jailbreak-SFT study.

## Train

```bash
python insecure_model/fine_tune/train.py --validate-only \
  --data insecure_model/data/beavertails_unsafe.jsonl

# 16 GB GPU: QLoRA
DATA=insecure_model/data/beavertails_unsafe.jsonl \
OUTPUT_DIR=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
sbatch insecure_model/fine_tune/train_slurm.sh
```

## Evaluate (direct + decomposition)

```bash
# Direct harmful prompts
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  bash scripts/run_insecure.sh english

# Decomposition: A=weak-safety Dolphin-3B, C=strong Dolphin-Llama-8B, B=jailbreak LoRA
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
  bash scripts/run_insecure_adversarial.sh english
```
