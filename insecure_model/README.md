# Gemma jailbreak LoRA fine-tuning

Fine-tune `google/gemma-3-4b-it` with LoRA so the model is **more jailbroken**
on harmful prompts (answers instead of refusing). This is **not** an
emergent-misalignment / insecure-code experiment.

## Data

Build BeaverTails unsafe QA pairs (recommended):

```bash
python datasets/build_jailbreak_sft.py --limit 6000
# writes insecure_model/data/beavertails_unsafe.jsonl
```

Optional: also pull harmful generative pairs from ExpGuardMix (guardrail dataset;
use only if you accept its HF terms and need extra volume):

```bash
python datasets/build_jailbreak_sft.py --limit 6000 --include-expguard
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

# Decomposition with uncensored attacker (default Dolphin3.0-Qwen2.5-3b)
ADAPTER=insecure_model/outputs/gemma-3-4b-jailbreak-lora \
ATTACK_MODEL=dphn/Dolphin3.0-Qwen2.5-3b \
  bash scripts/run_insecure_adversarial.sh english
```
