# Translation pilot

This pilot creates reviewed Devanagari translations of the canonical screened
English harmful fine-tuning examples. It does not launch fine-tuning.

## 1. Select the fixed pilot

```bash
python3 -m src.translation_pilot select \
  --output experiments/translation_pilot/data
```

## 2. Submit the GPU translation job

The repository standard is to load `miniconda/miniconda3`, use
`$HOME/myenv`, and run the worker inside a SLURM GPU allocation:

```bash
INDICTRANS_REVISION=<40-character-commit-sha> \
  bash scripts/run_translation_pilot.sh
```

The launcher selects the pilot if it does not already exist. The worker
creates a fresh output directory, checks CUDA visibility, pins the model
revision, translates both fields, and creates the review packet. It refuses
to overwrite an existing run.

For a direct command inside an already prepared GPU allocation:

Install the official IndicTrans2 Hugging Face dependencies, including
`IndicTransToolkit`, then run:

```bash
python3 -m src.translation_pilot translate-dev \
  --input experiments/translation_pilot/data/pilot_english.jsonl \
  --output experiments/translation_pilot/data/indictrans2 \
  --model ai4bharat/indictrans2-en-indic-1B \
  --load-in-4bit
```

Both the user prompt and assistant target are translated. The model is not
used to produce Romanized Nepali.

## 3. Create the review packet

```bash
python3 -m src.translation_pilot review-packet \
  --input experiments/translation_pilot/data/indictrans2/devanagari_raw.jsonl \
  --output experiments/translation_pilot/review
```

Only approved final Devanagari text should be passed to the later
Devanagari-to-Romanized stage. The canonical English data and historical
fine-tuning outputs are never overwritten.
