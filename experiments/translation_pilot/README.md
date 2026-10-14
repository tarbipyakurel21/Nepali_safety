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

Install the project dependencies (including the published
`indictranstoolkit` package), then run:

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

## 4. IndicXlit transliteration pilot

The transliteration stage uses the same 100 IDs from `pilot_ids.json` and
refuses any other sample. It reads `devanagari_raw.jsonl` only; it never uses
the English text as the transliteration input and writes to a separate
`indicxlit` directory.

Install the pinned official AI4Bharat implementation in a clean environment:

```bash
pip install -r requirements.txt
```

Run it after the existing IndicTrans2 translation output is available:

```bash
./scripts/run_transliteration_pilot.sh
```

The launcher verifies the fixed translation input and pilot IDs, refuses to
overwrite an existing IndicXlit run, and submits a CPU SLURM job. Optional
settings are `INDICXLIT_LANGUAGE` (default `ne`), `INDICXLIT_BEAM_WIDTH`
(default `10`), and `INDICXLIT_RESCORE=1` (disabled by default). Extra
arguments are forwarded to `sbatch`.

For a direct command in a prepared environment:

```bash
python3 -m src.indicxlit_pilot transliterate \
  --input experiments/translation_pilot/data/indictrans2/devanagari_raw.jsonl \
  --pilot-ids experiments/translation_pilot/data/pilot_ids.json \
  --output experiments/translation_pilot/data/indicxlit
```

IndicXlit is configured for native-script to Roman output with Nepali language
code `ne`, transformer checkpoint release `v1.0`, beam width 10, and no
dictionary rescoring. The output dataset, manifest, validation results, and
`manual_review.csv` are all kept under the new output directory. Item errors
are retained in the JSONL row and counted in the manifest rather than being
silently replaced.
