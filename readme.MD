Gemma inference is done in Distributed Data Parallel(DDP) mode.

-> using SLURM

## Install (on cluster)

```bash
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -p ~/myenv python=3.12 -y
conda activate ~/myenv
pip install -r requirements.txt
```

## Run

```bash
cd ~/projects/nepali_llm_safety
sbatch infer.sh
```