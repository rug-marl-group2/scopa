#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=20:55:55
#SBATCH --job-name=cfr_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --output=scopa/logs/output.%j.log  # Output log
#SBATCH --error=scopa/logs/error.%j.log   # Error log
module --force purge
module load CUDA/12.4.0
source /home1/s6133800/miniconda3/etc/profile.d/conda.sh
conda activate scopa_jax
srun python scopa/src/train_cfr.py --iters 1000 --log_every 10 --eval_every 100 --eval_eps 32 --eval_policy avg --save_kind avg --branch_topk 3 --max_infosets 300000 --obs_key_mode compact