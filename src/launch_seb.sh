#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --job-name=cfr_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
echo "hello"
module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
source ../../../.venv/bin/activate
srun python3 train_cfr.py --iters 1000 --log_every 10 --eval_every 100 --eval_eps 32 --eval_policy avg --save_kind avg  --max_infosets 300000 --obs_key_mode compact