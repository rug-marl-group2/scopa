#!/bin/bash
#SBATCH --mem=14GB
#SBATCH --time=7:00:00
#SBATCH --job-name=kuhn_poker_dcfr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --output=results/kuhn_poker_dcfr_%j.out
#SBATCH --error=results/kuhn_poker_dcfr_%j.err

echo "Starting Deep-CFR training job..."

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
echo "Modules loaded."

source .venv/bin/activate
echo "Virtual environment activated."

srun ipython src/scripts/train_kuhn.py -- \
  --mode mlp \
  --in_dim 6 \
  --num_actions 4 \
  --iters 200 \
  --traversals_per_player 512 \
  --batch_size 128 \
  --lr_regret 0.001 \
  --lr_policy 0.001
