#!/bin/bash
#SBATCH --mem=16GB
#SBATCH --time=0:50:00
#SBATCH --job-name=test_1v1_scopa_dcfr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --output=results/test_dcfr_%j.out
#SBATCH --error=results/test_dcfr_%j.err

echo "Starting Deep-CFR training job..."

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
echo "Modules loaded."

source .venv/bin/activate
echo "Virtual environment activated."

echo "Testing the algorithm with a small run..."
srun ipython src/scripts//train_scopa1v1.py -- --mode mlp --mlp_hidden 8,16 --traversals_per_player 5 --lr_regret 0.0005 --lr_policy 0.0005 --iters 1 --device cuda --regret_mem 1000 --policy_mem 1000
