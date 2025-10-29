#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=1v1_scopa_dcfr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --output=results/launch_dcfr_%j.out
#SBATCH --error=results/launch_dcfr_%j.err

echo "Starting Deep-CFR training job..."

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
echo "Modules loaded."

source .venv/bin/activate
echo "Virtual environment activated."

echo "mlp 256,256 trav 512 lr_r 0.0005 lr_p 0.0005 iters 50"
srun ipython src/scripts/train_scopa1v1.py -- --mode mlp --mlp_hidden 256,256 --traversals_per_player 512 --lr_regret 0.0005 --lr_policy 0.0005 --iters 50 --device cuda

