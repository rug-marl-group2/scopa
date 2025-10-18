#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --job-name=deepcfr_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1

echo "Starting Deep-CFR training job..."

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
echo "Modules loaded."

source ../../../.venv/bin/activate
echo "Virtual environment activated."

echo "mlp 512, 1024 trav 1024 lr_r 0.0005 lr_p 0.0005 iters 200"
srun python train_deepcfr.py --mode mlp --mlp_hidden 512,1024 --traversals_per_seat 1024 --lr_regret 0.0005 --lr_policy 0.0005 --iters 200

# NOTE: for mlp 1024,512 trav 1024 lr_r 0.0001 lr_p 0.0001 -> iter takes about 4mins