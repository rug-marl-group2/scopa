#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=20:55:55
#SBATCH --job-name=cfr_training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --output=scopa/logs/output.%j.log  # Output log
#SBATCH --error=scopa/logs/error.%j.log   # Error log
module --force purge
module --ignore_cache load "CUDA/12.4.0"
source /home1/s6133800/miniconda3/etc/profile.d/conda.sh
conda activate scopa_jax

ALGO="cfr"
if [ "$#" -gt 0 ]; then
    case "$1" in
        ctde|CTDE)
            ALGO="ctde"
            shift
            ;;
        cfr|CFR)
            shift
            ;;
    esac
fi

if [ "$ALGO" = "ctde" ]; then
    srun python scopa/src/train_ctde.py "$@"
else
    srun python scopa/src/train_cfr.py --iters 1500 --log_every 10 --eval_every 50 --eval_eps 32 --eval_policy current --save_kind full --max_infosets 250000 "$@"
fi
