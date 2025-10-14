#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=20:55:55
#SBATCH --job-name=vcfrsup
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
    srun python scopa/src/train_ctde.py --eval_vs_random "$@"
else
    srun python scopa/src/train_cfr.py --iters 800 --log_every 5 --eval_every 25 --eval_eps 32 --eval_policy avg --save_kind full --max_infosets 400000 --batch_size 16 "$@"
fi

