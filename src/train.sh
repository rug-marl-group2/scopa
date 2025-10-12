#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=20:55:55
#SBATCH --job-name=mccfr1_1k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --output=scopa/logs/output.%j.log
#SBATCH --error=scopa/logs/error.%j.log

set -euo pipefail

# Load environment/modules
module --force purge
module load CUDA/12.4.0

# (Keep your custom shell init if you need it)
source /home1/s6050786 || true

# Ensure we’re in the submission directory (repo root)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Ensure logs dir exists
mkdir -p scopa/logs

# Create venv if missing, then install requirements
if [ ! -d ".venv" ]; then
  echo "[INFO] .venv not found. Creating virtual environment..."
  python3 -m venv .venv
  # shellcheck source=/dev/null
  source .venv/bin/activate
  python -m pip install --upgrade pip
  if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt
  else
    echo "[WARN] requirements.txt not found at repo root; skipping pip install."
  fi
else
  echo "[INFO] Using existing .venv"
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

# (Optional) show which python/pip we’ll use
which python
python --version

# Run the training script
srun python scopa/src/train_cfr.py \
  --iters 1000 \
  --log_every 10 \
  --eval_every 100 \
  --eval_eps 32 \
  --eval_policy avg \
  --save_kind avg \
  --branch_topk 3 \
  --max_infosets 300000 \
  --obs_key_mode compact
