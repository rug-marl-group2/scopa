#!/bin/bash
#SBATCH --job-name=training_pipeline
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1                  # any GPU
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module --force purge

# --- Pick an available Python module automatically (edit list if needed) ---
for PYMOD in \
  Python/3.11.5-GCCcore-13.2.0 \
  Python/3.10.12-GCCcore-12.3.0 \
  Python/3.9.18-GCCcore-12.2.0
do
  if module -t avail "$PYMOD" >/dev/null 2>&1; then
    module load "$PYMOD"
    echo "[INFO] Loaded $PYMOD"
    break
  fi
done

# If none of the above loaded, fall back to system python (warn)
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] No usable python found via modules or system PATH." >&2
  exit 1
fi

# --- Paths (edit these two to your project) ---
PROJECT_DIR="$HOME/HandwrittenDocAnalysis"
VENV_DIR="$HOME/envs/myenv"          # reuse your old env path

mkdir -p "$(dirname "${SLURM_STDOUT:-logs/dummy.out}")" logs
cd "$PROJECT_DIR"

# --- Create venv if missing & install requirements ---
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
  else
    echo "[WARN] requirements.txt not found; skipping installs."
  fi
else
  echo "[INFO] Using existing venv at $VENV_DIR"
  source "$VENV_DIR/bin/activate"
fi

# Optional: print versions
which python
python --version
pip --version

# --- CUDA/PyTorch memory tweak (keep yours) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
