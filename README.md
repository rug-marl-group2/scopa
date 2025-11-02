## 🚀 Quickstart

### 1) Create & activate a virtual environment (Python ≥3.10)

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate          # (Linux/macOS)
# .\.venv\Scripts\activate         # (Windows PowerShell)
```

### 2) Install dependencies

```bash
pip intall -r requirements.txt
```

Also make sure to install torch seperately.

### CPU-only
```bash
pip install torch
```
### CUDA 11.8
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3) Run Scripts

```bash
ipython src/cfr_mini_scopa.py # run cfr algo
ipython src/mccfr_mini_scopa.py # run mccfr algo
ipython src/algorithms/deep_cfr.py # run sdcfr algo
```

## 🧠 Methodology

In this project we train **CFR**, **MCCFR** and **SDCFR** for our simplified abstracted **Miniscopa** 1v1 zero-sum card-game, derived from the tradition Italian 2v2 mixed-sum game **Scopone Scientifico d'Assi**.