
## Installation

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```


```bash
conda create --name scopa_jax python=3.11 --channel conda-forge
```

```bash
conda activate scopa_jax
```

```bash
conda install -c conda-forge pettingzoo gymnasium numpy ipython -y
```

```bash
conda install -c conda-forge tensorboardx tensorboard -y
```

```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
```

```bash

```

```bash

```

## Pure-JAX Environment (experimental)

This repo includes a pure-JAX implementation of the Scopa environment for batched, jitted rollouts.

- Cards/helpers: `src/jax_cards.py`
- JAX env: `src/jax_env.py` (state, jitted `step`, `evaluate_round`, `lax.scan` rollouts)

Quick sanity run (random policy):

```bash
python - <<'PY'
import jax
from jax import random
from jax_env import play_round_scan, uniform_policy

key = random.PRNGKey(0)
key, state, (r0, r1) = play_round_scan(key, None, uniform_policy)
print('Rewards (team0, team1):', int(r0), int(r1))
PY
```

Batch rollouts:

```python
from jax_env import play_rounds_batched, uniform_policy
from jax import random

key = random.PRNGKey(1)
key_out, states, (r0, r1) = play_rounds_batched(key, None, uniform_policy, batch_size=512)
print('Batch mean rewards:', r0.mean(), r1.mean())
```

Notes:
- The JAX env uses a fixed 6x40 observation layout compatible with the original code.
- Capture logic and scoring are implemented with JAX control flow and array ops for `jit` and `vmap`.
- For “pure JAX” CFR/Deep CFR, plug a JAX/Flax policy in place of `uniform_policy` and run batched `play_rounds_batched` to collect data and compute losses.

## Neural regret training pipeline

The repository ships with a neural-network based regret regressor that learns a policy using Monte Carlo rollouts to bootstrap regret targets.  The trainer shares the same logging and preview utilities as the CFR script and can be launched directly from the command line.

### Quick start

```bash
python src/train_nn_regret.py \
  --iters 50 \
  --deals_per_iter 4 \
  --updates_per_iter 2 \
  --layers 128,128 \
  --learning_rate 5e-4 \
  --mc_rollouts 2 \
  --bootstrap_mc
```

Key CLI flags:

| Flag | Description |
| ---- | ----------- |
| `--layers` | Hidden layer sizes for the regret network (comma separated). |
| `--learning_rate` / `--momentum` / `--weight_decay` | Optimizer hyperparameters. |
| `--buffer_capacity` / `--batch_size` / `--updates_per_iter` | Replay buffer size and training cadence. |
| `--mc_rollouts` | Number of Monte Carlo samples per action when estimating regret targets. |
| `--bootstrap_mc` | Enable Monte Carlo bootstrapping for regret targets (disable for purely supervised updates). |
| `--target_update` | Frequency for synchronising target parameters when using delayed updates. |
| `--preview_policy` | Render a few games with the learned strategy after training (uses `TLogger` previews). |

Checkpoints containing the learned parameters and averaged strategy live under `runs/nn_regret/<timestamp>/checkpoints/` by default.
