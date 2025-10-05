
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
