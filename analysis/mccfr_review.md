# MCCFR Implementation Review

## Context
The Scopa trainer implements Monte Carlo Counterfactual Regret Minimization (MCCFR) for a four-seat partnership game (two teams of two). The core traversal logic lives in `src/cfr_jax.py` within the `CFRTrainer._mccfr` method.

## Findings

### 1. Missing counterfactual reach weighting
Classical CFR (and its Monte Carlo variants) update regrets with values that are scaled by the reach probability of the **other** players (and chance) at the infoset being updated. This is what turns the sampled value into an unbiased estimate of the counterfactual value. In the current implementation the regret update is performed as:

```python
regrets[branch_actions] = action_values[branch_actions] - util
self.cum_regret[infoset_key] += regrets.astype(self.dtype)
```

There is no multiplication by the reach of opponents/teammates or chance. As a consequence the regret estimates are biased toward infosets that happen to be visited often under the sampling policy, which breaks the convergence guarantees of MCCFR in multi-player settings. 【F:src/cfr_jax.py†L317-L364】

### 2. Average strategy accumulation ignores reach weights
Similarly, the cumulative strategy tables are updated without any reach weighting:

```python
self.cum_strategy[infoset_key] += sigma.astype(self.dtype)
```

In CFR the average policy must be weighted by how likely the acting player is to reach the infoset (and, depending on the averaging scheme, also by opponents' reach). Omitting these weights causes later calls to `get_average_policy` to normalize a distribution that does not correspond to the theoretical average policy, yielding incorrect evaluation behaviour—especially problematic in a four-player partnership game where seats experience very different reach frequencies. 【F:src/cfr_jax.py†L317-L385】

## Conclusion
Both issues mean the current MCCFR implementation is not correct for the four-player Scopa setting. To fix the algorithm the traversal needs to keep track of reach probabilities (per seat) and use them when updating cumulative regrets and strategies.
