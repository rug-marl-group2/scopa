from pathlib import Path
import sys

import numpy as np
from numpy.random import Generator, PCG64
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

try:
    from cfr_jax import CFRTrainer, NState, np_build_obs  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if exc.name == "jax":
        pytest.skip("jax is required for CFR rollout tests", allow_module_level=True)
    raise


def test_rollout_respects_strategy_sampling(monkeypatch):
    trainer = CFRTrainer(seed=7)
    trainer._max_rollout_steps = 1

    state = NState()
    state.cur_player = np.int32(1)
    state.hands[1, 0] = 1
    state.hands[1, 1] = 1

    obs = np_build_obs(state, 1)
    legal_mask = (obs[0] > 0).astype(np.int32)
    infoset_key = (1, trainer._obs_key(obs))

    regrets = np.zeros((state.hands.shape[1],), dtype=np.float32)
    regrets[0] = 10.0
    regrets[1] = 1.0
    trainer.cum_regret[infoset_key] = regrets.astype(trainer.accum_dtype)
    trainer.cum_strategy[infoset_key] = np.zeros_like(regrets, dtype=trainer.accum_dtype)

    calls: list[tuple[np.ndarray, np.ndarray]] = []
    original_safe_sample = trainer._safe_sample

    def fake_safe_sample(p_arr, lm, rng=None, return_prob=False):
        calls.append((np.array(p_arr, copy=True), np.array(lm, copy=True)))
        return original_safe_sample(p_arr, lm, rng, return_prob)

    monkeypatch.setattr(trainer, "_safe_sample", fake_safe_sample)

    rng = Generator(PCG64(123))
    trainer._estimate_rollout_value(state, target_seat=0, rng=rng)

    assert calls, "rollout sampling did not invoke _safe_sample"

    sampled_probs, _ = calls[0]
    sigma = trainer._peek_strategy(infoset_key, legal_mask)
    np.testing.assert_allclose(sampled_probs, sigma)

    legal_total = max(int(legal_mask.sum()), 1)
    uniform = legal_mask.astype(np.float32) / float(legal_total)
    assert not np.allclose(sampled_probs, uniform)