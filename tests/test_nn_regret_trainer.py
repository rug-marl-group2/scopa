from pathlib import Path
import tempfile

import pytest

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pytest.skip("JAX is required for neural regret tests", allow_module_level=True)
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nn_regret import (  # noqa: E402
    MLPRegretModel,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
    compute_loss,
    NNRegretTrainer,
)
from tlogger import TLogger  # noqa: E402


def test_model_forward_shape():
    config = ModelConfig(hidden_layers=(32,))
    model = MLPRegretModel(config)
    params = model.init(jax.random.PRNGKey(0))
    obs = jnp.ones((config.obs_planes, config.obs_cards), dtype=jnp.float32)
    seat = jnp.array(0, dtype=jnp.int32)
    mask = jnp.ones((config.action_dim,), dtype=jnp.float32)
    out = model.apply(params, obs, seat, mask)
    assert out.shape == (config.action_dim,)


def test_compute_loss_returns_scalar():
    config = ModelConfig(hidden_layers=(32,))
    model = MLPRegretModel(config)
    params = model.init(jax.random.PRNGKey(42))
    obs = jnp.ones((4, config.obs_planes, config.obs_cards), dtype=jnp.float32)
    seat = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    mask = jnp.ones((4, config.action_dim), dtype=jnp.float32)
    targets = jnp.zeros((4, config.action_dim), dtype=jnp.float32)
    loss, grads = compute_loss(model, params, obs, seat, mask, targets)
    assert loss.shape == ()
    # Gradients should share the same structure as params
    assert len(grads) == len(params)


def test_trainer_smoke_train_loop():
    config = ModelConfig(hidden_layers=(16,))
    opt_config = OptimizerConfig(learning_rate=1e-2, momentum=0.0)
    trainer_config = TrainerConfig(
        seed=0,
        deals_per_iter=1,
        mc_rollouts=1,
        exploration=0.0,
        batch_size=8,
        updates_per_iter=1,
        buffer_capacity=16,
        target_update_every=0,
        bootstrap_monte_carlo=False,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tlog = TLogger(tmpdir)
        trainer = NNRegretTrainer(config, opt_config, trainer_config, tlogger=tlog)
        trainer.train(iterations=1, log_every=1)
        snapshot = trainer.snapshot_average_strategy()
        assert isinstance(snapshot, dict)
        tlog.close() 
