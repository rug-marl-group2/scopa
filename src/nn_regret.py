from __future__ import annotations
"""Neural regret approximation utilities for Scopa.

This module provides a small JAX-based multilayer perceptron used to regress
counterfactual regrets given an observation tensor, the acting seat, and the
legal action mask.  It also includes lightweight optimizer, replay buffer, and
checkpoint helpers that are shared by the neural regret training pipeline.
"""
"""Neural regret approximation utilities used by the NN trainer."""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray
Params = List[Tuple[Array, Array]]  # (weights, bias) per layer


@dataclass
class ModelConfig:
    """Configuration for the neural regret approximator."""

    obs_planes: int = 6
    obs_cards: int = 40
    action_dim: int = 40
    hidden_layers: Sequence[int] = dataclasses.field(default_factory=lambda: (256, 128))
    seat_embedding: bool = True
    mask_input: bool = True
    activation: str = "relu"

    @property
    def input_dim(self) -> int:
        base = self.obs_planes * self.obs_cards
        if self.seat_embedding:
            base += 4
        if self.mask_input:
            base += self.action_dim
        return base


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class TrainingSample:
    observation: np.ndarray  # [obs_planes, obs_cards]
    seat: int
    mask: np.ndarray  # [action_dim]
    target: np.ndarray  # [action_dim]


class ReplayBuffer:
    """Simple FIFO replay buffer for regret regression samples."""

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._storage: List[TrainingSample] = []
        self._next_index = 0

    def add(self, sample: TrainingSample) -> None:
        if len(self._storage) < self._capacity:
            self._storage.append(sample)
        else:
            self._storage[self._next_index] = sample
        self._next_index = (self._next_index + 1) % self._capacity

    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self._storage)

    def sample(self, batch_size: int, rng: np.random.Generator) -> List[TrainingSample]:
        batch_size = min(int(batch_size), len(self._storage))
        if batch_size <= 0:
            return []
        indices = rng.choice(len(self._storage), size=batch_size, replace=False)
        return [self._storage[i] for i in indices]


class SGDOptimizer:
    """Small SGD+momentum optimizer compatible with JAX parameter PyTrees."""

    def __init__(self, config: OptimizerConfig, params: Params):
        self.config = config
        self.velocity = [jnp.zeros_like(w) for w, _ in params]
        self.velocity_bias = [jnp.zeros_like(b) for _, b in params]

    def update(self, params: Params, grads: Params) -> Params:
        lr = self.config.learning_rate
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay

        new_params: Params = []
        new_vel = []
        new_vel_b = []
        for (w, b), (gw, gb), vw, vb in zip(params, grads, self.velocity, self.velocity_bias):
            if weight_decay > 0.0:
                gw = gw + weight_decay * w
            vw = momentum * vw + gw
            vb = momentum * vb + gb
            w_new = w - lr * vw
            b_new = b - lr * vb
            new_params.append((w_new, b_new))
            new_vel.append(vw)
            new_vel_b.append(vb)
        self.velocity = new_vel
        self.velocity_bias = new_vel_b
        return new_params


class MLPRegretModel:
    """Multilayer perceptron producing regret estimates."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def init(self, key: Array) -> Params:
        sizes = [self.config.input_dim, *self.config.hidden_layers, self.config.action_dim]
        params: Params = []
        k = key
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            k, kw, kb = jax.random.split(k, 3)
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            w = jax.random.uniform(kw, (in_dim, out_dim), minval=-limit, maxval=limit)
            b = jnp.zeros((out_dim,), dtype=jnp.float32)
            params.append((w, b))
        return params

    def _activation(self, x: Array) -> Array:
        act = self.config.activation.lower()
        if act == "relu":
            return jax.nn.relu(x)
        if act == "tanh":
            return jnp.tanh(x)
        if act == "elu":
            return jax.nn.elu(x)
        raise ValueError(f"Unsupported activation '{self.config.activation}'")

    def _build_input(self, obs: Array, seat: Array, mask: Array) -> Array:
        flat = obs.reshape((obs.shape[0] * obs.shape[1],))
        parts = [flat]
        if self.config.seat_embedding:
            seat_one_hot = jax.nn.one_hot(seat.astype(jnp.int32), 4, dtype=flat.dtype)
            parts.append(seat_one_hot)
        if self.config.mask_input:
            parts.append(mask.astype(flat.dtype))
        return jnp.concatenate(parts, axis=0)

    def apply(self, params: Params, obs: Array, seat: Array, mask: Array) -> Array:
        x = self._build_input(obs, seat, mask)
        for i, (w, b) in enumerate(params):
            x = jnp.dot(x, w) + b
            if i < len(params) - 1:
                x = self._activation(x)
        return x

    def batched_apply(self, params: Params, obs: Array, seat: Array, mask: Array) -> Array:
        """Vectorized apply for training batches."""

        def _single(o, s, m):
            return self.apply(params, o, s, m)

        vapply = jax.vmap(_single, in_axes=(0, 0, 0))
        return vapply(obs, seat, mask)


def compute_loss(model: MLPRegretModel, params: Params, obs: Array, seat: Array, mask: Array, targets: Array) -> Tuple[Array, Array]:
    def loss_fn(p: Params) -> Array:
        preds = model.batched_apply(p, obs, seat, mask)
        diff = preds - targets
        weights = jnp.where(mask > 0, 1.0, 0.0)
        sq = jnp.sum((diff ** 2) * weights, axis=1)
        denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        return jnp.mean(sq / denom)

    value, grads = jax.value_and_grad(loss_fn)(params)
    return value, grads


class CheckpointManager:
    """Persist and load trainer state."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save(self, filename: str, payload: Dict[str, Any]) -> str:
        path = os.path.join(self.directory, filename)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        return path

    def load(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.directory, filename)
        with open(path, "rb") as fh:
            return pickle.load(fh)


def regret_matching(regrets: np.ndarray, mask: np.ndarray, minimum: float = 1e-6) -> np.ndarray:
    """Convert regret estimates into a probability distribution."""

    legal = mask > 0
    if not np.any(legal):
        return np.full_like(mask, 1.0 / mask.size, dtype=np.float32)
    clipped = np.maximum(regrets, 0.0)
    clipped = clipped * legal.astype(np.float32)
    total = clipped.sum()
    if total <= minimum:
        probs = legal.astype(np.float32)
        probs /= probs.sum()
        return probs
    return clipped / total


def hash_infoset(obs: np.ndarray, seat: int) -> bytes:
    return obs.tobytes() + bytes([seat & 0xFF])


@dataclass
class TrainerConfig:
    seed: int = 0
    deals_per_iter: int = 4
    mc_rollouts: int = 1
    exploration: float = 0.1
    batch_size: int = 32
    updates_per_iter: int = 1
    buffer_capacity: int = 4096
    target_update_every: int = 0
    bootstrap_monte_carlo: bool = True


class NNRegretTrainer:
    """Trainer coordinating regret regression and policy updates."""

    def __init__(
        self,
        model_config: ModelConfig,
        opt_config: OptimizerConfig,
        trainer_config: TrainerConfig,
        tlogger,
    ):
        self.model = MLPRegretModel(model_config)
        self.trainer_config = trainer_config
        self.rng = np.random.default_rng(int(trainer_config.seed))
        key = jax.random.PRNGKey(int(trainer_config.seed))
        self.params = self.model.init(key)
        self.optimizer = SGDOptimizer(opt_config, self.params)
        self.replay = ReplayBuffer(trainer_config.buffer_capacity)
        self.avg_strategy: Dict[bytes, Tuple[np.ndarray, int]] = {}
        self.target_params = self.params
        self.iteration = 0
        self.tlogger = tlogger

    def act_from_obs(self, seat: int, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        obs_j = jnp.asarray(obs)
        mask_j = jnp.asarray(mask)
        preds = self.model.apply(self.params, obs_j, jnp.array(seat, dtype=jnp.int32), mask_j)
        probs = regret_matching(np.asarray(preds), np.asarray(mask))
        if explore and self.rng.random() < self.trainer_config.exploration:
            legal = np.where(mask > 0)[0]
            return int(self.rng.choice(legal)) if legal.size > 0 else int(np.argmax(probs))
        return int(self.rng.choice(np.arange(probs.size), p=probs))

    def _record_average_strategy(self, obs: np.ndarray, seat: int, probs: np.ndarray) -> None:
        key = hash_infoset(obs, seat)
        total, count = self.avg_strategy.get(key, (np.zeros_like(probs), 0))
        total = total + probs
        self.avg_strategy[key] = (total, count + 1)

    def _evaluate_terminal(self, state) -> Tuple[np.ndarray, np.ndarray]:
        from cfr_jax import np_round_scores

        return np_round_scores(state)

    def _rollout_uniform(self, state, start_seat: int) -> float:
        from cfr_jax import np_clone_state, np_is_terminal, np_legal_mask, np_step

        total = 0.0
        for _ in range(max(1, self.trainer_config.mc_rollouts)):
            rollout_state = np_clone_state(state)
            while not np_is_terminal(rollout_state):
                mask = np.asarray(np_legal_mask(rollout_state))
                legal = np.nonzero(mask)[0]
                if legal.size == 0:
                    break
                action = int(self.rng.choice(legal))
                rollout_state, _ = np_step(rollout_state, action)
            team_points, _ = self._evaluate_terminal(rollout_state)
            team = 0 if start_seat in (0, 2) else 1
            total += float(team_points[team])
        return total / max(1, self.trainer_config.mc_rollouts)

    def _compute_regret_targets(self, state, seat: int) -> np.ndarray:
        from cfr_jax import np_clone_state, np_legal_mask, np_step

        mask = np.asarray(np_legal_mask(state))
        targets = np.zeros((mask.size,), dtype=np.float32)
        legal = np.nonzero(mask)[0]
        if legal.size == 0:
            return targets
        if not self.trainer_config.bootstrap_monte_carlo:
            return targets
        values = []
        for action in legal:
            clone = np_clone_state(state)
            clone, _ = np_step(clone, int(action))
            values.append(self._rollout_uniform(clone, seat))
        values_np = np.asarray(values, dtype=np.float32)
        baseline = float(values_np.mean())
        for a, v in zip(legal, values_np):
            targets[a] = v - baseline
        return targets

    def _generate_deal(self) -> None:
        from cfr_jax import np_init_state, np_is_terminal, np_legal_mask, np_build_obs, np_step

        state = np_init_state(self.rng)
        while not np_is_terminal(state):
            seat = int(state.cur_player)
            obs = np.asarray(np_build_obs(state, seat), dtype=np.float32)
            mask = np.asarray(np_legal_mask(state), dtype=np.float32)
            preds = self.model.apply(
                self.target_params,
                jnp.asarray(obs),
                jnp.array(seat, dtype=jnp.int32),
                jnp.asarray(mask),
            )
            probs = regret_matching(np.asarray(preds), np.asarray(mask))
            self._record_average_strategy(obs, seat, probs)
            action = self.act_from_obs(seat, obs, mask)
            targets = self._compute_regret_targets(state, seat)
            sample = TrainingSample(observation=obs, seat=seat, mask=mask, target=targets)
            self.replay.add(sample)
            state, _ = np_step(state, action)

    def _train_step(self) -> Optional[float]:
        batch = self.replay.sample(self.trainer_config.batch_size, self.rng)
        if not batch:
            return None
        obs = jnp.asarray([s.observation for s in batch])
        seat = jnp.asarray([s.seat for s in batch], dtype=jnp.int32)
        mask = jnp.asarray([s.mask for s in batch], dtype=jnp.float32)
        targets = jnp.asarray([s.target for s in batch], dtype=jnp.float32)
        loss, grads = compute_loss(self.model, self.params, obs, seat, mask, targets)
        self.params = self.optimizer.update(self.params, grads)
        if self.trainer_config.target_update_every > 0 and (
            self.iteration % self.trainer_config.target_update_every == 0
        ):
            self.target_params = self.params
        return float(loss)

    def evaluate_policy(self, episodes: int = 16) -> Dict[str, float]:
        from cfr_jax import np_init_state, np_is_terminal, np_legal_mask, np_build_obs, np_step

        wins = 0
        ties = 0
        scopa_totals = np.zeros(4, dtype=np.float32)
        for _ in range(max(1, episodes)):
            state = np_init_state(self.rng)
            while not np_is_terminal(state):
                seat = int(state.cur_player)
                obs = np.asarray(np_build_obs(state, seat), dtype=np.float32)
                mask = np.asarray(np_legal_mask(state), dtype=np.float32)
                if seat in (0, 2):
                    action = self.act_from_obs(seat, obs, mask, explore=False)
                else:
                    legal = np.nonzero(mask)[0]
                    if legal.size == 0:
                        action = int(np.argmax(mask))
                    else:
                        action = int(self.rng.choice(legal))
                state, _ = np_step(state, action)
            team_points, _ = self._evaluate_terminal(state)
            # Accumulate scopa counts per player for averaging.
            scopa_totals += np.asarray(state.scopas, dtype=np.float32)
            if team_points[0] > team_points[1]:
                wins += 1
            elif team_points[0] == team_points[1]:
                ties += 1
        total = max(1, episodes)
        scopa_avg = scopa_totals / float(total)
        return {
            "win_rate": wins / total,
            "tie_rate": ties / total,
            "scopa_avg/p0": float(scopa_avg[0]),
            "scopa_avg/p1": float(scopa_avg[1]),
            "scopa_avg/p2": float(scopa_avg[2]),
            "scopa_avg/p3": float(scopa_avg[3]),
        }
    
    
    def train(self, iterations: int, log_every: int = 10) -> None:
        for it in range(1, int(iterations) + 1):
            self.iteration = it
            for _ in range(self.trainer_config.deals_per_iter):
                self._generate_deal()
            losses = []
            for _ in range(self.trainer_config.updates_per_iter):
                loss = self._train_step()
                if loss is not None:
                    losses.append(loss)
            if losses:
                mean_loss = float(np.mean(losses))
                if self.tlogger is not None:
                    self.tlogger.writer.add_scalar("nn_regret/loss", mean_loss, it)
            if log_every > 0 and it % int(log_every) == 0:
                metrics = self.evaluate_policy(episodes=4)
                if self.tlogger is not None:
                    for key, value in metrics.items():
                        self.tlogger.writer.add_scalar(f"nn_regret/{key}", value, it)
                print(f"[Iter {it}] loss={np.mean(losses) if losses else 0:.4f} metrics={metrics}")
        if self.tlogger is not None:
            self.tlogger.flush()
    def snapshot_average_strategy(self) -> Dict[bytes, np.ndarray]:
        snapshot: Dict[bytes, np.ndarray] = {}
        for key, (total, count) in self.avg_strategy.items():
            if count > 0:
                snapshot[key] = total / float(count)
        return snapshot