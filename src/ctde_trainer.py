"""Centralized Training with Decentralized Execution trainer for Scopa."""
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


def init_mlp(key: Array, sizes: Sequence[int]) -> List[Dict[str, Array]]:
    """Initialize a simple feed-forward network."""
    params: List[Dict[str, Array]] = []
    if len(sizes) < 2:
        raise ValueError("sizes must include input and output dimensions")
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (in_dim, out_dim) in zip(keys, zip(sizes[:-1], sizes[1:])):
        limit = 1.0 / np.sqrt(max(in_dim, 1))
        w = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
        b = jnp.zeros((out_dim,), dtype=jnp.float32)
        params.append({"w": w, "b": b})
    return params


def mlp_forward(params: Sequence[Dict[str, Array]], x: Array) -> Array:
    """Apply the MLP to the provided input."""
    h = x
    for i, layer in enumerate(params):
        h = jnp.matmul(h, layer["w"]) + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return h


def mask_logits(logits: Array, mask: Array) -> Array:
    """Apply an action mask by setting illegal logits to a large negative value."""
    neg = jnp.full_like(logits, -1e9)
    return jnp.where(mask > 0.5, logits, neg)


@dataclass
class Transition:
    seat: int
    obs: np.ndarray
    mask: np.ndarray
    action: int
    ret: float
    global_state: np.ndarray


def critic_forward(params: Sequence[Dict[str, Array]], global_state: Array, seat: Array, num_seats: int) -> Array:
    seat_one_hot = jax.nn.one_hot(seat, num_seats, dtype=jnp.float32)
    x = jnp.concatenate([global_state, seat_one_hot], axis=-1)
    value = mlp_forward(params, x)
    return jnp.squeeze(value, axis=-1)


def actor_loss(params: Sequence[Dict[str, Array]],
               critic_params: Sequence[Dict[str, Array]],
               batch: Dict[str, Array],
               num_seats: int) -> Array:
    logits = mlp_forward(params, batch["obs"])
    masked_logits = mask_logits(logits, batch["mask"])
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    actions = batch["actions"]
    idx = jnp.arange(actions.shape[0])
    chosen_log_probs = log_probs[idx, actions]
    values = critic_forward(critic_params, batch["global_state"], batch["seat"], num_seats)
    advantages = jax.lax.stop_gradient(batch["returns"] - values)
    return -jnp.mean(chosen_log_probs * advantages)


def critic_loss(params: Sequence[Dict[str, Array]],
                batch: Dict[str, Array],
                num_seats: int) -> Array:
    values = critic_forward(params, batch["global_state"], batch["seat"], num_seats)
    targets = batch["returns"]
    diff = values - targets
    return jnp.mean(jnp.square(diff))


def _actor_step(params: Sequence[Dict[str, Array]],
                critic_params: Sequence[Dict[str, Array]],
                batch: Dict[str, Array],
                lr: float,
                num_seats: int):
    loss, grads = jax.value_and_grad(actor_loss)(params, critic_params, batch, num_seats)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


def _critic_step(params: Sequence[Dict[str, Array]],
                 batch: Dict[str, Array],
                 lr: float,
                 num_seats: int):
    loss, grads = jax.value_and_grad(critic_loss)(params, batch, num_seats)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


actor_step = jax.jit(_actor_step, static_argnums=(4,))
critic_step = jax.jit(_critic_step, static_argnums=(3,))


class CTDETrainer:
    """Simple MADDPG-style trainer with centralized critic and decentralized actors."""

    def __init__(self,
                 env_fn: Callable[[], object],
                 seed: int = 0,
                 actor_hidden: Sequence[int] = (256, 128),
                 critic_hidden: Sequence[int] = (256, 128),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 0.2,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 tlogger: object = None):
        self.env_fn = env_fn
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tlogger = tlogger

        self.rng_key = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)

        env = self.env_fn()
        try:
            env.reset()
        except TypeError:
            env.reset(None)
        sample_agent = env.agent_selection
        sample_obs = np.asarray(env.observations[sample_agent], dtype=np.float32)
        sample_mask = np.asarray(env.infos[sample_agent]["action_mask"], dtype=np.float32)
        global_state = env.get_global_state()
        self.obs_dim = int(sample_obs.size)
        self.action_dim = int(sample_mask.size)
        self.global_state_dim = int(global_state.size)
        self.num_seats = len(env.game.players)
        if hasattr(env, "close"):
            env.close()

        self.rng_key, actor_key = jax.random.split(self.rng_key)
        actor_sizes = [self.obs_dim, *actor_hidden, self.action_dim]
        self.actor_params = init_mlp(actor_key, actor_sizes)

        self.rng_key, critic_key = jax.random.split(self.rng_key)
        critic_sizes = [self.global_state_dim + self.num_seats, *critic_hidden, 1]
        self.critic_params = init_mlp(critic_key, critic_sizes)

    def sample_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        obs_flat = jnp.asarray(obs.reshape(-1), dtype=jnp.float32)
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        logits = mlp_forward(self.actor_params, obs_flat)
        masked = mask_logits(logits, mask_arr)
        probs = jax.nn.softmax(masked)
        probs_np = np.asarray(probs, dtype=np.float32)
        legal = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 0.0)
        if legal.size == 0:
            return 0
        if epsilon > 0.0 and self.np_rng.random() < epsilon:
            return int(self.np_rng.choice(legal))
        safe_probs = probs_np[legal]
        total = safe_probs.sum()
        if total <= 0.0 or not np.isfinite(total):
            return int(self.np_rng.choice(legal))
        safe_probs = safe_probs / total
        chosen = int(self.np_rng.choice(legal, p=safe_probs))
        return chosen

    def collect_episode(self, epsilon: float):
        env = self.env_fn()
        seed = int(self.np_rng.integers(0, 2**31 - 1))
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        trajectories: Dict[int, List[Dict[str, object]]] = {i: [] for i in range(self.num_seats)}
        final_rewards = np.zeros(self.num_seats, dtype=np.float32)
        steps = 0
        while True:
            agent = env.agent_selection
            if agent is None:
                break
            if env.terminations[agent] or env.truncations[agent]:
                if all(env.terminations.values()) or all(env.truncations.values()):
                    for name, reward in env.rewards.items():
                        seat = env.agent_name_mapping[name]
                        final_rewards[seat] = reward
                    break
                env.step(None)
                continue
            seat = env.agent_name_mapping[agent]
            obs = np.asarray(env.observations[agent], dtype=np.float32)
            mask = np.asarray(env.infos[agent]["action_mask"], dtype=np.float32)
            global_state = np.asarray(env.get_global_state(), dtype=np.float32)
            action = self.sample_action(obs, mask, epsilon)
            env.step(int(action))
            reward = float(env.rewards[agent])
            trajectories[seat].append({
                "obs": obs,
                "mask": mask,
                "action": int(action),
                "reward": reward,
                "global_state": global_state,
            })
            steps += 1
        if hasattr(env, "close"):
            env.close()

        transitions: List[Transition] = []
        for seat, seq in trajectories.items():
            if not seq:
                continue
            seq[-1]["reward"] = float(final_rewards[seat])
            g = 0.0
            for item in reversed(seq):
                g = item["reward"] + self.gamma * g
                item["return"] = g
            for item in seq:
                transitions.append(Transition(seat=seat,
                                              obs=item["obs"],
                                              mask=item["mask"],
                                              action=item["action"],
                                              ret=float(item["return"]),
                                              global_state=item["global_state"]))
        team_reward = float(final_rewards[0]) if final_rewards.size > 0 else 0.0
        info = {
            "team_reward": team_reward,
            "steps": steps,
            "final_rewards": final_rewards,
        }
        return transitions, info

    def _batchify(self, transitions: List[Transition]) -> Dict[str, Array]:
        obs = jnp.asarray(np.stack([t.obs.reshape(-1) for t in transitions], axis=0), dtype=jnp.float32)
        mask = jnp.asarray(np.stack([t.mask for t in transitions], axis=0), dtype=jnp.float32)
        actions = jnp.asarray(np.array([t.action for t in transitions], dtype=np.int32))
        returns = jnp.asarray(np.array([t.ret for t in transitions], dtype=np.float32))
        global_state = jnp.asarray(np.stack([t.global_state for t in transitions], axis=0), dtype=jnp.float32)
        seat = jnp.asarray(np.array([t.seat for t in transitions], dtype=np.int32))
        return {
            "obs": obs,
            "mask": mask,
            "actions": actions,
            "returns": returns,
            "global_state": global_state,
            "seat": seat,
        }

    def train(self, epochs: int, episodes_per_epoch: int):
        total_steps = 0
        for epoch in range(1, epochs + 1):
            batch_transitions: List[Transition] = []
            returns: List[float] = []
            for _ in range(episodes_per_epoch):
                episode_transitions, info = self.collect_episode(self.epsilon)
                if episode_transitions:
                    batch_transitions.extend(episode_transitions)
                returns.append(info["team_reward"])
                total_steps += info["steps"]
            if not batch_transitions:
                continue
            batch = self._batchify(batch_transitions)
            self.actor_params, actor_l = actor_step(self.actor_params, self.critic_params, batch, self.actor_lr, self.num_seats)
            self.critic_params, critic_l = critic_step(self.critic_params, batch, self.critic_lr, self.num_seats)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            avg_return = float(np.mean(returns)) if returns else 0.0
            if self.tlogger is not None:
                self.tlogger.writer.add_scalar("CTDE/actor_loss", float(actor_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/critic_loss", float(critic_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/avg_return", avg_return, epoch)
                self.tlogger.writer.add_scalar("CTDE/epsilon", float(self.epsilon), epoch)
            print(f"Epoch {epoch:04d} | avg_return={avg_return:.3f} | actor_loss={float(actor_l):.4f} | critic_loss={float(critic_l):.4f} | eps={self.epsilon:.3f}")
