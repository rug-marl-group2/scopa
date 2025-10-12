"""Centralized Training with Decentralized Execution trainer for Scopa."""
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Optional
import os
import pickle
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np


Array = jnp.ndarray


def tree_copy(params):
    return jtu.tree_map(lambda x: x.copy(), params)


def tree_soft_update(target, source, tau):
    return jtu.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)


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
               target_critic_params: Sequence[Dict[str, Array]],
               batch: Dict[str, Array],
               num_seats: int) -> Array:
    logits = mlp_forward(params, batch["obs"])
    masked_logits = mask_logits(logits, batch["mask"])
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    actions = batch["actions"]
    idx = jnp.arange(actions.shape[0])
    chosen_log_probs = log_probs[idx, actions]
    baseline = critic_forward(target_critic_params, batch["global_state"], batch["seat"], num_seats)
    advantages = jax.lax.stop_gradient(batch["returns"] - baseline)
    return -jnp.mean(chosen_log_probs * advantages)


def critic_loss(params: Sequence[Dict[str, Array]],
                batch: Dict[str, Array],
                num_seats: int) -> Array:
    values = critic_forward(params, batch["global_state"], batch["seat"], num_seats)
    targets = batch["returns"]
    diff = values - targets
    return jnp.mean(jnp.square(diff))


def _actor_step(params: Sequence[Dict[str, Array]],
                target_critic_params: Sequence[Dict[str, Array]],
                batch: Dict[str, Array],
                lr: float,
                num_seats: int):
    loss, grads = jax.value_and_grad(actor_loss)(params, target_critic_params, batch, num_seats)
    new_params = jtu.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


def _critic_step(params: Sequence[Dict[str, Array]],
                 batch: Dict[str, Array],
                 lr: float,
                 num_seats: int):
    loss, grads = jax.value_and_grad(critic_loss)(params, batch, num_seats)
    new_params = jtu.tree_map(lambda p, g: p - lr * g, params, grads)
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
                 target_tau: float = 0.01,
                 target_update_interval: int = 1,
                 tlogger: object = None):
        self.env_fn = env_fn
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tlogger = tlogger

        self.target_tau = float(target_tau)
        self.target_update_interval = max(int(target_update_interval), 1)
        self._update_counter = 0

        self.rng_key = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)
        self.eval_rng = np.random.default_rng(seed + 100003)

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

        self.target_actor_params = tree_copy(self.actor_params)
        self.target_critic_params = tree_copy(self.critic_params)
        self.best_vs_random_win_rate: float = float("-inf")

    def _soft_update_targets(self):
        tau = self.target_tau
        self.target_actor_params = tree_soft_update(self.target_actor_params, self.actor_params, tau)
        self.target_critic_params = tree_soft_update(self.target_critic_params, self.critic_params, tau)

    def sample_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float,
                      params: Sequence[Dict[str, Array]] = None,
                      rng=None) -> int:
        obs_flat = jnp.asarray(obs.reshape(-1), dtype=jnp.float32)
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        net_params = self.actor_params if params is None else params
        rng = self.np_rng if rng is None else rng
        logits = mlp_forward(net_params, obs_flat)
        masked = mask_logits(logits, mask_arr)
        probs = jax.nn.softmax(masked)
        probs_np = np.asarray(probs, dtype=np.float32)
        legal = np.flatnonzero(np.asarray(mask, dtype=np.float32) > 0.0)
        if legal.size == 0:
            return 0
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.choice(legal))
        safe_probs = probs_np[legal]
        total = safe_probs.sum()
        if total <= 0.0 or not np.isfinite(total):
            return int(rng.choice(legal))
        safe_probs = safe_probs / total
        chosen = int(rng.choice(legal, p=safe_probs))
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

    def play_episode(self, epsilon: float, actor_params: Sequence[Dict[str, Array]], mode: str):
        env = self.env_fn()
        seed = int(self.eval_rng.integers(0, 2**31 - 1))
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
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
            if mode == "vs_random" and seat % 2 == 1:
                legal = np.flatnonzero(mask > 0.0)
                if legal.size == 0:
                    action = 0
                else:
                    action = int(self.eval_rng.choice(legal))
            else:
                action = self.sample_action(obs, mask, epsilon, params=actor_params, rng=self.eval_rng)
            env.step(int(action))
            steps += 1
        if hasattr(env, "close"):
            env.close()
        team_reward = float(final_rewards[0]) if final_rewards.size > 0 else 0.0
        return {
            "team_reward": team_reward,
            "final_rewards": final_rewards,
            "steps": steps,
        }

    def evaluate(self, episodes: int = 10, epsilon: float = 0.0, use_target: bool = True, incl_vs_random: bool = False) -> Dict[str, float]:
        params = self.target_actor_params if use_target else self.actor_params
        modes = [("self", "self")]
        if incl_vs_random:
            modes.append(("vs_random", "vs_random"))
        results: Dict[str, float] = {}
        for mode_key, mode in modes:
            returns: List[float] = []
            wins = 0
            ties = 0
            total_steps = 0
            count = max(int(episodes), 0)
            for _ in range(count):
                info = self.play_episode(epsilon, params, mode)
                ret = float(info["team_reward"])
                returns.append(ret)
                total_steps += int(info["steps"])
                if ret > 0.0:
                    wins += 1
                elif ret == 0.0:
                    ties += 1
            n = len(returns)
            avg_return = float(np.mean(returns)) if n > 0 else 0.0
            win_rate = float(wins) / float(n) if n > 0 else 0.0
            tie_rate = float(ties) / float(n) if n > 0 else 0.0
            avg_steps = float(total_steps) / float(n) if n > 0 else 0.0
            prefix = mode_key
            results[f"{prefix}/avg_return"] = avg_return
            results[f"{prefix}/win_rate"] = win_rate
            results[f"{prefix}/tie_rate"] = tie_rate
            results[f"{prefix}/avg_steps"] = avg_steps
        return results

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

    def train(self, epochs: int, episodes_per_epoch: int,
              eval_every: int = 0,
              eval_episodes: int = 16,
              eval_use_target: bool = True,
              eval_vs_random: bool = False,
              best_save_path: Optional[str] = None,
              best_actor_source: str = "target"):
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
            self.actor_params, actor_l = actor_step(self.actor_params, self.target_critic_params, batch, self.actor_lr, self.num_seats)
            self.critic_params, critic_l = critic_step(self.critic_params, batch, self.critic_lr, self.num_seats)
            self._update_counter += 1
            if self._update_counter % self.target_update_interval == 0:
                self._soft_update_targets()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            avg_return = float(np.mean(returns)) if returns else 0.0
            eval_metrics = None
            if eval_every > 0 and (epoch % eval_every == 0):
                include_vs_random = eval_vs_random or (best_save_path is not None)
                eval_metrics = self.evaluate(episodes=eval_episodes, epsilon=0.0,
                                             use_target=eval_use_target, incl_vs_random=include_vs_random)
            if self.tlogger is not None:
                self.tlogger.writer.add_scalar("CTDE/actor_loss", float(actor_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/critic_loss", float(critic_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/avg_return", avg_return, epoch)
                self.tlogger.writer.add_scalar("CTDE/epsilon", float(self.epsilon), epoch)
                if eval_metrics is not None:
                    for key, value in eval_metrics.items():
                        self.tlogger.writer.add_scalar(f"CTDEEval/{key}", float(value), epoch)
            print(f"Epoch {epoch:04d} | avg_return={avg_return:.3f} | actor_loss={float(actor_l):.4f} | critic_loss={float(critic_l):.4f} | eps={self.epsilon:.3f}")
            if eval_metrics is not None:
                grouped: Dict[str, Dict[str, float]] = {}
                for key, value in eval_metrics.items():
                    if "/" in key:
                        mode, metric = key.split("/", 1)
                    else:
                        mode, metric = "self", key
                    grouped.setdefault(mode, {})[metric] = float(value)
                mode_summaries = []
                for mode, metrics in grouped.items():
                    mode_summaries.append("{mode}: avg_return={avg:.3f} win_rate={win:.3f} tie_rate={tie:.3f} avg_steps={steps:.2f}".format(
                        mode=mode,
                        avg=float(metrics.get("avg_return", 0.0)),
                        win=float(metrics.get("win_rate", 0.0)),
                        tie=float(metrics.get("tie_rate", 0.0)),
                        steps=float(metrics.get("avg_steps", 0.0))
                    ))
                print("    Eval -> " + " || ".join(mode_summaries))
                if best_save_path is not None:
                    win_rate = float(eval_metrics.get("vs_random/win_rate", float("nan")))
                    if np.isfinite(win_rate):
                        if win_rate > self.best_vs_random_win_rate:
                            self.best_vs_random_win_rate = win_rate
                            actor_src = best_actor_source if best_actor_source in {"target", "online"} else "target"
                            try:
                                self.save(best_save_path, checkpoint_type="best", actor_source=actor_src)
                                print(f"[CTDE] Saved new best checkpoint to {best_save_path} (win_rate_team0={win_rate:.3f})")
                            except Exception as exc:
                                print(f"[CTDE] WARNING: failed to save best checkpoint to {best_save_path}: {exc}")

    def save(self, path: str, checkpoint_type: str = "final", actor_source: str = "online") -> None:
        """Persist trainer parameters to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        actor_source = actor_source if actor_source in {"online", "target"} else "online"
        actor_params = self.target_actor_params if actor_source == "target" else self.actor_params
        payload = {
            "checkpoint_type": checkpoint_type,
            "actor_source": actor_source,
            "actor_params": jtu.tree_map(lambda x: np.asarray(x), actor_params),
            "critic_params": jtu.tree_map(lambda x: np.asarray(x), self.critic_params),
            "target_actor_params": jtu.tree_map(lambda x: np.asarray(x), self.target_actor_params),
            "target_critic_params": jtu.tree_map(lambda x: np.asarray(x), self.target_critic_params),
            "best_vs_random_win_rate": float(self.best_vs_random_win_rate),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
