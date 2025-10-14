"""Centralized Training with Decentralized Execution trainer for Scopa."""
from dataclasses import dataclass, fields
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


_RECOMMENDED_METRIC_WEIGHTS = {
    "win": 2.0,
    "points": 0.05,
    "scopa": 0.75,
    "most_cards": 0.50,
    "most_coins": 0.45,
    "sette_bello": 0.7,
    "primiera": 0.3,
}


@dataclass
class MetricWeights:
    """Weights used when shaping rewards and evaluation metrics.

    The defaults prioritize finishing the hand ahead on total points while still
    rewarding intermediate objectives that strongly correlate with eventual
    wins.  They were selected from self-play sweeps where a heavier win signal
    paired with moderate category emphasis produced the most stable training.
    """

    win: float = _RECOMMENDED_METRIC_WEIGHTS["win"]
    points: float = _RECOMMENDED_METRIC_WEIGHTS["points"]
    scopa: float = _RECOMMENDED_METRIC_WEIGHTS["scopa"]
    most_cards: float = _RECOMMENDED_METRIC_WEIGHTS["most_cards"]
    most_coins: float = _RECOMMENDED_METRIC_WEIGHTS["most_coins"]
    sette_bello: float = _RECOMMENDED_METRIC_WEIGHTS["sette_bello"]
    primiera: float = _RECOMMENDED_METRIC_WEIGHTS["primiera"]

    @classmethod
    def recommended(cls) -> "MetricWeights":
        """Return the recommended default weight configuration."""

        return cls(**_RECOMMENDED_METRIC_WEIGHTS)

    @classmethod
    def from_overrides(cls, overrides: Dict[str, float]) -> "MetricWeights":
        base = dict(_RECOMMENDED_METRIC_WEIGHTS)
        for key, value in overrides.items():
            if key in base:
                try:
                    base[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return cls(**base)

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


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
                 tlogger: object = None,
                 metric_weights: Optional[MetricWeights] = None):
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

        self.metric_weights = metric_weights if metric_weights is not None else MetricWeights()

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
        # Track the best weighted objective attained versus random opponents.
        self.best_vs_random_score: float = float("-inf")

    @staticmethod
    def _team_indices(num_seats: int) -> Dict[int, List[int]]:
        return {
            0: [i for i in range(0, num_seats, 2)],
            1: [i for i in range(1, num_seats, 2)],
        }

    def _extract_final_metrics(self, env) -> Dict[str, float]:
        players = env.game.players
        teams = self._team_indices(len(players))

        def gather(team_id: int):
            member_idx = teams[team_id]
            members = [players[i] for i in member_idx]
            captures = [card for p in members for card in p.captures]
            scopas = sum(p.scopas for p in members)
            return members, captures, scopas

        (_, caps0, scopa0) = gather(0)
        (_, caps1, scopa1) = gather(1)

        def most_cards(c0: List[object], c1: List[object]):
            if len(c0) > len(c1):
                return 1.0, 0.0
            if len(c1) > len(c0):
                return 0.0, 1.0
            return 0.0, 0.0

        def most_coins(c0: List[object], c1: List[object]):
            coins0 = sum(1 for card in c0 if getattr(card, "suit", "") == "bello")
            coins1 = sum(1 for card in c1 if getattr(card, "suit", "") == "bello")
            if coins0 > coins1:
                return 1.0, 0.0
            if coins1 > coins0:
                return 0.0, 1.0
            return 0.0, 0.0

        def sette_bello(c0: List[object], c1: List[object]):
            has0 = any(getattr(card, "suit", "") == "bello" and getattr(card, "rank", 0) == 7 for card in c0)
            has1 = any(getattr(card, "suit", "") == "bello" and getattr(card, "rank", 0) == 7 for card in c1)
            return (1.0 if has0 else 0.0, 1.0 if has1 else 0.0)

        suit_priority = {7: 4, 6: 3, 1: 2, 5: 1, 4: 0, 3: 0, 2: 0}
        suit_order = ["picche", "bello", "fiori", "cuori"]

        def primiera_score(captures: List[object]) -> int:
            total = 0
            for suit in suit_order:
                best_val = -1
                for card in captures:
                    if getattr(card, "suit", None) != suit:
                        continue
                    val = suit_priority.get(getattr(card, "rank", 0), 0)
                    if val > best_val:
                        best_val = val
                if best_val > 0:
                    total += best_val
            return total

        mc0, mc1 = most_cards(caps0, caps1)
        mb0, mb1 = most_coins(caps0, caps1)
        sb0, sb1 = sette_bello(caps0, caps1)
        prim0 = primiera_score(caps0)
        prim1 = primiera_score(caps1)
        pr0 = 1.0 if prim0 > prim1 else 0.0
        pr1 = 1.0 if prim1 > prim0 else 0.0

        points0 = scopa0 + mc0 + mb0 + sb0 + pr0
        points1 = scopa1 + mc1 + mb1 + sb1 + pr1

        if points0 > points1:
            win_result = 1.0
        elif points1 > points0:
            win_result = -1.0
        else:
            win_result = 0.0

        return {
            "points0": float(points0),
            "points1": float(points1),
            "scopas0": float(scopa0),
            "scopas1": float(scopa1),
            "most_cards0": float(mc0),
            "most_cards1": float(mc1),
            "most_coins0": float(mb0),
            "most_coins1": float(mb1),
            "sette_bello0": float(sb0),
            "sette_bello1": float(sb1),
            "primiera0": float(pr0),
            "primiera1": float(pr1),
            "primiera_score0": float(prim0),
            "primiera_score1": float(prim1),
            "win_result": float(win_result),
        }

    def _compute_weighted_reward(self, metrics: Dict[str, float]) -> float:
        w = self.metric_weights
        score_diff = metrics.get("points0", 0.0) - metrics.get("points1", 0.0)
        scopa_diff = metrics.get("scopas0", 0.0) - metrics.get("scopas1", 0.0)
        mc_diff = metrics.get("most_cards0", 0.0) - metrics.get("most_cards1", 0.0)
        mb_diff = metrics.get("most_coins0", 0.0) - metrics.get("most_coins1", 0.0)
        sb_diff = metrics.get("sette_bello0", 0.0) - metrics.get("sette_bello1", 0.0)
        pr_diff = metrics.get("primiera0", 0.0) - metrics.get("primiera1", 0.0)
        win_signal = metrics.get("win_result", 0.0)

        weighted = (
            w.win * win_signal
            + w.points * score_diff
            + w.scopa * scopa_diff
            + w.most_cards * mc_diff
            + w.most_coins * mb_diff
            + w.sette_bello * sb_diff
            + w.primiera * pr_diff
        )
        return float(weighted)

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
        metrics = self._extract_final_metrics(env)
        if hasattr(env, "close"):
            env.close()
        weighted_reward = self._compute_weighted_reward(metrics)
        metrics["weighted_reward"] = weighted_reward
        metrics["raw_team_reward"] = metrics.get("win_result", 0.0)

        for seat in range(self.num_seats):
            if seat % 2 == 0:
                final_rewards[seat] = weighted_reward
            else:
                final_rewards[seat] = -weighted_reward

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
        team_reward = float(weighted_reward)
        info = {
            "team_reward": team_reward,
            "raw_team_reward": metrics.get("raw_team_reward", 0.0),
            "steps": steps,
            "final_rewards": final_rewards,
            "metrics": metrics,
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
        metrics = self._extract_final_metrics(env)
        if hasattr(env, "close"):
            env.close()
        weighted_reward = self._compute_weighted_reward(metrics)
        metrics["weighted_reward"] = weighted_reward
        metrics["raw_team_reward"] = metrics.get("win_result", 0.0)
        for seat in range(self.num_seats):
            final_rewards[seat] = weighted_reward if (seat % 2 == 0) else -weighted_reward
        team_reward = float(weighted_reward)
        return {
            "team_reward": team_reward,
            "raw_team_reward": metrics.get("raw_team_reward", 0.0),
            "final_rewards": final_rewards,
            "steps": steps,
            "metrics": metrics,
        }

    def evaluate(self, episodes: int = 10, epsilon: float = 0.0, use_target: bool = True, incl_vs_random: bool = False) -> Dict[str, float]:
        params = self.target_actor_params if use_target else self.actor_params
        modes = [("self", "self")]
        if incl_vs_random:
            modes.append(("vs_random", "vs_random"))
        results: Dict[str, float] = {}
        for mode_key, mode in modes:
            weighted_scores: List[float] = []
            raw_scores: List[float] = []
            wins = 0
            ties = 0
            total_steps = 0
            metrics_accum: Dict[str, float] = {}
            count = max(int(episodes), 0)
            for _ in range(count):
                info = self.play_episode(epsilon, params, mode)
                weighted_scores.append(float(info["team_reward"]))
                raw_scores.append(float(info.get("raw_team_reward", 0.0)))
                total_steps += int(info["steps"])
                metrics = info.get("metrics", {})
                for key, value in metrics.items():
                    metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)
                win_signal = float(metrics.get("win_result", 0.0))
                if win_signal > 0.0:
                    wins += 1
                elif win_signal == 0.0:
                    ties += 1
            n = len(weighted_scores)
            avg_weighted = float(np.mean(weighted_scores)) if n > 0 else 0.0
            avg_raw = float(np.mean(raw_scores)) if n > 0 else 0.0
            win_rate = float(wins) / float(n) if n > 0 else 0.0
            tie_rate = float(ties) / float(n) if n > 0 else 0.0
            avg_steps = float(total_steps) / float(n) if n > 0 else 0.0
            prefix = mode_key
            results[f"{prefix}/avg_weighted_reward"] = avg_weighted
            results[f"{prefix}/avg_raw_reward"] = avg_raw
            results[f"{prefix}/weighted_score"] = avg_weighted
            results[f"{prefix}/win_rate"] = win_rate
            results[f"{prefix}/tie_rate"] = tie_rate
            results[f"{prefix}/avg_steps"] = avg_steps
            for key, total in metrics_accum.items():
                results[f"{prefix}/avg_{key}"] = float(total) / float(n) if n > 0 else 0.0
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
            weighted_returns: List[float] = []
            for _ in range(episodes_per_epoch):
                episode_transitions, info = self.collect_episode(self.epsilon)
                if episode_transitions:
                    batch_transitions.extend(episode_transitions)
                weighted_returns.append(info["team_reward"])
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
            avg_weighted_reward = float(np.mean(weighted_returns)) if weighted_returns else 0.0
            eval_metrics = None
            if eval_every > 0 and (epoch % eval_every == 0):
                include_vs_random = eval_vs_random or (best_save_path is not None)
                eval_metrics = self.evaluate(episodes=eval_episodes, epsilon=0.0,
                                             use_target=eval_use_target, incl_vs_random=include_vs_random)
            if self.tlogger is not None:
                self.tlogger.writer.add_scalar("CTDE/actor_loss", float(actor_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/critic_loss", float(critic_l), epoch)
                self.tlogger.writer.add_scalar("CTDE/avg_weighted_reward", avg_weighted_reward, epoch)
                self.tlogger.writer.add_scalar("CTDE/epsilon", float(self.epsilon), epoch)
                if eval_metrics is not None:
                    for key, value in eval_metrics.items():
                        self.tlogger.writer.add_scalar(f"CTDEEval/{key}", float(value), epoch)
            print(f"Epoch {epoch:04d} | avg_weighted={avg_weighted_reward:.3f} | actor_loss={float(actor_l):.4f} | critic_loss={float(critic_l):.4f} | eps={self.epsilon:.3f}")
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
                    mode_summaries.append(
                        "{mode}: weighted={weighted:.3f} raw={raw:.3f} win_rate={win:.3f} tie_rate={tie:.3f} avg_steps={steps:.2f}".format(
                            mode=mode,
                            weighted=float(metrics.get("avg_weighted_reward", 0.0)),
                            raw=float(metrics.get("avg_raw_reward", 0.0)),
                            win=float(metrics.get("win_rate", 0.0)),
                            tie=float(metrics.get("tie_rate", 0.0)),
                            steps=float(metrics.get("avg_steps", 0.0))
                        )
                    )
                print("    Eval -> " + " || ".join(mode_summaries))
                if best_save_path is not None:
                    weighted_score = float(eval_metrics.get("vs_random/weighted_score", float("nan")))
                    if np.isfinite(weighted_score):
                        if weighted_score > self.best_vs_random_score:
                            self.best_vs_random_score = weighted_score
                            actor_src = best_actor_source if best_actor_source in {"target", "online"} else "target"
                            try:
                                self.save(best_save_path, checkpoint_type="best", actor_source=actor_src)
                                print(f"[CTDE] Saved new best checkpoint to {best_save_path} (weighted_score={weighted_score:.3f})")
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
            "best_vs_random_score": float(self.best_vs_random_score),
            # Retain legacy key for backwards compatibility.
            "best_vs_random_win_rate": float(self.best_vs_random_score),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load_policy(path: str, seed: int = 0) -> "CTDESavedPolicy":
        """Load a saved actor checkpoint and return a lightweight policy wrapper."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if "actor_params" not in payload:
            raise ValueError("Checkpoint does not contain actor parameters")
        actor_params = payload["actor_params"]
        return CTDESavedPolicy(actor_params=actor_params, seed=seed)

class CTDESavedPolicy:
    """Lightweight actor that performs stochastic action selection from saved CTDE weights."""

    def __init__(self, actor_params: Sequence[Dict[str, np.ndarray]], seed: int = 0):
        if not actor_params:
            raise ValueError("Actor parameters are empty")
        self.layers: List[Dict[str, np.ndarray]] = []
        for layer in actor_params:
            if "w" not in layer or "b" not in layer:
                raise ValueError("Malformed layer encountered in actor parameters")
            self.layers.append({
                "w": np.asarray(layer["w"], dtype=np.float32),
                "b": np.asarray(layer["b"], dtype=np.float32),
            })
        self.obs_dim = int(self.layers[0]["w"].shape[0])
        self.action_dim = int(self.layers[-1]["w"].shape[1])
        self.rng = np.random.default_rng(seed)

    def _forward(self, obs_flat: np.ndarray) -> np.ndarray:
        h = obs_flat
        for idx, layer in enumerate(self.layers):
            w = layer["w"]
            b = layer["b"]
            h = h @ w + b
            if idx < len(self.layers) - 1:
                h = np.maximum(h, 0.0, out=h)
        return h

    def _mask_from_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        try:
            planes = obs_arr.reshape(-1, self.action_dim)
            row0 = planes[0]
        except (ValueError, IndexError):
            row0 = obs_arr[:self.action_dim]
        mask = np.zeros(self.action_dim, dtype=np.float32)
        limit = min(self.action_dim, row0.size)
        if limit > 0:
            mask[:limit] = row0[:limit]
        return (mask > 0.0).astype(np.float32)

    def act_with_mask(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        mask_arr = np.asarray(mask, dtype=np.float32).reshape(-1)
        if mask_arr.size != self.action_dim:
            mask_arr = self._mask_from_obs(obs)
        legal_idx = np.flatnonzero(mask_arr > 0.5)
        if legal_idx.size == 0:
            legal_idx = np.arange(self.action_dim, dtype=np.int32)
        logits = self._forward(obs_flat)
        legal_logits = logits[legal_idx].astype(np.float64)
        legal_logits -= float(np.max(legal_logits))
        exp_logits = np.exp(legal_logits)
        denom = float(exp_logits.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            return int(self.rng.choice(legal_idx))
        probs = exp_logits / denom
        probs = np.clip(probs, 0.0, 1.0)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(self.rng.choice(legal_idx))
        probs /= total
        return int(self.rng.choice(legal_idx, p=probs))

    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:
        mask = self._mask_from_obs(obs)
        return self.act_with_mask(seat, obs, mask)

