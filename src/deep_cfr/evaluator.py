"""
Deep CFR evaluators (n-player, PettingZoo-compatible).

- Index players via env.possible_agents.index(env.agent_selection)
- Generic obs flatten: (H,W,...) -> 1D
- Legal mask from infos['action_mask'] (fallback to env.get_action_mask)
- No dependency on agent name formats like "player_0"
- Uses terminal round score if the env exposes it; otherwise sums cumulative rewards.

APIs:
- evaluate_selfplay(policy_nets, make_env, ...)
- evaluate_vs_random(policy_nets, make_env, ..., allow_selfplay_equivalence=False)
- evaluate_vs_random_one_net(policy_net, make_env, ..., seat_average=True)
"""

from __future__ import annotations
from typing import Callable, List, Tuple, Optional, Any
import numpy as np
import torch

from src.deep_cfr.nets import FlexibleNet, masked_softmax


# ---------------- helpers ----------------


def _flatten_obs(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _legal_mask(env, agent: str) -> np.ndarray:
    m = None
    try:
        m = env.infos.get(agent, {}).get("action_mask", None)
    except Exception:
        m = None
    if m is None:
        # fallbacks (current-agent variants)
        try:
            m = env.get_action_mask(agent)
        except Exception:
            try:
                m = env.get_action_mask()
            except Exception:
                pass
    if m is None:
        raise RuntimeError("No legal action mask available from env.")
    return np.asarray(m, dtype=np.float32)


def _sample_from_policy(
    net: FlexibleNet,
    device: str,
    obs_flat: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    with torch.no_grad():
        x = torch.from_numpy(obs_flat[None, :]).float().to(device)
        logits = net(x).squeeze(0)
        m = torch.from_numpy(mask[None, :]).float().to(device)
        probs = masked_softmax(logits[None, :], m).squeeze(0).cpu().numpy()
    legal = np.nonzero(mask)[0]
    if legal.size == 0:
        return 0  # dead-step; env will handle it (_was_dead_step)
    p = probs[legal]
    if (p.sum() <= 0) or (not np.isfinite(p).all()):
        return int(rng.choice(legal))
    return int(rng.choice(legal, p=p / p.sum()))


def _sample_random(mask: np.ndarray, rng: np.random.RandomState) -> int:
    legal = np.nonzero(mask)[0]
    if legal.size == 0:
        return 0  # dead-step
    return int(rng.choice(legal))


# ---------- scoring helpers ----------


def _accumulate_rewards(env, cum: dict) -> None:
    """Accumulate step rewards (PettingZoo-style)."""
    if hasattr(env, "rewards") and isinstance(env.rewards, dict):
        for a_name, r in env.rewards.items():
            cum[a_name] += float(r)


def _final_team_scores(env) -> Optional[Tuple[float, float]]:
    """
    Try to read terminal *round* scores from the env, if available.
    Returns (evens_score, odds_score) or None if not available.
    """
    # 1) Scopa-like envs: roundScores() -> list[(team_even, team_odd), ...]
    try:
        if hasattr(env, "roundScores") and callable(env.roundScores):
            scores_hist = env.roundScores()
            if scores_hist:
                last = scores_hist[-1]
                if isinstance(last, (tuple, list)) and len(last) == 2:
                    return float(last[0]), float(last[1])
    except Exception:
        pass

    # 2) Raw game hook: game.evaluate_round() -> [team0, team1]
    try:
        if hasattr(env, "game") and hasattr(env.game, "evaluate_round"):
            v = env.game.evaluate_round()
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return float(v[0]), float(v[1])
    except Exception:
        pass

    # 3) No terminal round score available
    return None


def _team_scores_from_cum(env, cum: dict) -> Tuple[float, float]:
    """
    Team up agents as (even seats) vs (odd seats) and sum cumulative rewards.
    This is used only when _final_team_scores() is unavailable.
    """
    n_agents = len(env.possible_agents)
    evens = sum(cum[env.possible_agents[i]] for i in range(0, n_agents, 2))
    odds = sum(cum[env.possible_agents[i]] for i in range(1, n_agents, 2))
    return float(evens), float(odds)


# ---------- evaluators ----------


def evaluate_selfplay(
    policy_nets: List["FlexibleNet"],
    make_env: Callable[[], Any],
    n_games: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
) -> Tuple[float, float]:
    """
    Self-play with one net per agent (len(nets) must equal num_agents).
    Returns (winrate_evens, avg_score_diff = evens - odds),
    where evens = seats 0,2,... and odds = 1,3,...
    """
    rng = np.random.RandomState(seed_offset + 999)
    wins = 0.0
    diffs: List[float] = []

    for g in range(n_games):
        env = make_env()
        env.reset(seed=seed_offset + g)

        n_agents = len(env.possible_agents)
        if len(policy_nets) != n_agents:
            raise ValueError(
                f"Self-play expects one net per agent: nets={len(policy_nets)}, agents={n_agents}"
            )

        cum = {a: 0.0 for a in env.possible_agents}

        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            idx = env.possible_agents.index(agent)
            obs_flat = _flatten_obs(env.observe(agent))
            mask = _legal_mask(env, agent)
            a = _sample_from_policy(policy_nets[idx], device, obs_flat, mask, rng)
            env.step(a)
            _accumulate_rewards(env, cum)

        # Prefer terminal round score if available
        term = _final_team_scores(env)
        if term is not None:
            evens, odds = term
        else:
            evens, odds = _team_scores_from_cum(env, cum)

        diffs.append(evens - odds)
        wins += 1.0 if (evens - odds) > 0 else (0.5 if (evens - odds) == 0 else 0.0)

    winrate = wins / max(1, n_games)
    score_diff = float(np.mean(diffs)) if diffs else 0.0
    return winrate, score_diff


def evaluate_vs_random(
    policy_nets: List["FlexibleNet"],
    make_env: Callable[[], Any],
    n_games: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
    allow_selfplay_equivalence: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate vs random opponents. Seat mapping rules:

    - If len(nets) == num_agents: everyone uses their net (degenerates to self-play).
      This raises unless allow_selfplay_equivalence=True.
    - If len(nets) == 1: net controls seat 0, others random. (For 1v1 experiments.)
    - If len(nets) == ceil(num_agents/2): nets control EVEN seats (0,2,...) in order; ODDS are random.
      (Use this for 2v2 with nets for seats 0 and 2.)

    Returns (winrate_evens, avg_score_diff = evens - odds).
    """
    rng = np.random.RandomState(seed_offset + 12345)
    wins = 0.0
    diffs: List[float] = []

    for g in range(n_games):
        env = make_env()
        env.reset(seed=seed_offset + 10_000 + g)

        n_agents = len(env.possible_agents)
        n_nets = len(policy_nets)

        if n_nets == n_agents and not allow_selfplay_equivalence:
            raise ValueError(
                "evaluate_vs_random received one net per agent; this degenerates to self-play.\n"
                "Either pass a single net (seat 0) for 1v1, half the agents (even seats) for 2v2, "
                "or set allow_selfplay_equivalence=True if you really want this."
            )

        # agent_index -> net_index (int) or None (random)
        control: dict[int, Optional[int]] = {}

        if n_nets == n_agents:
            control = {i: i for i in range(n_agents)}
        elif n_nets == 1:
            control[0] = 0
            for i in range(1, n_agents):
                control[i] = None
        elif n_nets == (n_agents + 1) // 2:
            for i in range(n_agents):
                if i % 2 == 0:
                    k = i // 2
                    control[i] = k if k < n_nets else None
                else:
                    control[i] = None
        else:
            raise ValueError(
                f"Unsupported evaluate_vs_random mapping: nets={n_nets}, agents={n_agents}. "
                f"Provide 1 net (seat 0), half the agents (even seats), or all agents."
            )

        cum = {a: 0.0 for a in env.possible_agents}

        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            idx = env.possible_agents.index(agent)
            obs_flat = _flatten_obs(env.observe(agent))
            mask = _legal_mask(env, agent)

            if control.get(idx, None) is None:
                a = _sample_random(mask, rng)
            else:
                net = policy_nets[control[idx]]
                a = _sample_from_policy(net, device, obs_flat, mask, rng)

            env.step(a)
            _accumulate_rewards(env, cum)

        # Prefer terminal round score if available
        term = _final_team_scores(env)
        if term is not None:
            evens, odds = term
        else:
            evens, odds = _team_scores_from_cum(env, cum)

        diffs.append(evens - odds)
        wins += 1.0 if (evens - odds) > 0 else (0.5 if (evens - odds) == 0 else 0.0)

    winrate = wins / max(1, n_games)
    score_diff = float(np.mean(diffs)) if diffs else 0.0
    return winrate, score_diff


def evaluate_vs_random_one_net(
    policy_net: "FlexibleNet",
    make_env: Callable[[], Any],
    n_games: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
    seat_average: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate a single net vs random opponents.

    - If seat_average=True and the game is 1v1, we run half the games with the net in seat 0,
      half in seat 1, then average the two results. This removes seat bias.
    - For n_agents > 2, the net always controls seat 0 (even team).

    Returns (winrate_evens, avg_score_diff = evens - odds).
    """
    rng = np.random.RandomState(seed_offset + 54321)

    def _run_with_seat0(g0: int) -> Tuple[float, float]:
        wins = 0.0
        diffs: List[float] = []
        for g in range(g0, g0 + n_games):
            env = make_env()
            env.reset(seed=seed_offset + 20_000 + g)

            n_agents = len(env.possible_agents)
            control: dict[int, Optional[int]] = {}
            # seat 0 is controlled by net; others random
            control[0] = 0
            for i in range(1, n_agents):
                control[i] = None

            cum = {a: 0.0 for a in env.possible_agents}

            while not all(env.terminations.values()) and not all(
                env.truncations.values()
            ):
                agent = env.agent_selection
                idx = env.possible_agents.index(agent)
                obs_flat = _flatten_obs(env.observe(agent))
                mask = _legal_mask(env, agent)

                if control.get(idx, None) is None:
                    a = _sample_random(mask, rng)
                else:
                    a = _sample_from_policy(policy_net, device, obs_flat, mask, rng)

                env.step(a)
                _accumulate_rewards(env, cum)

            term = _final_team_scores(env)
            if term is not None:
                evens, odds = term
            else:
                evens, odds = _team_scores_from_cum(env, cum)

            diffs.append(evens - odds)
            wins += 1.0 if (evens - odds) > 0 else (0.5 if (evens - odds) == 0 else 0.0)

        return wins / max(1, n_games), float(np.mean(diffs)) if diffs else 0.0

    # If not seat averaging or not 1v1, just run seat0 mode.
    env_tmp = make_env()
    if (not seat_average) or (len(env_tmp.possible_agents) != 2):
        return _run_with_seat0(0)

    # 1v1 + seat averaging: run half games as seat 0, half mirrored as seat 1
    wr0, sd0 = _run_with_seat0(0)

    # Mirror by swapping seat assignment: to simulate seat 1 control, we’ll
    # wrap the action selection so that when idx==1 we use the net, else random.
    def _run_with_seat1(g0: int) -> Tuple[float, float]:
        wins = 0.0
        diffs: List[float] = []
        for g in range(g0, g0 + n_games):
            env = make_env()
            env.reset(seed=seed_offset + 30_000 + g)

            cum = {a: 0.0 for a in env.possible_agents}

            while not all(env.terminations.values()) and not all(
                env.truncations.values()
            ):
                agent = env.agent_selection
                idx = env.possible_agents.index(agent)
                obs_flat = _flatten_obs(env.observe(agent))
                mask = _legal_mask(env, agent)

                if idx == 1:  # seat 1 controlled by our net
                    a = _sample_from_policy(policy_net, device, obs_flat, mask, rng)
                else:
                    a = _sample_random(mask, rng)

                env.step(a)
                _accumulate_rewards(env, cum)

            term = _final_team_scores(env)
            if term is not None:
                evens, odds = term
            else:
                evens, odds = _team_scores_from_cum(env, cum)

            diffs.append(evens - odds)
            wins += 1.0 if (evens - odds) > 0 else (0.5 if (evens - odds) == 0 else 0.0)

        return wins / max(1, n_games), float(np.mean(diffs)) if diffs else 0.0

    wr1, sd1 = _run_with_seat1(10_000)
    # Average both seat roles
    return (wr0 + wr1) / 2.0, (sd0 + sd1) / 2.0
