"""
Deep CFR evaluation functions for 4-player imperfect-information games.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch

from deep_cfr.nets import FlexibleNet, masked_softmax


def _sample_from_policynet(
    net: FlexibleNet, obs_flat: np.ndarray, mask: np.ndarray, device: str
) -> int:
    """
    Sample an action from the policy network given flattened observation and legal action mask.

    :param net: FlexibleNet policy network
    :param obs_flat: np.ndarray flattened observation (in_dim,)
    :param mask: np.ndarray legal action mask (A,)
    :param device: torch device for inference
    :return: sampled action (int)
    """
    with torch.no_grad():
        x = torch.from_numpy(obs_flat[None, :]).float().to(device)
        logits = net(x).squeeze(0)
        m = torch.from_numpy(mask[None, :]).float().to(device)
        probs = masked_softmax(logits[None, :], m).squeeze(0).cpu().numpy()
    legal = np.nonzero(mask)[0]
    p = probs[legal]
    if not np.isfinite(p).all() or p.sum() <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(legal, p=p))


def _sample_random(mask: np.ndarray) -> int:
    """
    Sample a random legal action given the legal action mask.
    :param mask: np.ndarray legal action mask (A,)
    :return: sampled action (int)
    """
    legal = np.nonzero(mask)[0]
    return int(np.random.choice(legal)) if legal.size else 0


def _observe_flat(env, agent: str) -> np.ndarray:
    """
    Get flattened observation for the given agent from the environment.
    :param env: game environment
    :param agent: agent identifier string
    :return: flattened observation as np.ndarray
    """
    obs = env.observe(agent)  # (4,40)
    return obs.astype(np.float32).reshape(-1)


def evaluate_vs_random(
    policy_nets: List[FlexibleNet],
    make_env: Callable[[], any],
    n_games: int = 200,
    device: str = "cpu",
    seed_offset: int = 0,
) -> Tuple[float, float]:
    """
    Evaluate PolicyNet seats {0,2} vs Random seats {1,3}.

    :param policy_nets: List of FlexibleNet policy networks for each seat.
    :param make_env: Callable that creates a new game environment instance.
    :param n_games: Number of games to play for evaluation.
    :param device: Torch device for inference.
    :param seed_offset: Seed offset for environment randomness.
    :return: Tuple (winrate_team02, avg_score_diff)
    """
    wins = 0
    total_diff = 0.0
    for g in range(n_games):
        env = make_env()
        env.reset(seed=seed_offset + g)
        # Play until termination/truncation
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            seat = int(agent.split("_")[-1])
            mask = env.get_action_mask(agent).astype(np.float32)
            obs_flat = _observe_flat(env, agent)
            if seat in (0, 2):  # our team plays learned policy
                a = _sample_from_policynet(policy_nets[seat], obs_flat, mask, device)
            else:  # opponents random
                a = _sample_random(mask)
            env.step(a)
        # Scoring
        round_scores = env.roundScores()[-1] if env.roundScores() else (0, 0)
        # round_scores = (team02_outcome, team13_outcome) in {-1,0,1}
        wins += 1 if round_scores[0] > round_scores[1] else 0
        total_diff += round_scores[0] - round_scores[1]
    winrate = wins / max(1, n_games)
    avg_diff = total_diff / max(1, n_games)
    return winrate, avg_diff


def evaluate_selfplay(
    policy_nets: List[FlexibleNet],
    make_env: Callable[[], any],
    n_games: int = 200,
    device: str = "cpu",
    seed_offset: int = 10_000,
) -> Tuple[float, float]:
    """
    Self-play: PolicyNet seats {0,2} vs PolicyNet seats {1,3}.

    :param policy_nets: List of FlexibleNet policy networks for each seat.
    :param make_env: Callable that creates a new game environment instance.
    :param n_games: Number of games to play for evaluation.
    :param device: Torch device for inference.
    :param seed_offset: Seed offset for environment randomness.
    :return: Tuple (winrate_team02, avg_score_diff)
    """
    wins = 0
    total_diff = 0.0
    for g in range(n_games):
        env = make_env()
        env.reset(seed=seed_offset + g)
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            seat = int(agent.split("_")[-1])
            mask = env.get_action_mask(agent).astype(np.float32)
            obs_flat = _observe_flat(env, agent)
            a = _sample_from_policynet(policy_nets[seat], obs_flat, mask, device)
            env.step(a)
        round_scores = env.roundScores()[-1] if env.roundScores() else (0, 0)
        wins += 1 if round_scores[0] > round_scores[1] else 0
        total_diff += round_scores[0] - round_scores[1]
    return wins / max(1, n_games), total_diff / max(1, n_games)
