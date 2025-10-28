"""
Deep CFR External Sampling Traverser (one-node update with fast rollouts)
for 4-player imperfect-information games (2v2 Scopa).

This version avoids exponential recursion at target nodes by:
- Evaluating each legal action via a single rollout-to-terminal, sampling
  all players from current PolicyNets.
- Computing instantaneous regrets only at the current infoset (one-node update).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch

from src.deep_cfr.buffers import PolicyMemory, RegretMemory
from src.deep_cfr.nets import FlexibleNet, masked_softmax, positive_regret_policy


class ExternalSamplingTraverser:
    """
    Deep CFR external-sampling traverser:
      - target_seat: seat whose regrets we update in this traversal
      - others sample actions from their current PolicyNets
      - we store:
          * Regret samples for target_seat (one-node updates)
          * Policy (average strategy) samples for ANY acting seat (with weight=1.0)

    Assumptions:
      - env.observe(agent) -> (4, 40)
      - env.get_action_mask(agent) -> (40,)
      - copy.deepcopy(env) is efficient (ScopaGame.__deepcopy__ provided)

    :param regret_nets: list of RegretNets for each seat
    :param policy_nets: list of PolicyNets for each seat
    :param regret_mems: list of RegretMemories for each seat
    :param policy_mems: list of PolicyMemories for each seat
    :param device: torch device for inference
    :param strict_illegal: if True, raise error on illegal actions; else resample
    :param rng: optional np.random.RandomState for reproducibility
    """

    def __init__(
        self,
        regret_nets: List[FlexibleNet],
        policy_nets: List[FlexibleNet],
        regret_mems: List[RegretMemory],
        policy_mems: List[PolicyMemory],
        device: str = "cpu",
        strict_illegal: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.regret_nets = regret_nets
        self.policy_nets = policy_nets
        self.regret_mems = regret_mems
        self.policy_mems = policy_mems
        self.device = device
        self.strict_illegal = strict_illegal
        self.rng = rng or np.random.RandomState(123)
        # used to map terminal utility during rollouts
        self._rollout_target_seat: Optional[int] = None

    # ---------- helpers ----------
    @staticmethod
    def _seat_from_agent(agent_name: str) -> int:
        """
        Extract seat index from agent name string.

        :param agent_name: string like "player_0", "player_1", etc.
        :return: seat index as integer
        """
        return int(agent_name.split("_")[-1])

    @staticmethod
    def _flatten_obs(obs: np.ndarray) -> np.ndarray:
        """
        Flatten the observation array.

        :param obs: observation array of shape (4, 40)
        :return: flattened observation array of shape (160,)
        """
        # obs: (4,40) -> (160,)
        return obs.astype(np.float32).reshape(-1)

    @staticmethod
    def _legal_mask(env, agent: str) -> np.ndarray:
        """
        Get the legal action mask for the given agent.

        :param env: game environment
        :param agent: agent name string
        :return: legal action mask as a float32 numpy array
        """
        return env.get_action_mask(agent).astype(np.float32)

    @staticmethod
    def _team_index(seat: int) -> int:
        """
        Map seat index to team index.

        :param seat: seat index (0-3)
        :return: team index (0 or 1)
        """
        # seats 0,2 -> team 0 ; seats 1,3 -> team 1
        return seat % 2

    @staticmethod
    def _is_terminal(env) -> bool:
        """
        Check if the game has reached a terminal state.

        :param env: game environment
        :return: True if terminal, False otherwise
        """
        return all(len(p.hand) == 0 for p in env.game.players)

    def _terminal_utility_for_seat(self, env, seat: int) -> float:
        """
        Get terminal utility for the given seat's team.

        :param env: game environment
        :param seat: seat index (0-3)
        :return: terminal utility for the seat's team
        """
        # Evaluate final points and map to seat utility
        team_scores = env.game.evaluate_round()
        return float(team_scores[self._team_index(seat)])

    # ---------- fast rollout ----------
    def _rollout_to_terminal(self, env) -> float:
        """
        From current env, sample ALL players from PolicyNets until terminal.
        Returns terminal utility for the ORIGINAL target seat's team.

        :param env: game environment
        :return: terminal utility for the target seat's team
        """
        assert self._rollout_target_seat is not None, "rollout target seat not set"
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            seat = self._seat_from_agent(agent)
            mask = self._legal_mask(env, agent)
            obs_flat = self._flatten_obs(env.observe(agent))
            # sample from current policy
            with torch.no_grad():
                x = torch.from_numpy(obs_flat[None, :]).float().to(self.device)
                logits = self.policy_nets[seat](x).squeeze(0)
                m = torch.from_numpy(mask[None, :]).float().to(self.device)
                probs = masked_softmax(logits[None, :], m).squeeze(0).cpu().numpy()
            legal = np.nonzero(mask)[0]
            p = probs[legal]
            if p.sum() <= 0 or not np.isfinite(p).all():
                a = int(np.random.choice(legal))
            else:
                a = int(np.random.choice(legal, p=p / p.sum()))
            env.step(a)

        team_scores = env.game.evaluate_round()
        return float(team_scores[self._team_index(self._rollout_target_seat)])

    # ---------- core traversal ----------
    def traverse(self, env, target_seat: int) -> float:
        """
        Returns the value for target_seat from the current node
        while adding regret/policy samples along the path.

        :param env: game environment
        :param target_seat: seat index whose regrets we update
        :return: node value for target_seat
        """
        if self._is_terminal(env):
            return self._terminal_utility_for_seat(env, target_seat)

        agent = env.agent_selection
        acting_seat = self._seat_from_agent(agent)
        obs = env.observe(agent)  # (4,40)
        mask = self._legal_mask(env, agent)  # (40,)
        obs_flat = self._flatten_obs(obs)  # (160,)

        # --- store average-policy sample for the acting seat ---
        with torch.no_grad():
            x = torch.from_numpy(obs_flat[None, :]).to(self.device)
            logits = self.policy_nets[acting_seat](x).squeeze(0)
            probs = masked_softmax(
                logits[None, :],
                torch.from_numpy(mask[None, :]).to(self.device),
            ).squeeze(0)
            policy_probs_np = probs.cpu().numpy()
        # weight=1.0 (you can replace with opponent-reach weight later if you track it)
        self.policy_mems[acting_seat].add(
            info_np=obs_flat, mask_np=mask, probs_np=policy_probs_np, weight=1.0
        )

        if acting_seat == target_seat:
            # -------- ONE-NODE UPDATE with rollouts --------
            legal = np.nonzero(mask)[0].tolist()
            if not legal:
                return self._terminal_utility_for_seat(env, target_seat)

            # baseline σ from positive regrets predicted by RegretNet[target]
            with torch.no_grad():
                adv_pred = self.regret_nets[target_seat](
                    torch.from_numpy(obs_flat[None, :]).to(self.device)
                ).squeeze(0)
                adv_sigma = (
                    positive_regret_policy(
                        adv_pred[None, :],
                        torch.from_numpy(mask[None, :]).to(self.device),
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            # rollout value for each legal action
            action_values: Dict[int, float] = {}
            self._rollout_target_seat = target_seat
            for a in legal:
                child = copy.deepcopy(env)
                child.step(int(a))
                v_a = self._rollout_to_terminal(child)
                action_values[a] = v_a

            # baseline v̄ = Σ σ(a) v(a)
            baseline = sum(float(adv_sigma[a]) * action_values[a] for a in legal)

            # instantaneous regrets at THIS infoset only
            advantages = np.zeros_like(mask, dtype=np.float32)
            for a in legal:
                advantages[a] = action_values[a] - baseline

            # store one regret sample for this infoset
            self.regret_mems[target_seat].add(
                info_np=obs_flat, mask_np=mask, adv_np=advantages
            )

            # return node value to parent
            return float(baseline)

        else:
            # -------- Non-target player: sample ONE action from PolicyNet and recurse --------
            with torch.no_grad():
                x = torch.from_numpy(obs_flat[None, :]).to(self.device)
                logits = self.policy_nets[acting_seat](x).squeeze(0)
                probs = (
                    masked_softmax(
                        logits[None, :],
                        torch.from_numpy(mask[None, :]).to(self.device),
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            legal = np.nonzero(mask)[0]
            p = probs[legal]
            if p.sum() <= 0 or not np.isfinite(p).all():
                a = int(np.random.choice(legal))
            else:
                a = int(np.random.choice(legal, p=p / p.sum()))
            child = copy.deepcopy(env)
            child.step(a)
            return self.traverse(child, target_seat)
