"""
Deep CFR External Sampling Traverser (one-node update with fast rollouts)
for n-player imperfect-information games.

Key ideas:
- Use REGRET-MATCHING (from regret nets) as the current strategy σ for:
  * sampling non-target players' actions,
  * sampling during rollouts,
  * producing average-policy targets stored in PolicyMemory.
- Policy nets are *not* used inside traversal; they are trained offline to
  imitate σ snapshots accumulated in PolicyMemory (Deep CFR standard design).

This avoids the self-copy loop that happens when targets are taken from the
policy nets themselves.
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
    Deep CFR external-sampling traverser with one-node updates.

    During a traversal for target_player:
      - If acting_player == target_player:
          • Evaluate each legal action via a single rollout to terminal where
            all players follow regret-matching σ from regret nets.
          • Compute instantaneous regrets only at this infoset and push ONE
            sample to RegretMemory[target_player].
      - Else (acting_player != target_player):
          • Sample ONE legal action from σ (regret-matching) and recurse.
      - For every acting player (including target), push an average-policy
        sample (σ) with weight=1.0 to PolicyMemory[acting_player].

    Assumptions:
      - env is PettingZoo AEC-like (agent_selection, possible_agents, observe(),
        infos['action_mask'] OR get_action_mask, terminations, truncations, rewards).
      - regret_nets / policy_nets / memories are length n_players and aligned
        with env.possible_agents order.
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

        # set/used only during rollout()
        self._rollout_target_player: Optional[int] = None

        # integrity checks
        n = len(self.regret_nets)
        assert n >= 2, "Need at least 2 players."
        assert len(self.policy_nets) == n
        assert len(self.regret_mems) == n
        assert len(self.policy_mems) == n

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _flatten_obs(obs: np.ndarray) -> np.ndarray:
        """Flatten any (H, W) observation to 1D float32."""
        return obs.astype(np.float32).reshape(-1)

    @staticmethod
    def _legal_mask(env, agent: str) -> np.ndarray:
        """Get legal action mask (float32). Prefer infos['action_mask'], else env.get_action_mask()."""
        m = None
        try:
            m = env.infos.get(agent, {}).get("action_mask", None)
        except Exception:
            m = None
        if m is None:
            try:
                m = env.get_action_mask(agent)
            except Exception:
                try:
                    m = env.get_action_mask()  # some envs expose current-only
                except Exception:
                    pass
        if m is None:
            raise RuntimeError("No legal action mask available from env.")
        m = np.asarray(m, dtype=np.float32)
        return m

    @staticmethod
    def _is_terminal(env) -> bool:
        """Prefer PettingZoo termination flags; fall back to raw game check if present."""
        # PettingZoo: terminal if all agents are terminated or truncated
        if (
            hasattr(env, "terminations")
            and isinstance(env.terminations, dict)
            and env.terminations
        ):
            term_over = all(env.terminations.get(a, False) for a in env.agents)
            trunc_over = all(env.truncations.get(a, False) for a in env.agents)
            return term_over or trunc_over
        # Fallback: if env exposes raw game with players/hands
        if hasattr(env, "game") and hasattr(env.game, "players"):
            try:
                return all(len(p.hand) == 0 for p in env.game.players)
            except Exception:
                pass
        return False

    @staticmethod
    def _terminal_utility_for_player(env, target_player: int) -> float:
        """
        Utility mapping at terminal via PettingZoo rewards.
        Assumes env.possible_agents order aligns with net/memory lists.
        """
        if hasattr(env, "rewards") and isinstance(env.rewards, dict):
            agent_name = env.possible_agents[target_player]
            return float(env.rewards.get(agent_name, 0.0))
        return 0.0

    # ---------------------------------------------------------------------
    # Rollout (ALL players sample from regret-matching σ)
    # ---------------------------------------------------------------------

    def _sigma_from_regret(self, player_idx: int, obs_flat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute regret-matching policy σ for a given player, observation, and mask."""
        with torch.no_grad():
            x = torch.from_numpy(obs_flat[None, :]).float().to(self.device)
            adv_pred = self.regret_nets[player_idx](x).squeeze(0)
            probs = positive_regret_policy(
                adv_pred[None, :],
                torch.from_numpy(mask[None, :]).float().to(self.device),
            ).squeeze(0).cpu().numpy()
        return probs

    def _rollout_to_terminal(self, env) -> float:
        """
        From current env, sample ALL players using regret-matching σ
        until terminal, then return terminal utility for self._rollout_target_player.
        """
        assert self._rollout_target_player is not None, "rollout target player not set"
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            player_idx = env.possible_agents.index(agent)
            mask = self._legal_mask(env, agent)
            obs_flat = self._flatten_obs(env.observe(agent))

            legal = np.nonzero(mask)[0]
            if legal.size == 0:
                # advance AEC with a dead step; env should handle it (_was_dead_step)
                env.step(0)
                continue

            probs = self._sigma_from_regret(player_idx, obs_flat, mask)
            p = probs[legal]
            if not np.isfinite(p).all() or p.sum() <= 0:
                a = int(self.rng.choice(legal))
            else:
                a = int(self.rng.choice(legal, p=p / p.sum()))
            env.step(a)
        return self._terminal_utility_for_player(env, self._rollout_target_player)

    # ---------------------------------------------------------------------
    # Core traversal (one-node update)
    # ---------------------------------------------------------------------

    def traverse(self, env, target_seat: int) -> float:
        """
        Return the value for target_seat (player index) from the current node,
        while storing regret/policy samples along the path.

        target_seat must align with env.possible_agents order and nets/memories.
        """
        if self._is_terminal(env):
            return self._terminal_utility_for_player(env, target_seat)

        agent = env.agent_selection
        acting_player = env.possible_agents.index(agent)
        obs = env.observe(agent)  # (H, W, ...) -> flattened
        mask = self._legal_mask(env, agent)  # (A,)
        obs_flat = self._flatten_obs(obs)  # (H*W,)

        # ---- derive σ via REGRET-MATCHING and store avg-policy target
        probs_sigma = self._sigma_from_regret(acting_player, obs_flat, mask)
        self.policy_mems[acting_player].add(
            info_np=obs_flat, mask_np=mask, probs_np=probs_sigma, weight=1.0
        )

        # ---- target player's node: one-node update via rollout per legal action ----
        if acting_player == target_seat:
            legal = np.nonzero(mask)[0].tolist()
            if not legal:
                return self._terminal_utility_for_player(env, target_seat)

            # baseline σ for the TARGET player only
            adv_sigma = self._sigma_from_regret(target_seat, obs_flat, mask)

            # evaluate each legal action by a single rollout
            action_values: Dict[int, float] = {}
            self._rollout_target_player = target_seat
            for a in legal:
                child = copy.deepcopy(env)
                child.step(int(a))
                v_a = self._rollout_to_terminal(child)
                action_values[a] = v_a

            # baseline v̄ = Σ_a σ(a) v(a) over legal actions
            baseline = float(sum(float(adv_sigma[a]) * action_values[a] for a in legal))

            # instantaneous regrets at THIS infoset only
            advantages = np.zeros_like(mask, dtype=np.float32)
            for a in legal:
                advantages[a] = action_values[a] - baseline

            # store one regret sample for this infoset
            self.regret_mems[target_seat].add(
                info_np=obs_flat, mask_np=mask, adv_np=advantages
            )

            return baseline

        # ---- non-target player's node: sample one action from σ and recurse ----
        legal = np.nonzero(mask)[0]
        if legal.size == 0:
            child = copy.deepcopy(env)
            child.step(0)  # dead step to advance the turn
            return self.traverse(child, target_seat)

        probs = probs_sigma  # already computed σ for acting_player
        p = probs[legal]
        if not np.isfinite(p).all() or p.sum() <= 0:
            a = int(self.rng.choice(legal))
        else:
            a = int(self.rng.choice(legal, p=p / p.sum()))
        child = copy.deepcopy(env)
        child.step(a)
        return self.traverse(child, target_seat)
