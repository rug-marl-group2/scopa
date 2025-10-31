"""
Deep CFR External Sampling Traverser (one-node update with fast rollouts)
for n-player imperfect-information games.

Version: fixed baseline, regret-matching target, and reach-weighted policy samples.
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
    External-sampling Deep CFR traverser (PettingZoo AEC compatible).

    Key points:
    - Baseline at target infoset → POLICY net (current strategy)
    - Policy targets stored → REGRET-MATCHING policy from regret net
    - Policy samples weighted by product of other players' reach probabilities
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

        self._rollout_target_player: Optional[int] = None
        n = len(self.regret_nets)
        assert n >= 2, "Need at least 2 players."
        assert len(self.policy_nets) == n
        assert len(self.regret_mems) == n
        assert len(self.policy_mems) == n

    # ---------------- utils ----------------
    @staticmethod
    def _flatten_obs(obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32).reshape(-1)

    @staticmethod
    def _legal_mask(env, agent: str) -> np.ndarray:
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
                    m = env.get_action_mask()
                except Exception:
                    pass
        if m is None:
            raise RuntimeError("No legal action mask available from env.")
        return np.asarray(m, dtype=np.float32)

    @staticmethod
    def _is_terminal(env) -> bool:
        if (
            hasattr(env, "terminations")
            and isinstance(env.terminations, dict)
            and env.terminations
        ):
            term_over = all(env.terminations.get(a, False) for a in env.agents)
            trunc_over = all(env.truncations.get(a, False) for a in env.agents)
            return term_over or trunc_over
        if hasattr(env, "game") and hasattr(env.game, "players"):
            try:
                return all(len(p.hand) == 0 for p in env.game.players)
            except Exception:
                pass
        return False

    @staticmethod
    def _terminal_utility_for_player(env, target_player: int) -> float:
        if hasattr(env, "rewards") and isinstance(env.rewards, dict):
            agent_name = env.possible_agents[target_player]
            return float(env.rewards.get(agent_name, 0.0))
        return 0.0

    # ---------------- rollout ----------------
    def _rollout_to_terminal(self, env) -> float:
        assert self._rollout_target_player is not None, "rollout target player not set"
        while not all(env.terminations.values()) and not all(env.truncations.values()):
            agent = env.agent_selection
            player_idx = env.possible_agents.index(agent)
            mask = self._legal_mask(env, agent)
            obs_flat = self._flatten_obs(env.observe(agent))
            with torch.no_grad():
                x = torch.from_numpy(obs_flat[None, :]).float().to(self.device)
                logits = self.policy_nets[player_idx](x).squeeze(0)
                m = torch.from_numpy(mask[None, :]).float().to(self.device)
                probs = masked_softmax(logits[None, :], m).squeeze(0).cpu().numpy()
            legal = np.nonzero(mask)[0]
            if legal.size == 0:
                env.step(0)
                continue
            p = probs[legal]
            if p.sum() <= 0 or not np.isfinite(p).all():
                a = int(self.rng.choice(legal))
            else:
                a = int(self.rng.choice(legal, p=p / p.sum()))
            env.step(a)
        return self._terminal_utility_for_player(env, self._rollout_target_player)

    def _mul_reach(self, reach: np.ndarray, player_idx: int, p: float) -> np.ndarray:
        r = reach.copy()
        r[player_idx] *= float(max(p, 1e-12))
        return r

    # ---------------- core traversal ----------------
    def traverse(self, env, target_seat: int, reach: np.ndarray) -> float:
        if self._is_terminal(env):
            return self._terminal_utility_for_player(env, target_seat)

        agent = env.agent_selection
        acting_player = env.possible_agents.index(agent)
        obs = env.observe(agent)
        mask = self._legal_mask(env, agent)
        obs_flat = self._flatten_obs(obs)

        # --- σ^+ from regret net of the ACTING player (for baseline / target nodes)
        with torch.no_grad():
            x = torch.from_numpy(obs_flat[None, :]).float().to(self.device)
            adv_pred = self.regret_nets[acting_player](x).squeeze(0)
            sigma_pos = (
                positive_regret_policy(
                    adv_pred[None, :],
                    torch.from_numpy(mask[None, :]).float().to(self.device),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        # --- NEW: opponent reach weight for policy sample at this infoset
        # w_i(s) = ∏_{j != i} reach[j]
        opp_reach_weight = float(np.prod(reach[np.arange(len(reach)) != acting_player]))

        # store policy sample for the acting player, with weight = opponent reach
        self.policy_mems[acting_player].add(
            info_np=self._flatten_obs(obs),
            mask_np=mask,
            probs_np=sigma_pos,
            weight=opp_reach_weight,
        )

        if acting_player == target_seat:
            legal = np.nonzero(mask)[0].tolist()
            if not legal:
                return self._terminal_utility_for_player(env, target_seat)

            # baseline using σ^+ of the target at this infoset
            action_values = {}
            self._rollout_target_player = target_seat
            for a in legal:
                child = copy.deepcopy(env)
                child.step(int(a))
                # For rollouts below, we propagate REACH for the target's action as well
                # using σ^+ probability (this matches sampling policy for reach accounting).
                p_a = float(sigma_pos[a])
                v_a = self._rollout_to_terminal(
                    child
                )  # rollout uses policy_nets for others
                action_values[a] = v_a
            self._rollout_target_player = None

            baseline = float(sum(sigma_pos[a] * action_values[a] for a in legal))

            advantages = np.zeros_like(mask, dtype=np.float32)
            for a in legal:
                advantages[a] = action_values[a] - baseline

            self.regret_mems[target_seat].add(
                info_np=obs_flat, mask_np=mask, adv_np=advantages
            )
            return baseline

        # --- Non-target node: sample from BEHAVIOR policy (policy_nets), update reach, recurse
        with torch.no_grad():
            logits = self.policy_nets[acting_player](
                torch.from_numpy(obs_flat[None, :]).float().to(self.device)
            ).squeeze(0)
            behav_probs = (
                masked_softmax(
                    logits[None, :],
                    torch.from_numpy(mask[None, :]).float().to(self.device),
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        legal = np.nonzero(mask)[0]
        if legal.size == 0:
            child = copy.deepcopy(env)
            child.step(0)
            return self.traverse(child, target_seat, reach)

        p = behav_probs[legal]
        if p.sum() <= 0 or not np.isfinite(p).all():
            a_idx = int(self.rng.choice(len(legal)))
        else:
            a_idx = int(self.rng.choice(len(legal), p=p / p.sum()))
        a = int(legal[a_idx])
        p_sel = float(max(behav_probs[a], 1e-12))

        child = copy.deepcopy(env)
        child.step(a)
        # --- NEW: update reach for the acting player
        new_reach = self._mul_reach(reach, acting_player, p_sel)
        return self.traverse(child, target_seat, new_reach)
