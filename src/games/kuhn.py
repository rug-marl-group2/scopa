from typing import Optional, Dict, Tuple, List
import numpy as np

from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

# Optional logger (same interface you use elsewhere)
try:
    from src.tlogger import TLogger
except Exception:  # pragma: no cover
    TLogger = None  # type: ignore


def env(tlogger: Optional[TLogger] = None, render_mode=None):
    """
    Factory for a 2-player Kuhn Poker environment (AEC).
    If render_mode == "ansi", wrap with CaptureStdoutWrapper for consistent behavior.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = KuhnPokerEnv(render_mode=internal_render_mode, tlogger=tlogger)
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    return e


class KuhnPokerEnv(AECEnv):
    """
    Minimal Kuhn Poker (2 players, 3-card deck: J<Q<K).
    - Antes: 1 chip each (pot starts at 2).
    - Single betting round; at most one bet/raise total.
    - Payoff: showdown (higher card) or immediate if fold.
    - Zero-sum rewards at terminal only.

    Actions (Discrete(4)) with masks:
      0 = check
      1 = bet
      2 = call
      3 = fold

    Observation (R^6), per-acting player:
      [ cJ, cQ, cK, pot_norm, bet_open, to_call ]
      - cJ,cQ,cK: one-hot of private card (J=0,Q=1,K=2)
      - pot_norm: current pot / 4.0  (max pot is 4 in Kuhn)
      - bet_open: 1 if a bet has been placed and awaits response, else 0
      - to_call:  1 if acting player faces a call decision, else 0

    Notes:
      - We expose only the *current* seat's local observation.
      - infos['action_mask'] always present (size 4, {0,1} entries).
      - Rewards are given on terminal step; cumulative rewards are tracked.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "kuhn_poker_v0",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, tlogger: Optional[TLogger] = None):
        super().__init__()
        self.render_mode = render_mode
        self.tlogger = tlogger

        # Public PZ attributes
        self.possible_agents: List[str] = ["player_0", "player_1"]
        self.agents: List[str] = []

        # Spaces
        self._obs_dim = 6
        self._action_space = spaces.Discrete(4)
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32
        )

        # Game state
        self._deck = np.array([0, 1, 2], dtype=int)  # 0=J, 1=Q, 2=K
        self._cards = [-1, -1]  # private cards for player_0, player_1
        self._pot = 2  # antes in
        self._acted_first = 0  # which player acts first this hand (0 or 1)
        self._to_act = 0  # index of current player (0 or 1)
        self._bet_open = False  # has someone bet and is waiting for a response?
        self._bettor = None  # which player made the bet (0/1) if any
        self._history: List[int] = []  # action history for debugging/render

        # PettingZoo bookkeeping
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}
        self.observations: Dict[str, np.ndarray] = {}

        self.agent_selection: Optional[str] = None

    # ---------------------------
    # PettingZoo required methods
    # ---------------------------

    @property
    def observation_spaces(self):
        return {agent: self._obs_space for agent in self.possible_agents}

    @property
    def action_spaces(self):
        return {agent: self._action_space for agent in self.possible_agents}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        # Shuffle & deal
        perm = np.random.permutation(self._deck)
        self._cards[0], self._cards[1] = int(perm[0]), int(perm[1])

        # Rotate who acts first each episode for symmetry (optional)
        self._acted_first = (self._acted_first + 1) % 2
        self._to_act = self._acted_first

        # Pot & betting state
        self._pot = 2
        self._bet_open = False
        self._bettor = None
        self._history.clear()

        # Bookkeeping
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Initial obs/infos for both agents (we mirror acting player's obs to both keys)
        self._refresh_obs_and_mask()
        self.agent_selection = self.possible_agents[self._to_act]

        return self.observations, self.infos

    def observe(self, agent: str):
        # Always return *current* actor's local observation
        return self._current_obs()

    def step(self, action: int):
        # Dead step handling
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Enforce action mask
        mask = self._current_action_mask()
        if not (0 <= action < mask.size and mask[action] == 1):
            legal = np.nonzero(mask)[0]
            if legal.size > 0:
                action = int(np.random.choice(legal))

        self._apply_action(action)

        # Terminal?
        if self._is_terminal():
            self._finalize_rewards()
            for a in self.agents:
                self.terminations[a] = True
            return

        # Otherwise, switch player and refresh
        self._to_act = 1 - self._to_act
        self.agent_selection = self.possible_agents[self._to_act]
        self._refresh_obs_and_mask()

    def render(self):
        if self.render_mode is None:
            return
        # Simple textual trace
        card_str = ["J", "Q", "K"]
        s = []
        s.append(f"pot={self._pot}, bet_open={self._bet_open}, bettor={self._bettor}")
        # Only safe to reveal cards for debug (not for agents)
        s.append(f"cards: p0={card_str[self._cards[0]]}, p1={card_str[self._cards[1]]}")
        s.append(f"history: {self._history}  (0=check,1=bet,2=call,3=fold)")
        out = " | ".join(s)
        print(out)
        return out

    def close(self):
        pass

    # ---------------------------
    # Helpers
    # ---------------------------

    def _current_obs(self) -> np.ndarray:
        # Private card one-hot
        c = self._cards[self._to_act]
        onehot = np.zeros(3, dtype=np.float32)
        onehot[c] = 1.0
        # Public flags
        pot_norm = float(self._pot) / 4.0
        bet_open = 1.0 if self._bet_open else 0.0
        to_call = 1.0 if self._bet_open and (self._to_act != self._bettor) else 0.0

        obs = np.array([*onehot, pot_norm, bet_open, to_call], dtype=np.float32)
        return obs

    def _current_action_mask(self) -> np.ndarray:
        # mask of size 4: [check, bet, call, fold]
        m = np.zeros(4, dtype=np.int8)

        if not self._bet_open:
            # No outstanding bet: can check or bet
            m[0] = 1  # check
            m[1] = 1  # bet
        else:
            # Facing a bet: can call or fold (bettor cannot act here)
            if self._to_act != self._bettor:
                m[2] = 1  # call
                m[3] = 1  # fold

        return m

    def _refresh_obs_and_mask(self):
        obs = self._current_obs()
        mask = self._current_action_mask()

        # Mirror acting player's obs to both keys (trainers read from agent_selection)
        self.observations = {a: obs for a in self.agents}
        for a in self.agents:
            self.infos[a]["action_mask"] = mask

        # IMPORTANT: only clear step rewards on non-terminal steps
        is_over = any(self.terminations.values()) or any(self.truncations.values())
        if not is_over:
            for a in self.agents:
                self.rewards[a] = 0.0

    def _apply_action(self, a: int):
        """Update state and pot according to the chosen action."""
        self._history.append(a)

        if not self._bet_open:
            # No bet open yet
            if a == 0:  # check
                # If both check back-to-back, it will be terminal next turn
                pass
            elif a == 1:  # bet
                self._bet_open = True
                self._bettor = self._to_act
                self._pot += 1
            else:
                # Illegal by mask; ignore
                pass
        else:
            # There is a bet open; only the *other* player acts with call/fold
            if self._to_act == self._bettor:
                # Shouldn't happen with mask, but safeguard
                return
            if a == 2:  # call
                self._pot += 1
                # After call → showdown
            elif a == 3:  # fold
                # immediate terminal
                pass
            # Bet round resolves here; terminal next check triggers payout
            # We don't toggle bet_open here; terminal computation uses history pattern.

    def _is_terminal(self) -> bool:
        # Terminal patterns:
        #  - Both checked (check, check)
        #  - Bet then Fold (bet, fold)
        #  - Bet then Call (bet, call)
        if len(self._history) < 2:
            return False

        h = self._history
        # last two actions determine termination
        a1, a2 = h[-2], h[-1]

        # check-check
        if (not self._bet_open) and a1 == 0 and a2 == 0:
            return True

        # bet-<call|fold> (bet_open must be True and responder just acted)
        if self._bet_open and ((a2 == 2) or (a2 == 3)):
            return True

        return False

    def _finalize_rewards(self):
        # Compute terminal payoff and assign to rewards
        p0, p1 = self.possible_agents[0], self.possible_agents[1]
        payoff_to_p0 = 0.0

        h = self._history
        a1, a2 = h[-2], h[-1]

        # Case 1: check-check → showdown
        if (not self._bet_open) and a1 == 0 and a2 == 0:
            winner = self._showdown_winner()
            payoff_to_p0 = self._pot if winner == 0 else -self._pot

        # Case 2: bet-call → showdown (pot already includes +1 bettor +1 caller)
        elif self._bet_open and a2 == 2:
            winner = self._showdown_winner()
            payoff_to_p0 = self._pot if winner == 0 else -self._pot

        # Case 3: bet-fold → bettor wins current pot immediately
        elif self._bet_open and a2 == 3:
            assert self._bettor is not None
            payoff_to_p0 = self._pot if self._bettor == 0 else -self._pot

        # Zero-sum assignment
        self.rewards[p0] = float(payoff_to_p0)
        self.rewards[p1] = float(-payoff_to_p0)
        self._cumulative_rewards[p0] += self.rewards[p0]
        self._cumulative_rewards[p1] += self.rewards[p1]

    def _showdown_winner(self) -> int:
        # Higher card wins; returns winner index (0 or 1)
        c0, c1 = self._cards[0], self._cards[1]
        return 0 if c0 > c1 else 1

    # Convenience for trainers/evaluators
    def get_action_mask(self, agent="current"):
        return self._current_action_mask()
