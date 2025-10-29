from typing import Optional, Dict, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

# Import your base 4-seat env
from src.games.scopa import env as make_4seat_env, MaScopaEnv
from src.tlogger import TLogger  # optional


@property
def game(self):
    # Back-compat for code that accesses env.game (e.g., traverser)
    return self.base.game


def env(tlogger: Optional[TLogger] = None, render_mode=None):
    """
    Factory for the 1v1 team abstraction.
    Keeps PettingZoo 'ansi' behavior consistent with your base env.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = TeamScopa1v1Env(render_mode=internal_render_mode, tlogger=tlogger)
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    return e


class TeamScopa1v1Env(AECEnv):
    """
    A thin 1v1 (TeamA vs TeamB) wrapper around the 4-seat MaScopaEnv.

    - Current acting seat in {0,1,2,3} belongs to TeamA iff seat % 2 == 0, else TeamB.
    - Observations and action masks are proxied from the currently acting seat.
    - Rewards at episode end are mapped from team outcomes {+1,0,-1} to {TeamA, TeamB}.
    - No partner private info is revealed (we forward seat-local observations only).

    Agents: ["TeamA", "TeamB"]
    """

    metadata = {
        "render_modes": ["human"],
        "name": "scopa_1v1_wrapper_v0",
        "is_parallelizable": True,
    }

    TEAM_A = "TeamA"
    TEAM_B = "TeamB"
    TEAM_NAMES = [TEAM_A, TEAM_B]

    def __init__(self, render_mode=None, tlogger: Optional[TLogger] = None):
        super().__init__()
        self.render_mode = render_mode
        self.tlogger = tlogger

        # Underlying 4-seat env
        self.base: MaScopaEnv = make_4seat_env(
            tlogger=self.tlogger, render_mode=self.render_mode
        )

        # Public PettingZoo attributes
        self.possible_agents = [self.TEAM_A, self.TEAM_B]
        self.agents = []  # set in reset()

        # Caches
        self._obs_shape = self._infer_obs_shape()
        self._team_action_space = spaces.Discrete(40)  # same as base per-seat space

        # State
        self._current_seat: Optional[int] = None  # in {0,1,2,3}
        self.agent_selection: Optional[str] = None

        # Bookkeeping
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}
        self.observations: Dict[str, np.ndarray] = {}

    # ---------------------------
    # Required PettingZoo methods
    # ---------------------------

    @property
    def observation_spaces(self):
        # Same obs space as the base seat-local obs (shape like (5,40) in your patched env)
        return {
            agent: spaces.Box(0.0, 1.0, shape=self._obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }

    @property
    def action_spaces(self):
        return {agent: self._team_action_space for agent in self.possible_agents}

    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, shape=self._obs_shape, dtype=np.float32)

    def action_space(self, agent):
        return self._team_action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset base env
        obs, infos = self.base.reset(seed=seed, options=options)

        # Determine the current acting seat in the base env
        self._current_seat = self._seat_index_from_name(self.base.agent_selection)
        current_team = self._team_from_seat(self._current_seat)

        # Initialize PZ bookkeeping
        self.agents = self.possible_agents[
            :
        ]  # both teams present for the whole episode
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Fill initial observations and action masks for both teams
        self._refresh_obs_and_masks()

        # Set which team acts first
        self.agent_selection = current_team

        return self.observations, self.infos

    def observe(self, agent: str):
        # Return the view for the *currently acting seat*, regardless of agent asked
        seat = self._current_seat
        seat_agent_name = self.base.possible_agents[seat]
        return self.base.observations[seat_agent_name]

    def step(self, action: int):
        # If episode over for this team, do dead step
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Enforce legal mask of the acting seat (base env already asserts legality & repairs)
        seat = self._current_seat
        seat_agent_name = self.base.possible_agents[seat]
        seat_mask = self.base.infos[seat_agent_name].get(
            "action_mask", self.base.get_action_mask(seat_agent_name)
        )

        # Defensive repair (should be handled by base too)
        if not (0 <= action < seat_mask.size and seat_mask[action] == 1):
            legal = np.nonzero(seat_mask)[0]
            if legal.size > 0:
                action = int(np.random.choice(legal))
            # else let base handle dead step fallback

        # Forward the step to the base env for the *current seat*
        self.base.step(action)

        # Advance seat pointer (base already moved to next seat)
        if self._episode_over_in_base():
            # Episode ended in base env: map team rewards/terminations
            self._map_terminal_from_base()
        else:
            # Not over – update current seat and whose turn it is
            self._current_seat = self._seat_index_from_name(self.base.agent_selection)
            self.agent_selection = self._team_from_seat(self._current_seat)
            self._refresh_obs_and_masks()

        # Optional: accumulate custom logs
        # (Your TLogger hooks are already called inside the base env.)

    def render(self):
        return self.base.render()

    # -------------
    # Helper methods
    # -------------

    def _infer_obs_shape(self) -> Tuple[int, int]:
        """
        Ask the base env for one seat's observation to determine shape.
        """
        # Ensure base is constructed and reset once
        if (
            not hasattr(self.base, "observations")
            or len(getattr(self.base, "observations", {})) == 0
        ):
            self.base.reset()
        any_agent = self.base.possible_agents[0]
        sample_obs = self.base.observations[any_agent]
        assert (
            isinstance(sample_obs, np.ndarray) and sample_obs.ndim == 2
        ), "Expected seat-local observation of shape (H, W)"
        return sample_obs.shape  # e.g., (5, 40)

    def _seat_index_from_name(self, name: str) -> int:
        """
        Base env names seats as 'player_0'..'player_3'.
        """
        # base.agent_name_mapping: {'player_0': 0, ...}
        return self.base.agent_name_mapping[name]

    def _team_from_seat(self, seat_index: int) -> str:
        return self.TEAM_A if (seat_index % 2 == 0) else self.TEAM_B

    def _refresh_obs_and_masks(self):
        """
        Update observations and infos['action_mask'] for both teams,
        but always computed from the *current seat* view.
        """
        seat = self._current_seat
        seat_agent_name = self.base.possible_agents[seat]

        # Proxy current seat view to both teams (only the acting team will use it to act)
        current_obs = self.base.observations[seat_agent_name]
        current_mask = self.base.infos[seat_agent_name].get(
            "action_mask", self.base.get_action_mask(seat_agent_name)
        )

        self.observations = {
            self.TEAM_A: current_obs,
            self.TEAM_B: current_obs,
        }
        # Forward seat index & team info (useful for trainers to condition on seat id)
        seat_meta = {
            "seat_index": seat,
            "team_local_seat": (
                0 if (seat % 2 == 0) else 1
            ),  # seat within its team: {0,1}
            "action_mask": current_mask.astype(int, copy=False),
        }
        self.infos[self.TEAM_A].update(seat_meta)
        self.infos[self.TEAM_B].update(seat_meta)

        # Keep rewards dict present (0 until terminal)
        for a in self.agents:
            self.rewards[a] = 0.0

    def _episode_over_in_base(self) -> bool:
        # Base env signals end via all seats terminated or truncated
        term_over = all(self.base.terminations.get(a, False) for a in self.base.agents)
        trunc_over = all(self.base.truncations.get(a, False) for a in self.base.agents)
        return term_over or trunc_over

    def _map_terminal_from_base(self):
        """
        Map terminal state from base env to TeamA/TeamB rewards and done flags.
        Base env already computed per-seat rewards = team outcome in {-1,0,+1}.
        """
        # Determine final per-team score from any seat (0 & 2 share team, 1 & 3 share team)
        # Use seat 0's reward as TeamA, seat 1's reward as TeamB (consistent with your base env).
        seat0_name = self.base.possible_agents[0]
        seat1_name = self.base.possible_agents[1]

        teamA_reward = float(self.base.rewards.get(seat0_name, 0.0))
        teamB_reward = float(self.base.rewards.get(seat1_name, 0.0))

        self.rewards = {
            self.TEAM_A: teamA_reward,
            self.TEAM_B: teamB_reward,
        }
        self._cumulative_rewards[self.TEAM_A] += teamA_reward
        self._cumulative_rewards[self.TEAM_B] += teamB_reward

        # Everyone done
        self.terminations = {a: True for a in self.agents}
        self.truncations = {a: False for a in self.agents}

        # Keep final observations/masks for completeness
        self._refresh_obs_and_masks()

    # PettingZoo convenience (optional, mirrors base)
    def get_action_mask(self, agent="current"):
        # Always return the mask of the *current seat*
        seat = self._current_seat
        seat_agent_name = self.base.possible_agents[seat]
        return self.base.infos[seat_agent_name].get(
            "action_mask", self.base.get_action_mask(seat_agent_name)
        )

    # Boilerplate: these make AECEnv happy when someone calls these inadvertently
    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()

    def roundScores(self):
        # Proxy underlying archive if needed for analysis
        if hasattr(self.base, "roundScores"):
            return self.base.roundScores()
        return []
