import pyspiel
from envs.mini_scopa_game import MiniScopaEnv


class MiniScopaState(pyspiel.State):
    """OpenSpiel-compatible state wrapper around MiniScopaEnv."""

    def __init__(self, game, env=None, num_players=2, skip_reset=False):
        super().__init__(game)
        self.num_players = num_players
        self.env = env or MiniScopaEnv(num_players=num_players)
        if not skip_reset:
            self.env.reset()
        self._is_terminal = False
        self.action_history = []  # Track actions for unique state identification

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        # Map agent_selection to OpenSpiel player index
        return self.env.agent_name_mapping[self.env.agent_selection]

    def legal_actions(self, player=None):
        """Returns legal actions based on cards in player's hand."""
        if self._is_terminal:
            return []
        
        # If no player specified, use current player
        if player is None:
            player = self.current_player()
        
        from envs.mini_scopa_game import MiniDeck
        p = self.env.game.players[player]
        legal = []
        
        # Convert each card in hand to its action index
        for card in p.hand:
            for action in range(16):
                suit_idx = action // 4
                card_idx = action % 4
                suits = MiniDeck.suits
                suit = suits[suit_idx]
                rank = MiniDeck.ranks[suit][card_idx]
                if card.rank == rank and card.suit == suit:
                    legal.append(action)
                    break
        
        return legal if legal else [0]  # Fallback to avoid empty list

    def apply_action(self, action):
        """Applies action to environment and updates terminal flag."""
        self.action_history.append(action)
        self.env.step(action)
        self._is_terminal = all(self.env.terminations.values())
    
    def _apply_action(self, action):
        """Internal method called by OpenSpiel for exploitability computation."""
        self.apply_action(action)

    def is_terminal(self):
        return self._is_terminal
    
    def is_chance_node(self):
        """Scopa has no chance nodes after initial deal."""
        return False
    
    def chance_outcomes(self):
        """No chance outcomes in Scopa (deterministic after deal)."""
        return []
    
    def history_str(self):
        """Return a unique string representing the game history."""
        # Use action history for unique identification
        history_str = "-".join(map(str, self.action_history))
        if self._is_terminal:
            # Include rewards in terminal state key to make it unique
            rewards_str = ",".join(f"{r:.2f}" for r in self.rewards())
            return f"TERMINAL:{history_str}:{rewards_str}"
        return f"H:{history_str}:P{self.current_player()}"

    def rewards(self):
        if not self._is_terminal:
            return [0] * self.num_players
        return [self.env.rewards[f"player_{i}"] for i in range(self.num_players)]

    def returns(self):
        # OpenSpiel expects returns() synonym for rewards()
        return self.rewards()

    def information_state_string(self, player=None):
        """Simplified info string: player's hand and table size."""
        if player is None:
            player = self.current_player()
        if self._is_terminal or player < 0:
            return f"TERMINAL"
        p = self.env.game.players[player]
        hand_str = "-".join(f"{c.rank}{c.suit[0]}" for c in p.hand)
        table_str = "-".join(f"{c.rank}{c.suit[0]}" for c in self.env.game.table)
        return f"P{player}:H[{hand_str}]_T[{table_str}]"

    def clone(self):
        """CFR-safe copy via state serialization."""
        from envs.mini_scopa_game import MiniScopaGame
        from gymnasium import spaces
        
        # Create new env without resetting
        new_env = MiniScopaEnv.__new__(MiniScopaEnv)
        new_env.num_players = self.num_players
        new_env.game = MiniScopaGame(num_players=self.num_players)
        new_env.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        new_env.agent_name_mapping = {name: i for i, name in enumerate(new_env.possible_agents)}
        new_env._action_spaces = {a: spaces.Discrete(16) for a in new_env.possible_agents}
        new_env.max_steps = 16
        new_env.seed = self.env.seed
        
        # Now set the state
        new_env.set_state(self.env.get_state())
        new_state = MiniScopaState(self.get_game(), env=new_env, num_players=self.num_players, skip_reset=True)
        new_state._is_terminal = self._is_terminal
        new_state.action_history = self.action_history.copy()  # Copy action history
        return new_state


class MiniScopaGame(pyspiel.Game):
    """Game wrapper for OpenSpiel registration."""

    def __init__(self, num_players=2):
        self._num_players = num_players
        game_type = pyspiel.GameType(
            short_name="mini_scopa",
            long_name="Two-Player Mini-Scopa",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=num_players,
            min_num_players=num_players,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=False,
            provides_observation_tensor=False,
            parameter_specification={},
            default_loadable=True,
            provides_factored_observation_string=False,
        )

        game_info = pyspiel.GameInfo(
            num_distinct_actions=16,     # 16 possible card encodings
            max_chance_outcomes=0,
            num_players=num_players,
            min_utility=-10.0,
            max_utility=10.0,
            utility_sum=0.0,
            max_game_length=num_players * 4,  # 4 cards per player
        )

        super().__init__(game_type, game_info, {})

    def num_players(self):
        """Return number of players as a method."""
        return self._num_players

    def new_initial_state(self):
        return MiniScopaState(self, num_players=self._num_players)


def _mini_scopa_factory(params=None):
    return MiniScopaGame()


pyspiel.register_game(
    pyspiel.GameType(
        short_name="mini_scopa",
        long_name="Two-Player Mini-Scopa",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=2,
        min_num_players=2,
        provides_information_state_string=True,
        provides_information_state_tensor=False,
        provides_observation_string=False,
        provides_observation_tensor=False,
        parameter_specification={},
        default_loadable=True,
        provides_factored_observation_string=False,
    ),
    _mini_scopa_factory
)
