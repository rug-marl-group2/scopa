import pyspiel
from envs.full_scopa_game import FullScopaEnv, FullScopaGame
from gymnasium import spaces
class FullScopaState(pyspiel.State):
    """OpenSpiel-compatible state wrapper around FullScopaEnv."""

    def __init__(self, game, env=None, num_players=2, skip_reset=False):
        super().__init__(game)
        self.num_players = num_players
        self.env = env or FullScopaEnv(num_players=num_players)
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
        
        from envs.full_scopa_game import FullDeck
        p = self.env.game.players[player]
        legal = []
        
        # Convert each card in hand to its action index
        for card in p.hand:
            # Action encoding: suit_idx * 10 + (rank - 1)
            suit_idx = FullDeck.suits.index(card.suit)
            action = suit_idx * 10 + (card.rank - 1)
            legal.append(action)
        
        return legal if legal else [0]  # Fallback to avoid empty list

    def apply_action(self, action):
        """Applies action to environment and updates terminal flag."""
        self.action_history.append(action)
        self.env.step(action)
        self._is_terminal = all(self.env.terminations.values())

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
        return self.rewards()

    def information_state_string(self, player):
        p = self.env.game.players[player]
        
        hand_cards = sorted([(c.rank, c.suit) for c in p.hand])
        hand_str = "-".join(f"{r}{s[0]}" for r, s in hand_cards)
        
        table_cards = sorted([(c.rank, c.suit) for c in self.env.game.table])
        table_str = "-".join(f"{r}{s[0]}" for r, s in table_cards)
        
        # Capture counts (approximate info - full captures would be too large)
        capture_counts = ",".join(str(len(pl.captures)) for pl in self.env.game.players)
        
        # Scopa counts
        scopa_counts = ",".join(str(pl.scopas) for pl in self.env.game.players)
        
        return f"P{player}:R{self.env.game.round_number}:H[{hand_str}]:T[{table_str}]:C[{capture_counts}]:S[{scopa_counts}]"

    def clone(self):
       
        new_env = FullScopaEnv.__new__(FullScopaEnv)
        new_env.num_players = self.num_players
        new_env.game = FullScopaGame(num_players=self.num_players)
        new_env.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        new_env.agent_name_mapping = {name: i for i, name in enumerate(new_env.possible_agents)}
        new_env._action_spaces = {a: spaces.Discrete(40) for a in new_env.possible_agents}
        new_env.max_steps = 200
        new_env.seed = self.env.seed
        
        # Set the state
        new_env.set_state(self.env.get_state())
        new_state = FullScopaState(self.get_game(), env=new_env, num_players=self.num_players, skip_reset=True)
        new_state._is_terminal = self._is_terminal
        new_state.action_history = self.action_history.copy()
        return new_state


class FullScopaGame(pyspiel.Game):
    """Game wrapper for OpenSpiel registration."""

    def __init__(self, num_players=2):
        self._num_players = num_players
        game_type = pyspiel.GameType(
            short_name="full_scopa",
            long_name="Full Italian Scopa",
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
            num_distinct_actions=40,    
            max_chance_outcomes=0,
            num_players=num_players,
            min_utility=-10.0,
            max_utility=10.0,
            utility_sum=0.0,
            max_game_length=40,  # Maximum 40 cards to play
        )

        super().__init__(game_type, game_info, {})

    def num_players(self):
        """Return number of players as a method."""
        return self._num_players

    def new_initial_state(self):
        return FullScopaState(self, num_players=self._num_players)


def _full_scopa_factory(params=None):
    return FullScopaGame()


# Register the game with OpenSpiel
pyspiel.register_game(
    pyspiel.GameType(
        short_name="full_scopa",
        long_name="Full Italian Scopa",
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
    _full_scopa_factory
)

