import pyspiel

from src.mini_scopa_game import MiniDeck, MiniScopaEnv


class MiniScopaState(pyspiel.State):
    """OpenSpiel-compatible state wrapper around MiniScopaEnv."""

    def __init__(self, game, env=None, num_players=2, skip_reset=False):
        super().__init__(game)
        self.num_players = num_players
        self.env = env or MiniScopaEnv(num_players=num_players)
        if not skip_reset:
            self.env.reset()
        self._is_terminal = False

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
        self.env.step(action)
        self._is_terminal = all(self.env.terminations.values())

    def is_terminal(self):
        return self._is_terminal

    def rewards(self):
        if not self._is_terminal:
            return [0] * self.num_players
        return [self.env.rewards[f"player_{i}"] for i in range(self.num_players)]

    def returns(self):
        # OpenSpiel expects returns() synonym for rewards()
        return self.rewards()

    def information_state_string(self, player):
        """Simplified info string: player's hand and table size."""
        p = self.env.game.players[player]
        hand_str = "-".join(f"{c.rank}{c.suit[0]}" for c in p.hand)
        table_str = "-".join(f"{c.rank}{c.suit[0]}" for c in self.env.game.table)
        return f"P{player}:H[{hand_str}]_T[{table_str}]"

    def clone(self):
        """CFR-safe copy via state serialization."""
        from gymnasium import spaces

        from src.mini_scopa_game import MiniScopaGame

        # Create new env without resetting
        new_env = MiniScopaEnv.__new__(MiniScopaEnv)
        new_env.num_players = self.num_players
        new_env.game = MiniScopaGame(num_players=self.num_players)
        new_env.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        new_env.agent_name_mapping = {
            name: i for i, name in enumerate(new_env.possible_agents)
        }
        new_env._action_spaces = {
            a: spaces.Discrete(16) for a in new_env.possible_agents
        }
        new_env.max_steps = 16
        new_env.seed = self.env.seed

        # Now set the state
        new_env.set_state(self.env.get_state())
        new_state = MiniScopaState(
            self.get_game(), env=new_env, num_players=self.num_players, skip_reset=True
        )
        new_state._is_terminal = self._is_terminal
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
            num_distinct_actions=16,  # 16 possible card encodings
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
    _mini_scopa_factory,
)
