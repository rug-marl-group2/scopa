import pyspiel
from tp_scopa_env import TwoPlayerScopaEnv

class ScopaState(pyspiel.State):
    def __init__(self, game, env=None):
        super().__init__(game)
        if env is None:
            self.env = TwoPlayerScopaEnv()
            self.env.reset()
        else:
            self.env = env
        self._is_terminal = False

    def current_player(self):
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        return 0 if self.env.agent_selection == "player_0" else 1

    def legal_actions(self, player):
        return list(range(40))

    def apply_action(self, action):
        self.env.step(action)
        self._is_terminal = all(self.env.terminations.values())

    def is_terminal(self):
        return self._is_terminal

    def clone(self):
        """Return a functional copy of the game state for CFR traversal."""
        import copy
        new_env = copy.deepcopy(self.env)  # uses your ScopaGame.__deepcopy__
        new_state = ScopaState(self.get_game(), env=new_env)
        new_state._is_terminal = self._is_terminal
        return new_state

    def rewards(self):
        if not self._is_terminal:
            return [0, 0]
        return [self.env.rewards["player_0"], self.env.rewards["player_1"]]

    def information_state_string(self, player):
        return f"hand_{player}_{self.env.game.players[player].hand}"

class ScopaGame(pyspiel.Game):
    def __init__(self):
        game_type = pyspiel.GameType(
            short_name="scopa_game",
            long_name="Two-Player Scopa",
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
        )

        game_info = pyspiel.GameInfo(
            num_distinct_actions=40,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=-1.0,
            max_utility=1.0,
            utility_sum=0.0,
            max_game_length=100,
        )

        super().__init__(game_type, game_info, {})

    def new_initial_state(self):
        return ScopaState(self)


def _scopa_game_factory(params=None):
    return ScopaGame()

pyspiel.register_game(
    pyspiel.GameType(
        short_name="scopa_game",
        long_name="Two-Player Scopa",
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
    _scopa_game_factory
)
