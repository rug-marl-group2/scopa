import pyspiel
from envs.team_mini_scopa_game import TeamMiniScopaEnv, MiniDeck,TeamMiniScopaGame
from gymnasium import spaces
        

class TPIMiniScopaState(pyspiel.State):
    

    def __init__(self, game, env=None, skip_reset=False):
        super().__init__(game)
        self.env = env or TeamMiniScopaEnv()
        if not skip_reset:
            self.env.reset()
        self._is_terminal = False
        self.action_history = []
        
        self._current_team = 0
        
        self._team_private_states = [None, None]

    def current_player(self):
        """Return which coordinator (team) is playing"""
        if self._is_terminal:
            return pyspiel.PlayerId.TERMINAL
        
        # Determine which team should play based on current game state
        agent_id = self.env.agent_name_mapping[self.env.agent_selection]
        return self.env.game.get_team(agent_id)

    def _get_private_state_id(self, player_id):
      
        player = self.env.game.players[player_id]
        cards = sorted([(c.rank, c.suit) for c in player.hand])
        return tuple(cards)

    def _get_team_private_states(self, team_id):
      
        # Get the two players on this team
        if team_id == 0:
            player_ids = [0, 1]
        else:
            player_ids = [2, 3]
        
        # Get private states for each player
        private_states = []
        for pid in player_ids:
            ps = self._get_private_state_id(pid)
            private_states.append(ps)
        
        return tuple(private_states)

    def legal_actions(self, player=None):
       
        if self._is_terminal:
            return []
        
        if player is None:
            player = self.current_player()
        
        team_id = player
        
        # Get the two players on this team
        if team_id == 0:
            player_ids = [0, 1]
        else:
            player_ids = [2, 3]
        
        # Get current game player (who actually moves in the underlying game)
        current_agent = self.env.agent_selection
        current_player_id = self.env.agent_name_mapping[current_agent]
        
        # Get legal actions for current player
        player_obj = self.env.game.players[current_player_id]
        player_legal_actions = []
        
        for card in player_obj.hand:
            for action in range(16):
                suit_idx = action // 4
                card_idx = action % 4
                suits = MiniDeck.suits
                suit = suits[suit_idx]
                rank = MiniDeck.ranks[suit][card_idx]
                if card.rank == rank and card.suit == suit:
                    player_legal_actions.append(action)
                    break
        
        if not player_legal_actions:
            return [0]
        
        # In the TPI representation, each prescription corresponds to
        # choosing an action for the current player's private state.
        # For simplicity in this implementation, we encode prescriptions
        # as just the action for the current player (since teammate actions
        # don't matter until their turn).
        return player_legal_actions

    def apply_action(self, action):
       
        self.action_history.append(action)
        self.env.step(action)
        self._is_terminal = all(self.env.terminations.values())

    def is_terminal(self):
        return self._is_terminal

    def is_chance_node(self):
        return False

    def chance_outcomes(self):
        return []

    def history_str(self):
        """Unique string for game history"""
        history_str = "-".join(map(str, self.action_history))
        if self._is_terminal:
            rewards_str = ",".join(f"{r:.2f}" for r in self.rewards())
            return f"TERMINAL:{history_str}:{rewards_str}"
        return f"H:{history_str}:T{self.current_player()}"

    def rewards(self):
        """Return rewards for each team (coordinator)"""
        if not self._is_terminal:
            return [0, 0]
        
        # Get per-player rewards and aggregate by team
        player_rewards = [self.env.rewards[f"player_{i}"] for i in range(4)]
        
        # Team 0 gets average of players 0 and 1
        # Team 1 gets average of players 2 and 3
        team_0_reward = (player_rewards[0] + player_rewards[1]) / 2
        team_1_reward = (player_rewards[2] + player_rewards[3]) / 2
        
        return [team_0_reward, team_1_reward]

    def returns(self):
        return self.rewards()

    def information_state_string(self, player):
        
        team_id = player
        
        # Get players on this team
        if team_id == 0:
            player_ids = [0, 1]
        else:
            player_ids = [2, 3]
        
        # Current player whose turn it is
        current_agent = self.env.agent_selection
        current_player_id = self.env.agent_name_mapping[current_agent]
        
        if current_player_id not in player_ids:
            # Not this team's turn, return info from perspective
            current_player_id = player_ids[0]
        
        # Get current player's hand
        player_obj = self.env.game.players[current_player_id]
        hand_cards = sorted([(c.rank, c.suit) for c in player_obj.hand])
        hand_str = "-".join(f"{r}{s[0]}" for r, s in hand_cards)
        
        # Table cards (public information)
        table_cards = sorted([(c.rank, c.suit) for c in self.env.game.table])
        table_str = "-".join(f"{r}{s[0]}" for r, s in table_cards)
        
        # Action history (public coordination)
        history_str = "-".join(map(str, self.action_history))
        
        return f"Team{team_id}:P{current_player_id}:H[{hand_str}]:T[{table_str}]:A[{history_str}]"

    def clone(self):
        
        new_env = TeamMiniScopaEnv.__new__(TeamMiniScopaEnv)
        new_env.game = TeamMiniScopaGame()
        new_env.possible_agents = [f"player_{i}" for i in range(4)]
        new_env.agent_name_mapping = {name: i for i, name in enumerate(new_env.possible_agents)}
        new_env._action_spaces = {a: spaces.Discrete(16) for a in new_env.possible_agents}
        new_env.max_steps = 16
        new_env.seed = self.env.seed
        
        # Copy state
        new_env.set_state(self.env.get_state())
        
        new_state = TPIMiniScopaState(self.get_game(), env=new_env, skip_reset=True)
        new_state._is_terminal = self._is_terminal
        new_state.action_history = self.action_history.copy()
        new_state._current_team = self._current_team
        
        return new_state


class TPIMiniScopaGame(pyspiel.Game):
    """
    Team-Public-Information game for Team Mini Scopa.
    
    Converts 2v2 team game into 2-player zero-sum game following
    Carminati et al. (ICML 2022) framework.
    """

    def __init__(self):
        game_type = pyspiel.GameType(
            short_name="team_mini_scopa_tpi",
            long_name="Team Mini Scopa - TPI Representation",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=2,  # 2 teams
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
            num_distinct_actions=16,  # 16 possible cards
            max_chance_outcomes=0,
            num_players=2,  # 2 teams (coordinators)
            min_utility=-20.0,
            max_utility=20.0,
            utility_sum=0.0,
            max_game_length=16,  # 4 players × 4 cards
        )

        super().__init__(game_type, game_info, {})

    def num_players(self):
        return 2  # 2 teams

    def new_initial_state(self):
        return TPIMiniScopaState(self)


def _team_mini_scopa_tpi_factory(params=None):
    return TPIMiniScopaGame()


# Register with OpenSpiel
pyspiel.register_game(
    pyspiel.GameType(
        short_name="team_mini_scopa_tpi",
        long_name="Team Mini Scopa - TPI Representation",
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
    _team_mini_scopa_tpi_factory
)

