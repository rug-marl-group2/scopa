from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
# PettingZoo changed agent_selector export; support both styles
try:
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
except Exception:  # pragma: no cover
    from pettingzoo.utils import agent_selector as _agent_selector_mod
    AgentSelector = _agent_selector_mod.agent_selector
from gymnasium import spaces
import numpy as np
import random
import itertools
import copy
from tlogger import TLogger

NUM_ITERS = 100  # Number of iterations before truncation
PRINT_DEBUG = False

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        rank_raster = self.rank

        if rank_raster == 10:
            rank_raster = "King"
        elif rank_raster == 9:
            rank_raster = "Queen"
        elif rank_raster == 8:
            rank_raster = "Jack"

        if self.suit == "bello":
            return f"{self.rank} {self.suit}"
        else:
            return f"{self.rank} di {self.suit}"

class Deck:
    suits = ['picche', 'bello', 'fiori', 'cuori']
    ranks = list(range(1, 11))  # Ranks from 1 to 7, plus 8, 9, and 10 for face cards.

    def __init__(self):
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    def shuffle(self, seed = 42):
        random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, num_cards: int):
        return [self.cards.pop() for _ in range(num_cards)]

class Player:
    def __init__(self, side: int, name: str):
        self.side = side
        self.name = name
        self.hand = []
        self.captures = []
        self.history = []
        self.scopas = 0

    def reset(self):
        
        self.hand = []
        self.captures = []
        self.history = []
        self.scopas = 0

    def capture(self, cards, _with = None):
        self.captures.extend(cards)
        if _with is not None and _with in self.hand:
            self.hand.remove(_with)

    def play_card(self, card_index):
        return self.hand.pop(card_index)

class ScopaGame:
    def __init__(self, logger):
        self.deck = Deck()
        self.players = [Player(1, 'player_0'), Player(2, 'player_1'), Player(1, 'player_2'), Player(2, 'player_3')]
        self.table = []
        self.last_capture = None
        self.tlogger = logger

    def reset(self, seed = 42):
        self.deck = Deck()
        self.deck.shuffle(seed=seed)
        self.table = []
        self.last_capture = None
        for player in self.players:
            player.reset()
        for player in self.players:
            player.hand = self.deck.deal(10)

    def card_in_table(self, card):
        # Find a subset of table cards whose ranks sum to the played card's rank.
        # Optimized with DP in O(n * target) rather than enumerating all subsets.
        target = card.rank
        if target <= 0 or not self.table:
            return False, []

        ranks = [c.rank for c in self.table]
        # comb_sums[s] stores a tuple of indices in `self.table` whose ranks sum to s
        comb_sums = [None] * (target + 1)
        comb_sums[0] = ()
        for idx, r in enumerate(ranks):
            # iterate sums in descending order to avoid reusing the same card
            for s in range(target, r - 1, -1):
                if comb_sums[s] is None and comb_sums[s - r] is not None:
                    comb_sums[s] = comb_sums[s - r] + (idx,)

        if comb_sums[target] is None:
            return False, []

        combo = [self.table[i] for i in comb_sums[target]]
        return True, combo

    def play_card(self, card, player):
        player.history.append(card)
        player_index = self.players.index(player)

        # Ace mechanics: captures entire table without subset search
        if card.rank == 1:
            if PRINT_DEBUG: print(f"\t!!! Ace == {card} for player {[card.__str__() for card in player.hand]}")
            self.table.append(card)
            player.capture(self.table, _with=card)
            self.table = []
            self.last_capture = player_index
            return

        # Non-ace: check for a capturing combination
        isin, comb = self.card_in_table(card)
        if isin:
            for c in comb:
                self.table.remove(c)
            comb.append(card)
            player.capture(comb, _with=card)
            self.last_capture = player_index
            if len(self.table) == 0:
                player.scopas += 1
                if self.tlogger is not None:
                    self.tlogger.scopa(player)
                if PRINT_DEBUG: print(f"\t!!! Scopa for player {player.side}")
        else:
            player.hand.remove(card)
            self.table.append(card)

    def evaluate_round(self):
        if self.table and self.last_capture is not None:
            capturing_player = self.players[self.last_capture]
            capturing_player.capture(self.table)
            self.table = []
            self.last_capture = None
            
        # Shared captures by team
        team1_captures = [card for player in self.players if player.side == 1 for card in player.captures]
        team2_captures = [card for player in self.players if player.side == 2 for card in player.captures]

        # Initialize points
        team1_points = 0
        team2_points = 0

        # Count Scopas
        team1_points += sum(player.scopas for player in self.players if player.side == 1)
        team2_points += sum(player.scopas for player in self.players if player.side == 2)

        # Most Cards
        if len(team1_captures) > len(team2_captures):
            team1_points += 1
        elif len(team2_captures) > len(team1_captures):
            team2_points += 1

        # Most Coins ("ori")
        team1_coins = [card for card in team1_captures if card.suit == 'bello']
        team2_coins = [card for card in team2_captures if card.suit == 'bello']
        if len(team1_coins) > len(team2_coins):
            team1_points += 1
        elif len(team2_coins) > len(team1_coins):
            team2_points += 1

        # Sette Bello (Seven of Coins)
        for card in team1_captures:
            if card.rank == 7 and card.suit == 'bello':
                team1_points += 1
                break
        for card in team2_captures:
            if card.rank == 7 and card.suit == 'bello':
                team2_points += 1
                break

        # Primiera
        suit_priority = {7: 4, 6: 3, 1: 2, 5: 1, 4: 0, 3: 0, 2: 0}
        team1_best_cards = [max((card for card in team1_captures if card.suit == suit), key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]
        team2_best_cards = [max((card for card in team2_captures if card.suit == suit), key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]

        team1_primiera = sum(suit_priority.get(card.rank, 0) for card in team1_best_cards if card)
        team2_primiera = sum(suit_priority.get(card.rank, 0) for card in team2_best_cards if card)

        if team1_primiera > team2_primiera:
            team1_points += 1
        elif team2_primiera > team1_primiera:
            team2_points += 1

        # Return final round scores
        # if team1_points > team2_points:
        #     return (team1_points - team2_points) , (team2_points - team1_points)
        # elif team2_points > team1_points:
        #     return (team2_points - team1_points), (team1_points - team2_points)
        # else:
        #     return 0, 0

        # 1 or -1 reward type
        if team1_points > team2_points:
            return 1 , -1
        elif team2_points > team1_points:
            return -1, 1
        else:
            return 0, 0

    def __deepcopy__(self, memo):
        # Custom light-weight copy: share Card objects, copy only list containers.
        # This drastically reduces overhead for CFR branching.
        new_game = ScopaGame(logger=None)
        memo[id(self)] = new_game
        # Deck is not used after reset within a round; share reference to avoid heavy copy
        new_game.deck = self.deck
        # Copy players with shallow copies of their lists
        new_players = []
        for p in self.players:
            np = Player(p.side, p.name)
            np.hand = list(p.hand)
            np.captures = list(p.captures)
            np.history = list(p.history)
            np.scopas = p.scopas
            new_players.append(np)
        new_game.players = new_players
        # Shallow copy of table and last_capture
        new_game.table = list(self.table)
        new_game.last_capture = self.last_capture
        new_game.tlogger = None
        return new_game

def env(tlogger, render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = MaScopaEnv(render_mode=internal_render_mode, tlogger=tlogger)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    return env

class MaScopaEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "scopa_v0",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, tlogger=None):
        super().__init__()
        self.render_mode = render_mode

        self.tlogger = tlogger
        self.commulative_sides = [0, 0]
        self.archive_scores = []

        self.game = ScopaGame(logger=self.tlogger)
        self.possible_agents = [player.name for player in self.game.players]
        self.agent_name_mapping = {agent: int(agent[-1]) for  agent in self.possible_agents}

        self._suit_offset = {
            'cuori': 0,
            'picche': 10,
            'fiori': 20,
            'bello': 30,
        }
        players = len(self.game.players)
        self.global_state_dim = (players * 3 * 40) + 40 + (players * 2)

        self._action_spaces = {
            #agent: spaces.Box(0, 1, shape=(1,40)) for agent in self.possible_agents
            agent: spaces.Discrete(40) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(0, 1, shape=(4, 40), dtype=np.float32) for agent in self.possible_agents
        }

        self.reset()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _encode_cards(self, cards):
        vec = np.zeros(40, dtype=np.float32)
        for card in cards:
            index = (card.rank - 1) + self._suit_offset[card.suit]
            vec[index] += 1.0
        return vec

    def get_global_state(self):
        """Return a flat global state vector for centralized critics."""
        hands = [self._encode_cards(player.hand) for player in self.game.players]
        table = self._encode_cards(self.game.table)
        captures = [self._encode_cards(player.captures) for player in self.game.players]
        history = [self._encode_cards(player.history) for player in self.game.players]
        scopas = np.asarray([player.scopas for player in self.game.players], dtype=np.float32)
        current = np.zeros(len(self.game.players), dtype=np.float32)
        if getattr(self, 'agent_selection', None) in self.agent_name_mapping:
            current_index = self.agent_name_mapping[self.agent_selection]
            current[current_index] = 1.0
        feature_list = hands + [table] + captures + history + [scopas, current]
        return np.concatenate(feature_list).astype(np.float32, copy=False)


    def observe(self, agent):
        player_index = self.agent_name_mapping[agent]
        friend_index = (player_index + 2) % 4
        player = self.game.players[player_index]
        friend = self.game.players[friend_index]

        state = np.zeros((4, 40), dtype=np.float32)
        offset = self._suit_offset

        for card in player.hand:
            index = (card.rank - 1) + offset[card.suit]
            state[0][index] = 1.0

        for card in self.game.table:
            index = (card.rank - 1) + offset[card.suit]
            state[1][index] = 1.0

        for card in player.captures:
            index = (card.rank - 1) + offset[card.suit]
            state[2][index] = 1.0

        for card in friend.captures:
            index = (card.rank - 1) + offset[card.suit]
            state[2][index] = 1.0

        for card in friend.history:
            index = (card.rank - 1) + offset[card.suit]
            state[3][index] = 1.0

        return state

    def reset(self, seed='42', options='43'):
        #if seed != '42' or options != '43':

            #print(f'### WARNING: seed and options are not used in this environment. Expected [42|43]. Recieved: [{seed}|{options}]')
        self.game.reset(seed=seed)
        self.num_moves = 0
        # Randomize the starting player SUPER IMPORTANT otherwise the not-starting side would have an advantage
        random.seed(seed)
        randstart = random.randint(0, 3)
        self.possible_agents = self.possible_agents[randstart:] + self.possible_agents[:randstart]
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"action_mask": self.get_action_mask(agent)} for agent in self.agents}
        self.observations = {agent: self.observe(agent) for agent in self.agents}
        
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def get_action_mask(self, agent = 'current'):

        if agent == 'current':
            action_mask = np.zeros(40, dtype=int)
            for card in self.game.players[self.agent_name_mapping[self.agent_selection]].hand:
                index = (card.rank - 1) + {
                    'cuori': 0,
                    'picche': 10,
                    'fiori': 20,
                    'bello': 30
                }[card.suit]
                action_mask[index] = 1
            
        elif agent is None:
            action_mask = np.zeros(160, dtype=int)

            for t, player in enumerate(self.game.players):
                for card in player.hand:
                    index = (t * 40) + (card.rank - 1) + {
                        'cuori': 0,
                        'picche': 10,
                        'fiori': 20,
                        'bello': 30
                    }[card.suit]
                    action_mask[index] = 1
        else:
            action_mask = np.zeros(40, dtype=int)
            player_index = self.agent_name_mapping[agent]
            player = self.game.players[player_index]

            for card in player.hand:
                index = (card.rank - 1) + {
                    'cuori': 0,
                    'picche': 10,
                    'fiori': 20,
                    'bello': 30
                }[card.suit]
                action_mask[index] = 1

        return action_mask

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return


        agent = self.agent_selection
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]

        card = None

        

        for c in player.hand:
            ind = (c.rank - 1) + {
                'cuori': 0,
                'picche': 10,
                'fiori': 20,
                'bello': 30
            }[c.suit]

            if ind == action:
                card = c


        if PRINT_DEBUG: 
            print(f"\n### Agent {agent} plays card: {card}")
            print(f"### Table state before play: {[card.__str__() for card in self.game.table]}")

        self.game.play_card(card, player)

        for name in self.agents:
            self.observations[name] = self.observe(name)
            if name not in self.infos:
                self.infos[name] = {}
            self.infos[name]["action_mask"] = self.get_action_mask(name)

        if PRINT_DEBUG: 
            print(f"### Table state after play: {[card.__str__() for card in self.game.table]}")

        # Check if all players have played their cards
        if all(len(player.hand) == 0 for player in self.game.players):
            # Evaluate the round and assign rewards
            round_scores = self.game.evaluate_round()
            self.archive_scores.append(round_scores)
            self.commulative_sides[0] += round_scores[0]
            self.commulative_sides[1] += round_scores[1]
            self.tlogger.writer.add_scalar("Scores/Side/0", self.commulative_sides[0], self.tlogger.simulation_clock)   
            self.tlogger.writer.add_scalar("Scores/Side/1", self.commulative_sides[1], self.tlogger.simulation_clock)
            if PRINT_DEBUG: print(f"### Round scores: {round_scores}")
            for i, agent in enumerate(self.possible_agents):
                self.rewards[agent] = round_scores[self.agent_name_mapping[agent] % 2]
            if PRINT_DEBUG: print(f"### Rewards after termination: {self.rewards}")
            self.terminations = {agent: True for agent in self.agents}  # End the game

        if PRINT_DEBUG: print(f"### Observations updated for agent {agent}")
        self.num_moves += 1

        if self.num_moves >= NUM_ITERS:
            self.truncations = {a: True for a in self.agents}

        if PRINT_DEBUG: print(f"### Agent selection moved to next agent")
        self.agent_selection = self._agent_selector.next()

        self.tlogger.record_step()

    def roundScores(self):
        return self.archive_scores

    def render(self):
        if self.render_mode == "human":
            print(self.game.table)
