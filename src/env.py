from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
try:
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
except Exception:  
    from pettingzoo.utils import agent_selector as _agent_selector_mod
    AgentSelector = _agent_selector_mod.agent_selector
from gymnasium import spaces
import numpy as np
import random
import itertools
import copy
from tlogger import TLogger

NUM_ITERS = 100
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
            return f"{self.rank} {self.suit}"  # bello == denari (coins)
        else:
            return f"{self.rank} di {self.suit}"

class Deck:
    suits = ['picche', 'bello', 'fiori', 'cuori']
    ranks = list(range(1, 11))

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
        self.last_scopa_player = None  # track who made the most recent scopa (to uncount on last move)
        self.tlogger = logger

    def reset(self, seed = 42):
        self.deck = Deck()
        self.deck.shuffle(seed=seed)
        self.table = []
        self.last_capture = None
        self.last_scopa_player = None
        for player in self.players:
            player.reset()
        for player in self.players:
            player.hand = self.deck.deal(10)

    def _single_equal_rank(self, target_rank: int):
        """Return first single-card match index on table, else -1."""
        for i, c in enumerate(self.table):
            if c.rank == target_rank:
                return i
        return -1

    def card_in_table(self, card):
        """Capture choice with Scopa priority rule:
        1) Prefer single equal-rank capture if available.
        2) Else try subset-sum combination to target rank.
        """
        # 1) single-card exact match priority
        si = self._single_equal_rank(card.rank)
        if si != -1:
            return True, [self.table[si]]

        # 2) subset-sum DP for combinations
        target = card.rank
        if target <= 0 or not self.table:
            return False, []

        ranks = [c.rank for c in self.table]
        comb_sums = [None] * (target + 1)
        comb_sums[0] = ()
        for idx, r in enumerate(ranks):
            for s in range(target, r - 1, -1):
                if comb_sums[s] is None and comb_sums[s - r] is not None:
                    comb_sums[s] = comb_sums[s - r] + (idx,)

        if comb_sums[target] is None:
            return False, []

        combo = [self.table[i] for i in comb_sums[target]]
        return True, combo

    def play_card(self, card, player):
        """Apply a move. Handles capture priority, Ace sweep, Scopa detection.
        NOTE: sets self.last_capture and self.last_scopa_player (if scopa)."""
        player.history.append(card)

        player_index = self.players.index(player)

        # Ace mechanics: capture entire table + played card
        if card.rank == 1:
            was_non_empty = len(self.table) > 0
            self.table.append(card)
            player.capture(self.table, _with=card)
            self.table = []
            self.last_capture = player_index
            if was_non_empty:
                # count scopa here (may be undone at hand end)
                player.scopas += 1
                self.last_scopa_player = player_index
                if self.tlogger is not None:
                    self.tlogger.scopa(player)
                if PRINT_DEBUG: print(f"\t!!! Scopa (Ace) for player side {player.side}")
            return

        # Non-ace: check capture(s)
        isin, comb = self.card_in_table(card)
        if isin:
            for c in comb:
                self.table.remove(c)
            comb.append(card)
            player.capture(comb, _with=card)
            self.last_capture = player_index
            if len(self.table) == 0:
                # count scopa here (may be undone at hand end)
                player.scopas += 1
                self.last_scopa_player = player_index
                if self.tlogger is not None:
                    self.tlogger.scopa(player)
                if PRINT_DEBUG: print(f"\t!!! Scopa for player side {player.side}")
        else:
            # No capture -> must place on table
            player.hand.remove(card)
            self.table.append(card)

    def evaluate_round(self):
        """Award leftover table to last capturer, then score."""
        # Award leftover table to last capturer (if any)
        if self.last_capture is not None and len(self.table) > 0:
            self.players[self.last_capture].captures.extend(self.table)
            self.table = []

        # Shared captures by team
        team1_captures = [card for player in self.players if player.side == 1 for card in player.captures]
        team2_captures = [card for player in self.players if player.side == 2 for card in player.captures]

        # Initialize points
        team1_points = 0
        team2_points = 0

        # Count Scopas (already tracked per player)
        team1_points += sum(player.scopas for player in self.players if player.side == 1)
        team2_points += sum(player.scopas for player in self.players if player.side == 2)

        # Most Cards
        if len(team1_captures) > len(team2_captures):
            team1_points += 1
        elif len(team2_captures) > len(team1_captures):
            team2_points += 1

        # Most Coins (denari == 'bello')
        team1_coins = [card for card in team1_captures if card.suit == 'bello']
        team2_coins = [card for card in team2_captures if card.suit == 'bello']
        if len(team1_coins) > len(team2_coins):
            team1_points += 1
        elif len(team2_coins) > len(team1_coins):
            team2_points += 1

        # Sette Bello (7 of denari)
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
        team1_best_cards = [max((card for card in team1_captures if card.suit == suit),
                                key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]
        team2_best_cards = [max((card for card in team2_captures if card.suit == suit),
                                key=lambda c: suit_priority.get(c.rank, 0), default=None) for suit in Deck.suits]

        team1_primiera = sum(suit_priority.get(card.rank, 0) for card in team1_best_cards if card)
        team2_primiera = sum(suit_priority.get(card.rank, 0) for card in team2_best_cards if card)

        if team1_primiera > team2_primiera:
            team1_points += 1
        elif team2_primiera > team1_primiera:
            team2_points += 1

        # Final reward: 1 / -1 / 0
        if team1_points > team2_points:
            return 1 , -1
        elif team2_points > team1_points:
            return -1, 1
        else:
            return 0, 0

    def __deepcopy__(self, memo):
        new_game = ScopaGame(logger=None)
        memo[id(self)] = new_game
        new_game.deck = self.deck
        new_players = []
        for p in self.players:
            np = Player(p.side, p.name)
            np.hand = list(p.hand)
            np.captures = list(p.captures)
            np.history = list(p.history)
            np.scopas = p.scopas
            new_players.append(np)
        new_game.players = new_players
        new_game.table = list(self.table)
        new_game.last_capture = self.last_capture
        new_game.last_scopa_player = self.last_scopa_player
        new_game.tlogger = None
        return new_game

# ...env(...) unchanged...

class MaScopaEnv(AECEnv):
    # metadata unchanged...

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

        self._action_spaces = {agent: spaces.Discrete(40) for agent in self.possible_agents}
        self._observation_spaces = {agent: spaces.Box(0, 1, shape=(6, 40), dtype=np.float32) for agent in self.possible_agents}

        self.reset()

    # observation_space/action_space/observe() unchanged...

    def reset(self, seed='42', options='43'):
        self.game.reset(seed=seed)
        self.num_moves = 0
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

    # get_action_mask unchanged...

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]

        # Find chosen card; validate legality
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
                break
        assert card is not None, f"Illegal action index {action} for agent {agent}"

        if PRINT_DEBUG:
            print(f"\n### Agent {agent} plays card: {card}")
            print(f"### Table state before play: {[card.__str__() for card in self.game.table]}")

        # Apply move
        self.game.play_card(card, player)

        if PRINT_DEBUG:
            print(f"### Table state after play: {[card.__str__() for card in self.game.table]}")

        # Check terminal (all hands empty)
        is_terminal = all(len(p.hand) == 0 for p in self.game.players)

        # If the move emptied the table AND it is the last move of the hand,
        # undo the scopa that was just counted (Scopa convention).
        if is_terminal and self.game.last_scopa_player is not None:
            lp = self.game.last_scopa_player
            self.game.players[lp].scopas = max(0, self.game.players[lp].scopas - 1)
            self.game.last_scopa_player = None  # clear flag

        if is_terminal:
            # Score the round
            round_scores = self.game.evaluate_round()
            self.archive_scores.append(round_scores)
            self.commulative_sides[0] += round_scores[0]
            self.commulative_sides[1] += round_scores[1]
            if self.tlogger is not None:
                self.tlogger.writer.add_scalar("Scores/Side/0", self.commulative_sides[0], self.tlogger.simulation_clock)
                self.tlogger.writer.add_scalar("Scores/Side/1", self.commulative_sides[1], self.tlogger.simulation_clock)
            if PRINT_DEBUG: print(f"### Round scores: {round_scores}")
            for i, ag in enumerate(self.possible_agents):
                self.rewards[ag] = round_scores[self.agent_name_mapping[ag] % 2]
            if PRINT_DEBUG: print(f"### Rewards after termination: {self.rewards}")
            self.terminations = {ag: True for ag in self.agents}

        if PRINT_DEBUG: print(f"### Observations updated for agent {agent}")
        self.num_moves += 1

        if self.num_moves >= NUM_ITERS:
            self.truncations = {a: True for a in self.agents}

        if PRINT_DEBUG: print(f"### Agent selection moved to next agent")
        self.agent_selection = self._agent_selector.next()

        if self.tlogger is not None:
            self.tlogger.record_step()
