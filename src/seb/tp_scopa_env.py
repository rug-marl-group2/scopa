import random
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

NUM_ITERS = 100

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        rank_name = {8: "Jack", 9: "Queen", 10: "King"}.get(self.rank, str(self.rank))
        return f"{rank_name} di {self.suit}"

class Deck:
    suits = ['picche', 'bello', 'fiori', 'cuori']
    ranks = list(range(1, 11))

    def __init__(self):
        self.cards = [Card(r, s) for s in self.suits for r in self.ranks]
        self.shuffle()

    def shuffle(self, seed=42):
        random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, n):
        return [self.cards.pop() for _ in range(n)]

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.captures = []
        self.scopas = 0

    def reset(self):
        self.hand, self.captures = [], []
        self.scopas = 0

class Scopa2PGame:
    def __init__(self):
        self.deck = Deck()
        self.players = [Player("player_0"), Player("player_1")]
        self.table = []
        self.last_capture = None

    def reset(self, seed=42):
        self.deck.shuffle(seed)
        self.table = self.deck.deal(4)
        for p in self.players:
            p.reset()
            p.hand = self.deck.deal(3)

    def card_in_table(self, card):
        """Simple capture logic."""
        for c in self.table:
            if c.rank == card.rank:
                return True, [c]
        return False, []

    def play_card(self, card, player):
        isin, captured = self.card_in_table(card)
        if isin:
            for c in captured:
                self.table.remove(c)
            player.captures.extend(captured + [card])
            self.last_capture = player
            if not self.table:
                player.scopas += 1
        else:
            self.table.append(card)
            player.hand.remove(card)

    def evaluate_round(self):
        p1 = self.players[0]
        p2 = self.players[1]
        p1_cards = len(p1.captures)
        p2_cards = len(p2.captures)
        if p1_cards > p2_cards:
            return 1, -1
        elif p2_cards > p1_cards:
            return -1, 1
        else:
            return 0, 0

class TwoPlayerScopaEnv(AECEnv):
    def __init__(self):
        super().__init__()
        self.game = Scopa2PGame()
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {"player_0": 0, "player_1": 1}
        self._action_spaces = {a: spaces.Discrete(40) for a in self.possible_agents}
        self._observation_spaces = {a: spaces.Box(0, 1, shape=(6, 40), dtype=np.float32) for a in self.possible_agents}
        self.reset()

    def reset(self, seed=42):
        self.game.reset(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.agent_selection = self.agents[0]

    def step(self, action):
        agent = self.agent_selection
        player = self.game.players[self.agent_name_mapping[agent]]
        rank = (action % 10) + 1
        suit = ['cuori', 'picche', 'fiori', 'bello'][action // 10]
        card = next((c for c in player.hand if c.rank == rank and c.suit == suit), None)
        if card:
            self.game.play_card(card, player)
        is_terminal = all(len(p.hand) == 0 for p in self.game.players)
        if is_terminal:
            r1, r2 = self.game.evaluate_round()
            self.rewards = {"player_0": r1, "player_1": r2}
            self.terminations = {a: True for a in self.agents}
        self.agent_selection = self.agents[(self.agents.index(agent) + 1) % 2]
