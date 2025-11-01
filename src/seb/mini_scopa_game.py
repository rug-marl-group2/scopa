import random
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}_of_{self.suit}"


class MiniDeck:
    """16-card deck: 4 suits 4 ranks each, pairwise duplicated ranks across suits."""
    suits = ["cuori", "fiori", "picche", "bello"]
    ranks = {
        "cuori": [2, 5, 8, 10],
        "fiori": [2, 5, 7, 9],
        "picche": [3, 6, 8, 9],
        "bello": [3, 6, 7, 10],
    }

    def __init__(self, seed=42):
        self.cards = [Card(r, s) for s in self.suits for r in self.ranks[s]]
        random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, n):
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt


class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.captures = []
        self.scopas = 0

    def reset(self):
        self.hand.clear()
        self.captures.clear()
        self.scopas = 0

class MiniScopaGame:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.deck = MiniDeck()
        self.players = [Player(f"player_{i}") for i in range(num_players)]
        self.table = []
        self.last_capture = None

    def reset(self, seed=42):
        self.deck = MiniDeck(seed)
        self.table.clear()
        cards_per_player = 4  # Fixed: 4 cards per player
        for p in self.players:
            p.reset()
            p.hand = self.deck.deal(cards_per_player)
        # self.table = self.deck.deal(4) #for now no card on the table
        self.last_capture = None

    def card_in_table(self, card):
        """Find subset of table cards that sum to card's rank (proper Scopa rule)."""
        target = card.rank
        if target <= 0 or not self.table:
            return False, []
        
        # First check for exact rank match (preferred in Scopa)
        exact_matches = [c for c in self.table if c.rank == target]
        if exact_matches:
            return True, [exact_matches[0]]  # Take first exact match
        
        ranks = [c.rank for c in self.table]
        # comb_sums[s] stores tuple of indices whose ranks sum to s
        comb_sums = [None] * (target + 1)
        comb_sums[0] = ()
        
        for idx, r in enumerate(ranks):
            # Iterate in descending order to avoid reusing same card
            for s in range(target, r - 1, -1):
                if comb_sums[s] is None and comb_sums[s - r] is not None:
                    comb_sums[s] = comb_sums[s - r] + (idx,)
        
        if comb_sums[target] is None:
            return False, []
        
        combo = [self.table[i] for i in comb_sums[target]]
        return True, combo

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

    def evaluate_game(self):
        """Final reward as zero-sum vector (+1 per capture, +2 per scopa)."""
        rewards = [len(p.captures) + 2 * p.scopas for p in self.players]
        total = sum(rewards)
        if total == 0:
            return [0] * self.num_players
        # normalize zero-sum around mean
        mean = total / self.num_players
        return [r - mean for r in rewards]


class MiniScopaEnv(AECEnv):
    metadata = {"name": "Mini-Scopa-v0"}

    def __init__(self, seed=42, num_players=2):
        super().__init__()
        self.num_players = num_players
        self.game = MiniScopaGame(num_players=num_players)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self._action_spaces = {a: spaces.Discrete(16) for a in self.possible_agents}
        self.max_steps = num_players * 4  # 4 cards per player
        self.seed = seed
        self.reset(seed)

    def reset(self, seed=None):
        self.game.reset(seed or self.seed)
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agents[0]
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.step_count = 0

    def step(self, action):
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player = self.game.players[self.agent_name_mapping[agent]]

        # Convert action (0..15) to Card via fixed mapping
        suit_idx = action // 4
        card_idx = action % 4
        suits = MiniDeck.suits
        suit = suits[suit_idx]
        rank = MiniDeck.ranks[suit][card_idx]

        card = next((c for c in player.hand if c.rank == rank and c.suit == suit), None)
        if card:
            self.game.play_card(card, player)

        self.step_count += 1
        is_terminal = all(len(p.hand) == 0 for p in self.game.players) or self.step_count >= self.max_steps
        if is_terminal:
            r = self.game.evaluate_game()
            for i, a in enumerate(self.agents):
                self.rewards[a] = r[i]
                self.terminations[a] = True

        self.agent_selection = self.agents[(self.agents.index(agent) + 1) % self.num_players]

    def get_state(self):
        return {
            "table": [(c.rank, c.suit) for c in self.game.table],
            "hands": [[(c.rank, c.suit) for c in p.hand] for p in self.game.players],
            "captures": [[(c.rank, c.suit) for c in p.captures] for p in self.game.players],
            "scopas": [p.scopas for p in self.game.players],
            "agent_selection": self.agent_selection,
            "step_count": self.step_count,
            "agents": self.agents[:],
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "truncations": dict(self.truncations),
        }

    def set_state(self, state):
        self.game.table = [Card(r, s) for r, s in state["table"]]
        for i, p in enumerate(self.game.players):
            p.hand = [Card(r, s) for r, s in state["hands"][i]]
            p.captures = [Card(r, s) for r, s in state["captures"][i]]
            p.scopas = state["scopas"][i]
        self.agent_selection = state["agent_selection"]
        self.step_count = state["step_count"]
        self.agents = state["agents"][:]
        self.rewards = dict(state["rewards"])
        self.terminations = dict(state["terminations"])
        self.truncations = dict(state["truncations"])

