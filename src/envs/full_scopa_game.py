import random
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}_{self.suit}"
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))


class FullDeck:
    """Standard Italian 40-card deck for Scopa"""
    suits = ["denari", "coppe", "spade", "bastoni"]  # diamonds, cups, swords, clubs
    ranks = list(range(1, 11))  # 1-10 for each suit
    
    # Primiera values for scoring (traditional Italian Scopa values)
    primiera_values = {
        7: 21, 6: 18, 1: 16, 5: 15, 4: 14, 3: 13, 2: 12,
        10: 10, 9: 10, 8: 10
    }

    def __init__(self, seed=42):
        self.cards = [Card(r, s) for s in self.suits for r in self.ranks]
        random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, n):
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt
    
    def cards_remaining(self):
        return len(self.cards)


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


class FullScopaGame:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.deck = FullDeck()
        self.players = [Player(f"player_{i}") for i in range(num_players)]
        self.table = []
        self.last_capture = None
        self.round_number = 0
        self.cards_per_hand = 3  # Standard Scopa: 3 cards per hand

    def reset(self, seed=42):
        self.deck = FullDeck(seed)
        self.table.clear()
        self.round_number = 0
        
        # Reset players
        for p in self.players:
            p.reset()
        
        # Deal initial 4 cards to table
        self.table = self.deck.deal(4)
        
        # Deal 3 cards to each player
        for p in self.players:
            p.hand = self.deck.deal(self.cards_per_hand)
        
        self.last_capture = None

    def can_deal_new_round(self):
        """Check if there are enough cards to deal another round"""
        return self.deck.cards_remaining() >= self.num_players * self.cards_per_hand
    
    def deal_new_round(self):
        """Deal 3 new cards to each player"""
        if self.can_deal_new_round():
            for p in self.players:
                p.hand = self.deck.deal(self.cards_per_hand)
            self.round_number += 1
            return True
        return False

    def find_capture_combinations(self, card):
        """Find ALL possible subsets of table cards that sum to card's rank"""
        target = card.rank
        if target <= 0 or not self.table:
            return []
        
        # First check for exact rank match (highest priority in Scopa)
        exact_matches = [c for c in self.table if c.rank == target]
        if exact_matches:
            return [[exact_matches[0]]]
        
        # Find all subset sums using dynamic programming
        combinations = []
        n = len(self.table)
        
        # Use bit manipulation to check all subsets
        for mask in range(1, 1 << n):
            subset = []
            subset_sum = 0
            for i in range(n):
                if mask & (1 << i):
                    subset.append(self.table[i])
                    subset_sum += self.table[i].rank
            
            if subset_sum == target:
                combinations.append(subset)
        
        return combinations

    def play_card(self, card, player, capture_choice=None):
        """
        Play a card. If multiple capture options exist, use capture_choice.
        capture_choice is an index into the list of possible combinations.
        """
        combinations = self.find_capture_combinations(card)
        
        if combinations:
            # Choose which combination to capture
            if capture_choice is None or capture_choice >= len(combinations):
                captured = combinations[0]  # Default to first combination
            else:
                captured = combinations[capture_choice]
            
            # Remove captured cards from table
            for c in captured:
                self.table.remove(c)
            
            # Add to player's captures
            player.captures.extend(captured + [card])
            self.last_capture = player
            
            # Check for scopa (cleared the table)
            if not self.table:
                player.scopas += 1
        else:
            # No capture possible, card goes to table
            self.table.append(card)
        
        # Remove card from hand
        player.hand.remove(card)

    def calculate_primiera_score(self, captured_cards):
        """Calculate primiera score for a player's captured cards"""
        best_per_suit = {}
        
        for card in captured_cards:
            value = FullDeck.primiera_values[card.rank]
            if card.suit not in best_per_suit or value > best_per_suit[card.suit][1]:
                best_per_suit[card.suit] = (card, value)
        
        # Player needs at least one card from each suit to score primiera
        if len(best_per_suit) == 4:
            return sum(val for _, val in best_per_suit.values())
        return 0

    def evaluate_game(self):
        """
        Calculate final scores using traditional Scopa scoring:
        - 1 point for most cards (carte)
        - 1 point for most denari/diamonds (denari)
        - 1 point for 7 of denari (sette bello)
        - 1 point for best primiera
        - 1 point per scopa
        """
        scores = [0] * self.num_players
        
        # Award remaining table cards to last capturer
        if self.table and self.last_capture:
            self.last_capture.captures.extend(self.table)
        
        # Count cards
        card_counts = [len(p.captures) for p in self.players]
        max_cards = max(card_counts)
        winners = [i for i, c in enumerate(card_counts) if c == max_cards]
        if len(winners) == 1:  # No tie
            scores[winners[0]] += 1
        
        # Count denari (diamonds)
        denari_counts = [sum(1 for c in p.captures if c.suit == "denari") for p in self.players]
        max_denari = max(denari_counts)
        winners = [i for i, c in enumerate(denari_counts) if c == max_denari]
        if len(winners) == 1:  # No tie
            scores[winners[0]] += 1
        
        # Sette bello (7 of denari)
        sette_bello = Card(7, "denari")
        for i, p in enumerate(self.players):
            if sette_bello in p.captures:
                scores[i] += 1
                break
        
        # Primiera
        primiera_scores = [self.calculate_primiera_score(p.captures) for p in self.players]
        if any(s > 0 for s in primiera_scores):  # At least one player has valid primiera
            max_primiera = max(primiera_scores)
            winners = [i for i, s in enumerate(primiera_scores) if s == max_primiera and s > 0]
            if len(winners) == 1:  # No tie
                scores[winners[0]] += 1
        
        # Scopas
        for i, p in enumerate(self.players):
            scores[i] += p.scopas
        
        # Convert to zero-sum rewards
        total = sum(scores)
        if total == 0:
            return [0] * self.num_players
        mean = total / self.num_players
        return [s - mean for s in scores]


class FullScopaEnv(AECEnv):
    metadata = {"name": "Full-Scopa-v0"}

    def __init__(self, seed=42, num_players=2):
        super().__init__()
        self.num_players = num_players
        self.game = FullScopaGame(num_players=num_players)
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        # 40 possible cards (action space)
        self._action_spaces = {a: spaces.Discrete(40) for a in self.possible_agents}
        self.max_steps = 200  # Safety limit (40 cards / 2 players = 20 turns each max, with margin)
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

        # Convert action (0..39) to Card via fixed mapping
        suit_idx = action // 10
        rank = (action % 10) + 1  # ranks are 1-10
        suits = FullDeck.suits
        suit = suits[suit_idx]

        # Find the card in player's hand
        card = next((c for c in player.hand if c.rank == rank and c.suit == suit), None)
        if card:
            self.game.play_card(card, player)

        self.step_count += 1
        
        # Check if all players need new cards
        all_hands_empty = all(len(p.hand) == 0 for p in self.game.players)
        if all_hands_empty:
            if self.game.can_deal_new_round():
                self.game.deal_new_round()
            else:
                # Game over - no more cards to deal
                r = self.game.evaluate_game()
                for i, a in enumerate(self.agents):
                    self.rewards[a] = r[i]
                    self.terminations[a] = True
        
        # Safety check for max steps
        if self.step_count >= self.max_steps:
            r = self.game.evaluate_game()
            for i, a in enumerate(self.agents):
                self.rewards[a] = r[i]
                self.terminations[a] = True

        # Move to next agent
        self.agent_selection = self.agents[(self.agents.index(agent) + 1) % self.num_players]

    def get_state(self):
        return {
            "table": [(c.rank, c.suit) for c in self.game.table],
            "hands": [[(c.rank, c.suit) for c in p.hand] for p in self.game.players],
            "captures": [[(c.rank, c.suit) for c in p.captures] for p in self.game.players],
            "scopas": [p.scopas for p in self.game.players],
            "deck_remaining": self.game.deck.cards_remaining(),
            "round_number": self.game.round_number,
            "last_capture": self.game.players.index(self.game.last_capture) if self.game.last_capture else None,
            "agent_selection": self.agent_selection,
            "step_count": self.step_count,
            "agents": self.agents[:],
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "truncations": dict(self.truncations),
        }

    def set_state(self, state):
        # Reconstruct deck
        self.game.deck = FullDeck()
        # Remove dealt cards (simplified - in practice would need full deck state)
        dealt_count = 40 - state["deck_remaining"]
        self.game.deck.cards = self.game.deck.cards[dealt_count:]
        
        # Restore game state
        self.game.table = [Card(r, s) for r, s in state["table"]]
        for i, p in enumerate(self.game.players):
            p.hand = [Card(r, s) for r, s in state["hands"][i]]
            p.captures = [Card(r, s) for r, s in state["captures"][i]]
            p.scopas = state["scopas"][i]
        
        self.game.round_number = state["round_number"]
        if state["last_capture"] is not None:
            self.game.last_capture = self.game.players[state["last_capture"]]
        else:
            self.game.last_capture = None
        
        # Restore environment state
        self.agent_selection = state["agent_selection"]
        self.step_count = state["step_count"]
        self.agents = state["agents"][:]
        self.rewards = dict(state["rewards"])
        self.terminations = dict(state["terminations"])
        self.truncations = dict(state["truncations"])

