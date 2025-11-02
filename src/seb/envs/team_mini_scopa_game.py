"""
Team Mini Scopa - 2v2 Adversarial Team Game

Implementation following "A Marriage between Adversarial Team Games and 2-player Games"
by Carminati et al. (ICML 2022).

Game Structure:
- 4 players divided into 2 teams of 2
- Team 0: Players 0 and 1
- Team 1: Players 2 and 3
- 16-card deck (4 suits × 4 ranks)
- Each player gets 4 cards
- Teams coordinate ex ante (before game starts)
- No communication during play
- Asymmetric information within teams (players don't see teammate's hand)
"""

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


class MiniDeck:
    """16-card deck: 4 suits 4 ranks each"""
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
    def __init__(self, name, team_id):
        self.name = name
        self.team_id = team_id  # 0 or 1
        self.hand = []
        self.captures = []
        self.scopas = 0

    def reset(self):
        self.hand.clear()
        self.captures.clear()
        self.scopas = 0


class TeamMiniScopaGame:
    """
    2v2 Team Mini Scopa Game
    
    Key properties for adversarial team game framework:
    - Ex ante coordination: Teams decide strategy before play
    - Asymmetric information: Team members see own hand only
    - Zero-sum: Team 0 vs Team 1
    - Sequential: Players alternate
    """
    
    def __init__(self):
        self.deck = MiniDeck()
        self.players = [
            Player("player_0", team_id=0),  # Team 0
            Player("player_1", team_id=0),  # Team 0
            Player("player_2", team_id=1),  # Team 1
            Player("player_3", team_id=1),  # Team 1
        ]
        self.table = []
        self.last_capture_team = None

    def reset(self, seed=42):
        self.deck = MiniDeck(seed)
        self.table.clear()
        
        # Deal 4 cards to each player
        for p in self.players:
            p.reset()
            p.hand = self.deck.deal(4)
        
        self.last_capture_team = None

    def get_team(self, player_id):
        """Get team ID for a player"""
        return self.players[player_id].team_id

    def card_in_table(self, card):
        """Find subset of table cards that sum to card's rank"""
        target = card.rank
        if target <= 0 or not self.table:
            return False, []
        
        # First check for exact rank match (preferred in Scopa)
        exact_matches = [c for c in self.table if c.rank == target]
        if exact_matches:
            return True, [exact_matches[0]]
        
        # Use DP to find subset sum
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
        """Play a card for a player"""
        isin, captured = self.card_in_table(card)
        if isin:
            for c in captured:
                self.table.remove(c)
            player.captures.extend(captured + [card])
            self.last_capture_team = player.team_id
            if not self.table:
                player.scopas += 1
        else:
            self.table.append(card)
        player.hand.remove(card)

    def evaluate_game(self):
        """
        Return zero-sum rewards for teams
        Team scoring: captures + 2 * scopas per player
        """
        team_scores = [0, 0]
        
        # Award remaining table cards to last capturing team
        if self.table and self.last_capture_team is not None:
            # Find any player from last capturing team
            for p in self.players:
                if p.team_id == self.last_capture_team:
                    p.captures.extend(self.table)
                    break
        
        # Calculate scores per team
        for p in self.players:
            team_scores[p.team_id] += len(p.captures) + 2 * p.scopas
        
        # Zero-sum normalization
        total = sum(team_scores)
        if total == 0:
            return [0, 0, 0, 0]
        
        mean = total / 2
        team_0_reward = team_scores[0] - mean
        team_1_reward = team_scores[1] - mean
        
        # Return per-player rewards (same for teammates)
        return [team_0_reward, team_0_reward, team_1_reward, team_1_reward]


class TeamMiniScopaEnv(AECEnv):
    """PettingZoo environment for Team Mini Scopa"""
    metadata = {"name": "Team-Mini-Scopa-v0"}

    def __init__(self, seed=42):
        super().__init__()
        self.game = TeamMiniScopaGame()
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self._action_spaces = {a: spaces.Discrete(16) for a in self.possible_agents}
        self.max_steps = 16  # 4 players × 4 cards
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

        # Convert action (0..15) to Card
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

        # Move to next agent
        self.agent_selection = self.agents[(self.agents.index(agent) + 1) % 4]

    def get_state(self):
        """Get complete game state for cloning"""
        return {
            "table": [(c.rank, c.suit) for c in self.game.table],
            "hands": [[(c.rank, c.suit) for c in p.hand] for p in self.game.players],
            "captures": [[(c.rank, c.suit) for c in p.captures] for p in self.game.players],
            "scopas": [p.scopas for p in self.game.players],
            "last_capture_team": self.game.last_capture_team,
            "agent_selection": self.agent_selection,
            "step_count": self.step_count,
            "agents": self.agents[:],
            "rewards": dict(self.rewards),
            "terminations": dict(self.terminations),
            "truncations": dict(self.truncations),
        }

    def set_state(self, state):
        """Restore game state from dict"""
        self.game.table = [Card(r, s) for r, s in state["table"]]
        for i, p in enumerate(self.game.players):
            p.hand = [Card(r, s) for r, s in state["hands"][i]]
            p.captures = [Card(r, s) for r, s in state["captures"][i]]
            p.scopas = state["scopas"][i]
        
        self.game.last_capture_team = state["last_capture_team"]
        self.agent_selection = state["agent_selection"]
        self.step_count = state["step_count"]
        self.agents = state["agents"][:]
        self.rewards = dict(state["rewards"])
        self.terminations = dict(state["terminations"])
        self.truncations = dict(state["truncations"])

