from typing import Dict, Optional, Tuple

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

# PettingZoo changed agent_selector export; support both styles
try:
    from pettingzoo.utils.agent_selector import agent_selector as AgentSelector
except Exception:  # pragma: no cover
    from pettingzoo.utils import agent_selector as _agent_selector_mod

    AgentSelector = _agent_selector_mod.agent_selector

import copy
import itertools
import random

import numpy as np
from gymnasium import spaces

from src.tlogger import TLogger  # optional logger

NUM_ITERS = 100  # Number of iterations before truncation
PRINT_DEBUG = False


class Card:
    def __init__(self, rank: int, suit: str):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        # pretty-print face cards
        rank_raster = self.rank
        if self.rank == 10:
            rank_raster = "King"
        elif self.rank == 9:
            rank_raster = "Queen"
        elif self.rank == 8:
            rank_raster = "Jack"

        if self.suit == "bello":
            return f"{rank_raster} {self.suit}"
        else:
            return f"{rank_raster} di {self.suit}"


class Deck:
    suits = ["picche", "bello", "fiori", "cuori"]
    ranks = list(range(1, 11))  # 1..7 plus 8=Jack, 9=Queen, 10=King

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()
        self.cards = [Card(rank, suit) for suit in self.suits for rank in self.ranks]
        self.shuffle()

    def shuffle(self):
        # use local RNG, avoid global seeding
        self._rng.shuffle(self.cards)

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

    def capture(self, cards, _with=None):
        self.captures.extend(cards)
        if _with is not None and _with in self.hand:
            self.hand.remove(_with)

    def play_card(self, card_index):
        return self.hand.pop(card_index)


class ScopaGame:
    def __init__(self, logger: Optional[TLogger]):
        self.rng = random.Random()
        self.deck = Deck(self.rng)
        self.players = [
            Player(1, "player_0"),
            Player(2, "player_1"),
            Player(1, "player_2"),
            Player(2, "player_3"),
        ]
        self.table = []
        self.last_capture = None
        self.tlogger = logger

    def reset(self, seed: Optional[int] = None):
        # local RNG (no global pollution)
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)  # deck shuffles on init with local RNG
        self.table = []
        self.last_capture = None
        for player in self.players:
            player.reset()
        for player in self.players:
            player.hand = self.deck.deal(10)

    def card_in_table(self, card: Card):
        # Find a subset of table cards whose ranks sum to the played card's rank.
        # DP in O(n * target)
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

    def play_card(self, card: Card, player: Player):
        player.history.append(card)
        player_index = self.players.index(player)

        # Ace: captures entire table
        if card.rank == 1:
            if PRINT_DEBUG:
                print(f"\t!!! Ace == {card} for player {[str(c) for c in player.hand]}")
            self.table.append(card)
            player.capture(self.table, _with=card)
            self.table = []
            self.last_capture = player_index
            return

        # Non-ace capture via subset sum
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
                    try:
                        self.tlogger.scopa(player)
                    except Exception:
                        pass
                if PRINT_DEBUG:
                    print(f"\t!!! Scopa for player {player.side}")
        else:
            # just place the card on the table
            if card in player.hand:
                player.hand.remove(card)
            self.table.append(card)

    def evaluate_round(self):
        if self.table and self.last_capture is not None:
            capturing_player = self.players[self.last_capture]
            capturing_player.capture(self.table)
            self.table = []
            self.last_capture = None

        # Team captures
        team1_captures = [
            card
            for player in self.players
            if player.side == 1
            for card in player.captures
        ]
        team2_captures = [
            card
            for player in self.players
            if player.side == 2
            for card in player.captures
        ]

        team1_points = 0
        team2_points = 0

        # Scopas
        team1_points += sum(
            player.scopas for player in self.players if player.side == 1
        )
        team2_points += sum(
            player.scopas for player in self.players if player.side == 2
        )

        # Most Cards
        if len(team1_captures) > len(team2_captures):
            team1_points += 1
        elif len(team2_captures) > len(team1_captures):
            team2_points += 1

        # Most Coins ("ori") -> suit 'bello'
        team1_coins = [card for card in team1_captures if card.suit == "bello"]
        team2_coins = [card for card in team2_captures if card.suit == "bello"]
        if len(team1_coins) > len(team2_coins):
            team1_points += 1
        elif len(team2_coins) > len(team1_coins):
            team2_points += 1

        # Sette Bello (7 of Coins)
        if any((c.rank == 7 and c.suit == "bello") for c in team1_captures):
            team1_points += 1
        if any((c.rank == 7 and c.suit == "bello") for c in team2_captures):
            team2_points += 1

        # Primiera
        suit_priority = {7: 4, 6: 3, 1: 2, 5: 1, 4: 0, 3: 0, 2: 0}
        team1_best = [
            max(
                (c for c in team1_captures if c.suit == s),
                key=lambda c: suit_priority.get(c.rank, 0),
                default=None,
            )
            for s in Deck.suits
        ]
        team2_best = [
            max(
                (c for c in team2_captures if c.suit == s),
                key=lambda c: suit_priority.get(c.rank, 0),
                default=None,
            )
            for s in Deck.suits
        ]
        team1_prim = sum(suit_priority.get(c.rank, 0) for c in team1_best if c)
        team2_prim = sum(suit_priority.get(c.rank, 0) for c in team2_best if c)
        if team1_prim > team2_prim:
            team1_points += 1
        elif team2_prim > team1_prim:
            team2_points += 1

        # Return per-team outcome in {-1,0,1}
        if team1_points > team2_points:
            return 1, -1
        elif team2_points > team1_points:
            return -1, 1
        else:
            return 0, 0

    def __deepcopy__(self, memo):
        # Lightweight copy for branching algorithms
        new_game = ScopaGame(logger=None)
        memo[id(self)] = new_game
        # Share RNG/deck (deck unused after reset)
        new_game.rng = self.rng
        new_game.deck = self.deck
        # Copy players shallowly
        new_players = []
        for p in self.players:
            np_ = Player(p.side, p.name)
            np_.hand = list(p.hand)
            np_.captures = list(p.captures)
            np_.history = list(p.history)
            np_.scopas = p.scopas
            new_players.append(np_)
        new_game.players = new_players
        new_game.table = list(self.table)
        new_game.last_capture = self.last_capture
        new_game.tlogger = None
        return new_game


def env(tlogger: Optional[TLogger], render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    e = MaScopaEnv(render_mode=internal_render_mode, tlogger=tlogger)
    if render_mode == "ansi":
        e = wrappers.CaptureStdoutWrapper(e)
    return e


class MaScopaEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "scopa_v0",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, tlogger: Optional[TLogger] = None):
        super().__init__()
        self.render_mode = render_mode
        self.tlogger = tlogger
        self.cumulative_sides = [0, 0]
        self.archive_scores = []

        # suit indexing (shared everywhere)
        self._suit_offset = {
            "cuori": 0,
            "picche": 10,
            "fiori": 20,
            "bello": 30,
        }

        self.game = ScopaGame(logger=self.tlogger)
        self.possible_agents = [p.name for p in self.game.players]
        self.agent_name_mapping = {
            agent: int(agent[-1]) for agent in self.possible_agents
        }

        players = len(self.game.players)
        # hands, captures, history are per-player; table is once; plus scopas & current one-hot
        self.global_state_dim = (players * 3 * 40) + 40 + (players * 2)

        self._action_spaces = {
            agent: spaces.Discrete(40) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(0, 1, shape=(4, 40), dtype=np.float32)
            for agent in self.possible_agents
        }

        # PettingZoo/Gymnasium-conform reset
        self._rng = random.Random()
        self.reset()

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _encode_cards(self, cards):
        vec = np.zeros(40, dtype=np.float32)
        off = self._suit_offset
        for card in cards:
            index = (card.rank - 1) + off[card.suit]
            vec[index] += 1.0
        return vec

    def get_global_state(self):
        """Return a flat global state vector for centralized critics."""
        hands = [self._encode_cards(p.hand) for p in self.game.players]
        table = self._encode_cards(self.game.table)
        captures = [self._encode_cards(p.captures) for p in self.game.players]
        history = [self._encode_cards(p.history) for p in self.game.players]
        scopas = np.asarray([p.scopas for p in self.game.players], dtype=np.float32)
        current = np.zeros(len(self.game.players), dtype=np.float32)
        if getattr(self, "agent_selection", None) in getattr(
            self, "agent_name_mapping", {}
        ):
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
        off = self._suit_offset

        for card in player.hand:
            state[0][(card.rank - 1) + off[card.suit]] = 1.0
        for card in self.game.table:
            state[1][(card.rank - 1) + off[card.suit]] = 1.0
        for card in player.captures:
            state[2][(card.rank - 1) + off[card.suit]] = 1.0
        for card in friend.captures:
            state[2][(card.rank - 1) + off[card.suit]] = 1.0
        for card in friend.history:
            state[3][(card.rank - 1) + off[card.suit]] = 1.0

        return state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # local RNG
        self._rng = random.Random(seed)
        self.game.reset(seed=seed)
        self.num_moves = 0

        # Randomize the starting player (fairness)
        randstart = self._rng.randint(0, 3)
        self.possible_agents = (
            self.possible_agents[randstart:] + self.possible_agents[:randstart]
        )
        self.agents = self.possible_agents[:]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {
            agent: {"action_mask": self.get_action_mask(agent)} for agent in self.agents
        }
        self.observations = {agent: self.observe(agent) for agent in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        return self.observations, self.infos

    def get_action_mask(self, agent="current"):
        off = self._suit_offset
        if agent == "current":
            action_mask = np.zeros(40, dtype=int)
            cur_idx = self.agent_name_mapping[self.agent_selection]
            for card in self.game.players[cur_idx].hand:
                action_mask[(card.rank - 1) + off[card.suit]] = 1
            return action_mask

        elif agent is None:
            action_mask = np.zeros(160, dtype=int)
            for t, player in enumerate(self.game.players):
                for card in player.hand:
                    action_mask[(t * 40) + (card.rank - 1) + off[card.suit]] = 1
            return action_mask

        else:
            action_mask = np.zeros(40, dtype=int)
            player_index = self.agent_name_mapping[agent]
            for card in self.game.players[player_index].hand:
                action_mask[(card.rank - 1) + off[card.suit]] = 1
            return action_mask

    def _map_action_to_card(self, player, action: int):
        """Map discrete action index to the actual Card object in player's hand (or None)."""
        off = self._suit_offset
        for c in player.hand:
            ind = (c.rank - 1) + off[c.suit]
            if ind == action:
                return c
        return None

    def step(self, action: int):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_index = self.agent_name_mapping[agent]
        player = self.game.players[player_index]

        # Enforce legal action mask
        mask = self.infos[agent].get("action_mask", self.get_action_mask(agent))
        illegal = not (0 <= action < mask.size and mask[action] == 1)
        if illegal:
            # fallback to a random legal action to keep env stable
            legal_actions = np.nonzero(mask)[0]
            if legal_actions.size == 0:
                # no legal action (shouldn't happen) -> dead step
                self._was_dead_step(action)
                return
            action = int(self._rng.choice(list(legal_actions)))
            self.infos[agent]["illegal_action"] = True

        card = self._map_action_to_card(player, action)
        if card is None:
            # defensive fallback (shouldn't happen now)
            legal_actions = np.nonzero(mask)[0]
            action = int(self._rng.choice(list(legal_actions)))
            card = self._map_action_to_card(player, action)

        if PRINT_DEBUG:
            print(f"\n### Agent {agent} plays card: {card}")
            print(f"### Table before: {[str(c) for c in self.game.table]}")

        self.game.play_card(card, player)

        # Update per-agent observations & masks
        for name in self.agents:
            self.observations[name] = self.observe(name)
            if name not in self.infos:
                self.infos[name] = {}
            self.infos[name]["action_mask"] = self.get_action_mask(name)

        if PRINT_DEBUG:
            print(f"### Table after: {[str(c) for c in self.game.table]}")

        # End of episode if all hands empty
        if all(len(p.hand) == 0 for p in self.game.players):
            round_scores = self.game.evaluate_round()
            self.archive_scores.append(round_scores)
            self.cumulative_sides[0] += round_scores[0]
            self.cumulative_sides[1] += round_scores[1]

            if self.tlogger is not None and hasattr(self.tlogger, "writer"):
                try:
                    self.tlogger.writer.add_scalar(
                        "Scores/Side/0",
                        self.cumulative_sides[0],
                        getattr(self.tlogger, "simulation_clock", 0),
                    )
                    self.tlogger.writer.add_scalar(
                        "Scores/Side/1",
                        self.cumulative_sides[1],
                        getattr(self.tlogger, "simulation_clock", 0),
                    )
                except Exception:
                    pass

            if PRINT_DEBUG:
                print(f"### Round scores: {round_scores}")

            # Team reward per agent
            for i, ag in enumerate(self.possible_agents):
                self.rewards[ag] = round_scores[self.agent_name_mapping[ag] % 2]

            if PRINT_DEBUG:
                print(f"### Rewards after termination: {self.rewards}")

            self.terminations = {ag: True for ag in self.agents}

        self.num_moves += 1
        if self.num_moves >= NUM_ITERS:
            self.truncations = {a: True for a in self.agents}

        # Next agent
        self.agent_selection = self._agent_selector.next()

        if self.tlogger is not None:
            try:
                self.tlogger.record_step()
            except Exception:
                pass

    def roundScores(self):
        return self.archive_scores

    def render(self):
        if self.render_mode == "human":
            print([str(c) for c in self.game.table])
