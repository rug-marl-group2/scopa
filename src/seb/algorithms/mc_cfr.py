from typing import Any


import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from open_spiel.python import policy

@dataclass
class InfoNode:
    legal_actions: np.ndarray
    regret_sum: np.ndarray = None
    strategy_sum: np.ndarray = None

    def __post_init__(self):
        n = self.legal_actions.size
        self.regret_sum = np.zeros(n)
        self.strategy_sum = np.zeros(n)

    def current_strategy(self):
        pos = np.maximum(self.regret_sum, 0)
        if pos.sum() == 0:
            return np.ones_like(pos) / len(pos)
        return pos / pos.sum()


class MCCFRTrainer:
    def __init__(self, game):
        self.game = game
        self.info_sets = {}

    def _get_node(self, key, legal_actions):
        if key not in self.info_sets:
            self.info_sets[key] = InfoNode(np.array(legal_actions))
        return self.info_sets[key]

    def _sample(self, state, traversing_player, reach_probs, sampling_probs):
        if state.is_terminal():
            return state.rewards()[traversing_player]

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, probs = zip[tuple[Any, ...]](*outcomes)
            action = np.random.choice(acts, p=probs)
            next_state = state.clone()
            next_state.apply_action(action)
            return self._sample(next_state, traversing_player, reach_probs, sampling_probs)

        player = state.current_player()
        info_key = (player, state.information_state_string(player))
        legal = state.legal_actions(player)
        node = self._get_node(info_key, legal)

        sigma = node.current_strategy()
        action = np.random.choice(legal, p=sigma)
        next_state = state.clone()
        next_state.apply_action(action)

        new_reach = reach_probs.copy()
        new_sampling = sampling_probs.copy()
        if player == traversing_player:
            new_sampling[player] *= sigma[legal.index(action)]
        else:
            new_reach[player] *= sigma[legal.index(action)]
            new_sampling[player] *= sigma[legal.index(action)]

        util = self._sample(next_state, traversing_player, new_reach, new_sampling)

        if player == traversing_player:
            # Compute counterfactual regrets for all actions
            cfv_all = np.zeros(len(legal))
            for i, a in enumerate(legal):
                tmp_state = state.clone()
                tmp_state.apply_action(a)
                # When evaluating action a, update sampling prob for that branch
                tmp_sampling = sampling_probs.copy()
                tmp_sampling[player] *= sigma[i]
                cfv_all[i] = self._sample(tmp_state, traversing_player, reach_probs, tmp_sampling)
            v = np.dot(sigma, cfv_all)
            # Importance sampling weight: opponent reach / our sampling
            opp_reach = np.prod([reach_probs[p] for p in range(len(reach_probs)) if p != player])
            weight = opp_reach / sampling_probs[player] if sampling_probs[player] > 0 else 0
            node.regret_sum += weight * (cfv_all - v)
            node.strategy_sum += reach_probs[player] * sigma

        return util

    def iteration(self):
        """Run a single iteration of MCCFR (one pass for each player)."""
        for player in range(self.game.num_players()):
            s = self.game.new_initial_state()
            self._sample(s, player, np.ones(2), np.ones(2))
    
    def train(self, iterations=10000):
        history = []
        for t in tqdm(range(iterations), desc="MCCFR Training"):
            self.iteration()
        
        return history

    def tabular_policy(self):
        return ScopaLearnedPolicy(self.game, self.info_sets)

class ScopaLearnedPolicy(policy.Policy):
    def __init__(self, game, info_sets):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self.info_sets = info_sets
    
    def action_probabilities(self, state):
        if state.is_terminal():
            return {}
        
        player = state.current_player()
        info_state = state.information_state_string(player)
        key = (player, info_state)
        
        if key in self.info_sets:
            node = self.info_sets[key]
            total = node.strategy_sum.sum()
            if total > 1e-12:
                probs = node.strategy_sum / total
            else:
                probs = np.ones(len(node.legal_actions)) / len(node.legal_actions)
            
            return {action: probs[i] for i, action in enumerate(node.legal_actions)}
        else:
            legal = state.legal_actions(player)
            prob = 1.0 / len(legal)
            return {action: prob for action in legal}

class RandomPolicy(policy.Policy):
    """Policy that chooses actions uniformly at random."""
    def __init__(self, game):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)

    def action_probabilities(self, state):
        if state.is_terminal():
            return {}
        player = state.current_player()
        legal_actions = state.legal_actions(player)
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}

def evaluate_agent(game, trained_policy, opponent_policy, num_episodes=10000):
    num_players = game.num_players()
    if num_players != 2:
        raise ValueError("evaluate_agent only supports 2-player games")
    
    total_winnings = 0
    avg_reward_history = []
    trained_scopas = 0
    opponent_scopas = 0
    scopa_history = {'trained': [], 'opponent': [], 'diff': []}
    
    for episode in range(num_episodes):
        if episode < num_episodes / 2:
            agent_seat = 0
            policies = [trained_policy, opponent_policy]
        else:
            agent_seat = 1
            policies = [opponent_policy, trained_policy]
            
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, probs = zip(*outcomes)
                action = np.random.choice(acts, p=probs)
                state.apply_action(action)
            else:
                player = state.current_player()
                action_probs = policies[player].action_probabilities(state)
                actions, probs = zip(*action_probs.items())
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
        
        total_winnings += state.rewards()[agent_seat]
        avg_reward_history.append(total_winnings / (episode + 1))
        
        players = state.env.game.players
        agent_scopas = players[agent_seat].scopas
        opp_scopas = players[1 - agent_seat].scopas
        
        trained_scopas += agent_scopas
        opponent_scopas += opp_scopas
        
        scopa_history['trained'].append(trained_scopas / (episode + 1))
        scopa_history['opponent'].append(opponent_scopas / (episode + 1))
        scopa_history['diff'].append((trained_scopas - opponent_scopas) / (episode + 1))
           
        
    avg_reward = total_winnings / num_episodes
    avg_trained_scopas = trained_scopas / num_episodes
    avg_opponent_scopas = opponent_scopas / num_episodes
    
    scopa_stats = {
        'trained_avg': avg_trained_scopas,
        'opponent_avg': avg_opponent_scopas,
        'difference': avg_trained_scopas - avg_opponent_scopas,
        'history': scopa_history,
        'data_collected': len(scopa_history['trained']) > 0
    }
    
    return avg_reward, avg_reward_history, scopa_stats
