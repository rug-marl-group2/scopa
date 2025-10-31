import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
import pyspiel
import openspiel_scopa 
import matplotlib.pyplot as plt

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
            acts, probs = zip(*outcomes)
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
            # Compute counterfactual regrets for this sampled action
            cfv_all = np.zeros(len(legal))
            for i, a in enumerate(legal):
                tmp_state = state.clone()
                tmp_state.apply_action(a)
                cfv_all[i] = self._sample(tmp_state, traversing_player, reach_probs, sampling_probs)
            v = np.dot(sigma, cfv_all)
            node.regret_sum += (reach_probs[1 - player] / sampling_probs[player]) * (cfv_all - v)
            node.strategy_sum += reach_probs[player] * sigma

        return util

    def train(self, iterations=10000, eval_interval=1000):
        history = []
        for t in tqdm(range(iterations), desc="MCCFR Training"):
            for player in range(self.game.num_players()):
                s = self.game.new_initial_state()
                self._sample(s, player, np.ones(2), np.ones(2))
            if (t + 1) % eval_interval == 0:
                pol = self.tabular_policy()
                expl = exploitability.exploitability(self.game, pol)
                history.append((t + 1, expl))
        return history

    def tabular_policy(self):
        tab = policy.TabularPolicy(self.game)
        for key, node in self.info_sets.items():
            player, info_state = key
            if info_state not in tab.state_lookup:
                continue
            idx = tab.state_lookup[info_state]
            probs = np.zeros(self.game.num_distinct_actions())
            total = node.strategy_sum.sum()
            if total > 1e-12:  # avoid division by zero
                probs[node.legal_actions] = node.strategy_sum / total
            else:
                # fallback to uniform distribution over legal actions
                probs[node.legal_actions] = 1.0 / len(node.legal_actions)
            tab.action_probability_array[idx] = probs
        return tab


    def plot(self, history):
        plt.figure(figsize=(12, 5))
        iterations, expl_values = zip(*history)
        plt.plot(iterations, expl_values, marker='o', linestyle='-', markersize=4)
        plt.title('Exploitability over Time')
        plt.xlabel('Training Iterations')
        plt.ylabel('Exploitability (NashConv)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig("kuhn_mccfr.png")
        
if __name__ == "__main__":
    game_kuhn = pyspiel.load_game("kuhn_poker")
    game_leduc= pyspiel.load_game("leduc_poker") 
    game_scopa = pyspiel.load_game("scopa_game")
    trainer = MCCFRTrainer(game_leduc)
    hist = trainer.train(iterations=5000, eval_interval=50)
    print("Exploitability:", hist[-1])
    trainer.plot(hist)