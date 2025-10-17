import numpy as np
from dataclasses import dataclass
import pyspiel
from tqdm import tqdm
import matplotlib.pyplot as plt

from open_spiel.python import policy 
from open_spiel.python.algorithms import exploitability

@dataclass
class InfoNode:
    legal_actions: np.ndarray
    regret_sum: np.ndarray = None
    strategy_sum: np.ndarray = None
    local_strategy: np.ndarray = None
    
    def __post_init__(self):
        if self.regret_sum is None:
            self.regret_sum = np.zeros(self.legal_actions.size)
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(self.legal_actions.size)
        if self.local_strategy is None:
            self.local_strategy = np.ones(self.legal_actions.size) / self.legal_actions.size

    def get_strategy(self):
        positive_regrets = np.maximum(self.regret_sum, 0)
        norm_sum = np.sum(positive_regrets)
        
        if norm_sum > 0:
            return positive_regrets / norm_sum
        else:
            return np.ones(self.legal_actions.size) / self.legal_actions.size
class CFRTrainer:
    """
        1. If node is terminal return reward
        2. If node chance_node go over all possible outcomes, and compute expected reward and weight it by probability of that chance outcome occuring
        3. Player node
    """
    def __init__(self, game ):
        self.game = game
        self.info_set_map = {} # Mapping from key to info node, where each key is tuple (playerId,observation) 
    
    def _get_or_create_node(self, info_set_key, legal_actions) -> InfoNode:
        if info_set_key not in self.info_set_map:
            self.info_set_map[info_set_key] = InfoNode(legal_actions)
        return self.info_set_map[info_set_key]
    
    def _cfr_recursive(self , state:pyspiel.State, traversing_player, reach_p0, reach_p1):
        
        if state.is_terminal():
            return state.rewards()[traversing_player]
        
        if state.is_chance_node():
            node_utility = 0.0
            #iterate over all possible actions
            for outcome,prob in state.chance_outcomes():
                next_state = state.clone() #Because openspiel nodes are mutable we want a new one 
                next_state.apply_action(outcome)
                
                node_utility += prob * self._cfr_recursive(next_state, traversing_player, reach_p0, reach_p1)
            return node_utility
        
        #player node
        #current player is who is playing right now, traversing_player is the perspective
        node = self._get_or_create_node(state.information_state_string(), np.array(state.legal_actions()))
        
        node_util_for_current_player = 0
        action_utils = np.zeros(len(state.legal_actions()))    
        
        for i, action in enumerate(state.legal_actions()):
            next_state = state.clone()
            next_state.apply_action(action)
            if state.current_player() == 0:
                action_utils[i] = self._cfr_recursive(next_state, traversing_player, reach_p0 * node.local_strategy[i], reach_p1)
            else:
                action_utils[i] = self._cfr_recursive(next_state, traversing_player, reach_p0 , reach_p1 * node.local_strategy[i])
        
        node_util_for_current_player = np.sum(node.local_strategy * action_utils) 
        
        if state.current_player() == traversing_player:
            reach_prob = reach_p0 if state.current_player() == traversing_player else reach_p1
            opponent_reach_prob = reach_p1 if state.current_player() == traversing_player else reach_p0
                        
            regret = action_utils -node_util_for_current_player
            node.regret_sum += opponent_reach_prob * regret
            node.strategy_sum += reach_prob * node.local_strategy
            
        node.local_strategy = node.get_strategy()
        
        return node_util_for_current_player

    def train(self, steps: int, eval_interval: int = 1000):
        for t in tqdm(range(steps), desc="CFR Training"):
            for i in range(2):
                initial_state = self.game.new_initial_state()
                self._cfr_recursive(initial_state, i, 1.0, 1.0)

if __name__ == "__main__":
    #Load the openspiel game which already provides the tree, in our case we will generate the tree in montecarlo way
    kuhn_poker_game = pyspiel.load_game("kuhn_poker")

    trainer = CFRTrainer(game=kuhn_poker_game)
    history = trainer.train(steps=100000)