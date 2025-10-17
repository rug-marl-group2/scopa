import numpy as np
from dataclasses import dataclass
import pyspiel
from tqdm import tqdm
@dataclass
class InfoNode:
    num_legal_actions: int 
    
    regret_sum: np.ndarray = None
    strategy_sum: np.ndarray = None
    
    def __post_init__(self):
        if self.regret_sum is None:
            self.regret_sum = np.zeros(self.num_legal_actions, dtype=np.float32)
        if self.strategy_sum is None:
            self.strategy_sum = np.zeros(self.num_legal_actions, dtype=np.float32)

    def get_strategy(self):
        positive_regrets = np.maximum(self.regret_sum, 0)
        norm_sum = np.sum(positive_regrets)
        
        if norm_sum > 0:
            return positive_regrets / norm_sum
        else:
            return np.ones(self.num_legal_actions) / self.num_legal_actions

    def get_average_strategy(self) -> np.ndarray:
        norm_sum = np.sum(self.strategy_sum)
        if norm_sum > 0:
            return self.strategy_sum / norm_sum
        else:
            return np.full(self.num_legal_actions, 1.0 / self.num_legal_actions, dtype=np.float32)
        
class CFRTrainer:
    """
        1. If node is terminal return reward
        2. If node chance_node go over all possible outcomes, and compute expected reward and weight it by probability of that chance outcome occuring
        3. Player node
    """
    def __init__(self, game ):
        self.game = game
        self.info_set_map = {} # Mapping from key to info node, where each key is tuple (playerId,observation) 
    
    def _get_or_create_node(self, info_set_key, num_legal_actions) -> InfoNode:
        if info_set_key not in self.info_set_map:
            self.info_set_map[info_set_key] = InfoNode(num_legal_actions=num_legal_actions)
        return self.info_set_map[info_set_key]
    
    def _cfr_recursive(self , state:pyspiel.State, reach_p1, reach_p2):
        
        if state.is_terminal():
            return state.returns()[0]
        
        if state.is_chance_node():
            node_utility = 0.0
            #iterate over all possible actions
            for outcome,prob in state.chance_outcomes():
                next_state = state.clone() #Because openspiel nodes are mutable we want a new one 
                next_state.apply_action(outcome)
                
                node_utility += prob * self._cfr_recursive(next_state, reach_p1, reach_p2)
            return node_utility
        
        #player node
        current_player = state.current_player()
        info_set_key = state.information_state_string()
        legal_actions = state.legal_actions()
        node = self._get_or_create_node(info_set_key, len(legal_actions))
        
        strategy = node.get_strategy()
        action_utils = np.zeros(len(legal_actions))    
        
        for i, action in enumerate(legal_actions):
            next_state = state.clone()
            next_state.apply_action(action)
            if current_player == 0:
                action_utils[i] = -self._cfr_recursive(next_state, reach_p1 * strategy[i], reach_p2)
            else:
                action_utils[i] = -self._cfr_recursive(next_state, reach_p1 , reach_p2 * strategy[i])
        node_util = np.sum(strategy * action_utils)
        
        reach_prob = reach_p1 if current_player == 0 else reach_p2
        opponent_reach_prob = reach_p2 if current_player == 0 else reach_p1
        
        node.strategy_sum += reach_prob * strategy
        
        regret = action_utils - node_util
        node.regret_sum += opponent_reach_prob * regret

        return node_util
    
    def train(self, iterations: int):
        for _ in tqdm(range(iterations), desc="CRF Training"):
            initial_state = self.game.new_initial_state()
            self._cfr_recursive(initial_state, 1.0, 1.0)
    
    def print_strategy(self):
        print("\n--- Final Average Strategy (Nash Equilibrium) ---")
        sorted_keys = sorted(self.info_set_map.keys())
        for info_set_key in sorted_keys:
            node = self.info_set_map[info_set_key]
            avg_strategy = node.get_average_strategy()
            print(f"Info State: {info_set_key}")
            print(f"  - Action: Pass -> Prob: {avg_strategy[0]:.3f}")
            print(f"  - Action: Bet  -> Prob: {avg_strategy[1]:.3f}")

if __name__ == "__main__":
    #Load the openspiel game which already provides the tree, in our case we will generate the tree in montecarlo way
    kuhn_poker_game = pyspiel.load_game("kuhn_poker")

    trainer = CFRTrainer(game=kuhn_poker_game)
    trainer.train(iterations=100000)
    trainer.print_strategy()
    
