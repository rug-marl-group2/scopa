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
    
    @property
    def policy(self) -> np.ndarray:
        norm_sum = np.sum(self.strategy_sum)
        if norm_sum > 0:
            return self.strategy_sum / norm_sum
        else:
            # If no strategies were accumulated, return a uniform random strategy.
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
            self.info_set_map[info_set_key] = InfoNode(np.array(legal_actions))
        return self.info_set_map[info_set_key]
    
    def _cfr_recursive(self , state:pyspiel.State, traversing_player, reach_p0, reach_p1):
        
        if state.is_terminal():
            return state.rewards()[traversing_player]
        
        if state.is_chance_node():
            node_utility = 0.0
            #iterate over all possible actions (Infeasible in scopa, here we need to do do monte carlo cfr or whatever other method)
            for outcome,prob in state.chance_outcomes():
                next_state = state.clone() #Because openspiel nodes are mutable we want a new one 
                next_state.apply_action(outcome)
                
                node_utility += prob * self._cfr_recursive(next_state, traversing_player, reach_p0, reach_p1)
            return node_utility
        
        #player node
        #current player is who is playing right now, traversing_player is the perspective
        node = self._get_or_create_node(state.information_state_string(), state.legal_actions()) #players need to have different information state string
        
        node_util_for_traversing_player = 0
        action_utils = np.zeros(len(state.legal_actions()))    
        
        for i, action in enumerate(state.legal_actions()):
            next_state = state.clone()
            next_state.apply_action(action)
            if state.current_player() == 0:
                action_utils[i] = self._cfr_recursive(next_state, traversing_player, reach_p0 * node.local_strategy[i], reach_p1)
            else:
                action_utils[i] = self._cfr_recursive(next_state, traversing_player, reach_p0 , reach_p1 * node.local_strategy[i])
        
        node_util_for_traversing_player = np.sum(node.local_strategy * action_utils) 
        
        if state.current_player() == traversing_player:
            reach_prob = reach_p0 if 0 == traversing_player else reach_p1
            opponent_reach_prob = reach_p1 if 0 == traversing_player else reach_p0
                        
            regret = action_utils - node_util_for_traversing_player
            node.regret_sum += opponent_reach_prob * regret
            node.strategy_sum += reach_prob * node.local_strategy
            
        node.local_strategy = node.get_strategy()
        
        return node_util_for_traversing_player
    
    def get_openspiel_policy(self) -> policy.TabularPolicy:
        tabular_policy = policy.TabularPolicy(self.game)
        for state_key, infoset in self.info_set_map.items():
            i = tabular_policy.state_lookup[state_key]
            policy_probs = np.zeros(self.game.num_distinct_actions())
            policy_probs[infoset.legal_actions] = infoset.policy
            tabular_policy.action_probability_array[i] = policy_probs
        return tabular_policy

    def train(self, steps: int, eval_interval: int = 1000):
        exploitability_history = []
        for t in tqdm(range(steps), desc="CFR Training"):
            for i in range(self.game.num_players()):
                initial_state = self.game.new_initial_state()
                self._cfr_recursive(initial_state, i, 1.0, 1.0)
            
            if (t + 1) % eval_interval == 0:
                policy = self.get_openspiel_policy()
                expl = exploitability.exploitability(self.game, policy)
                exploitability_history.append((t + 1, expl))
                
        return exploitability_history

class RandomPolicy(policy.Policy):
    """Policy that chooses actions uniformly at random."""
    def __init__(self, game):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)

    def action_probabilities(self, state):
        legal_actions = state.legal_actions()
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}

def evaluate_agent(game, trained_policy, opponent_policy, num_episodes=10000):
    """
    Evaluates a policy against an opponent, playing half the games in each seat
    to remove seat bias. Returns the unbiased average reward and bankroll.
    """
    total_winnings = 0
    bankroll_history = []
    
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
                # Correctly sample a chance outcome
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
        
        # Correctly aggregate the reward based on the agent's seat
        total_winnings += state.rewards()[agent_seat]
        bankroll_history.append(total_winnings)
        
    return total_winnings / num_episodes, bankroll_history

def plot(history, bankroll):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    iterations, expl_values = zip(*history)
    plt.plot(iterations, expl_values, marker='o', linestyle='-', markersize=4)
    plt.title('Exploitability over Time')
    plt.xlabel('Training Iterations')
    plt.ylabel('Exploitability (NashConv)')
    plt.grid(True)
    
    # Bankroll Plot
    plt.subplot(1, 2, 2)
    plt.plot(bankroll)
    plt.title('Bankroll vs. Random Agent')
    plt.xlabel('Games Played')
    plt.ylabel('Cumulative Winnings')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cfr_leduc_final_performance.png")
    plt.show()

if __name__ == "__main__":
    kuhn_poker_game = pyspiel.load_game("kuhn_poker")
    leduc_poker_game = pyspiel.load_game("leduc_poker")

    trainer = CFRTrainer(game=kuhn_poker_game)
    history = trainer.train(steps=500, eval_interval=50)
    
    cfr_policy = trainer.get_openspiel_policy()
    random_policy = RandomPolicy(kuhn_poker_game)

    avg_reward, bankroll = evaluate_agent(
        kuhn_poker_game, 
        cfr_policy, 
        random_policy, 
        num_episodes=20000
    )
    print("average reward", avg_reward)
    plot(history, bankroll)