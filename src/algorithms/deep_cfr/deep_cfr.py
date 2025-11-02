import os
import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pyspiel
import torch
import torch.nn as nn
import torch.optim as optim
from nets import FlexibleNet, positive_regret_policy
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.envs.mini_scopa_game import MiniDeck
from src.envs.openspiel_mini_scopa import MiniScopaGame


HIDDEN = [128, 64]


class AdvantageNetwork:
    """Manages the advantage network for one player."""

    def __init__(self, input_dim, num_actions, device="cpu", lr=5e-4):
        self.device = device
        self.num_actions = num_actions

        # Simple MLP for advantages
        self.net = FlexibleNet(
            mode="mlp",
            input_shape=(input_dim,),
            output_dim=num_actions,
            mlp_hidden=HIDDEN,
            mlp_act="relu",
            mlp_norm="none",
            mlp_dropout=0.0,
        ).to(device)

        # Better initialization
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Experience buffer for training
        self.buffer = deque(maxlen=100000)

    def get_advantages(self, state_features, legal_actions_mask):
        """Get advantages for a batch of states."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)

            mask_tensor = torch.FloatTensor(legal_actions_mask).to(self.device)
            if len(mask_tensor.shape) == 1:
                mask_tensor = mask_tensor.unsqueeze(0)

            advantages = self.net(state_tensor)
            # Mask illegal actions
            advantages = advantages * mask_tensor - 1e6 * (1 - mask_tensor)
            return advantages.cpu().numpy()

    def add_experience(self, state_features, advantages, legal_actions_mask):
        """Add training data to buffer."""
        # Normalize advantages to prevent exploding gradients
        if np.max(np.abs(advantages)) > 0:
            advantages = advantages / (np.max(np.abs(advantages)) + 1e-8)
        self.buffer.append((state_features, advantages, legal_actions_mask))

    def train(self, batch_size=128, epochs=1):
        """Train on experiences in buffer."""
        if len(self.buffer) < batch_size:
            batch_size = min(len(self.buffer), 32)
            if batch_size == 0:
                return 0.0

        total_loss = 0.0

        for _ in range(epochs):
            # Get the batch
            batch = random.sample(self.buffer, batch_size)

            # Convert batch to numpy arrays first to avoid the tensor warning
            batch_data = list(
                zip(*batch)
            )  # This unpacks the list of tuples into separate lists
            states_np = np.array(batch_data[0])
            target_adv_np = np.array(batch_data[1])
            masks_np = np.array(batch_data[2])

            # Convert to tensors efficiently
            states = torch.FloatTensor(states_np).to(self.device)
            target_adv = torch.FloatTensor(target_adv_np).to(self.device)
            masks = torch.FloatTensor(masks_np).to(self.device)

            self.optimizer.zero_grad()
            pred_adv = self.net(states)

            # Only compute loss on legal actions
            loss = self.criterion(pred_adv * masks, target_adv * masks)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / epochs


class StrategyBuffer:
    """Stores past strategies for final policy computation."""

    def __init__(self, max_size=100):
        self.strategies = []
        self.weights = []
        self.max_size = max_size

    def add_strategy(self, strategy_net, iteration):
        """Add a strategy network with weight based on iteration."""
        if len(self.strategies) >= self.max_size:
            # Remove oldest strategy
            self.strategies.pop(0)
            self.weights.pop(0)

        self.strategies.append(strategy_net)
        self.weights.append(iteration + 1)

    def get_average_policy(self, state_features, legal_actions_mask):
        """Compute average policy over all stored strategies."""
        if not self.strategies:
            # Uniform random if no strategies
            mask = legal_actions_mask.astype(np.float32)
            return mask / mask.sum()

        policy = np.zeros_like(legal_actions_mask, dtype=np.float32)
        total_weight = sum(self.weights)

        for strategy, weight in zip(self.strategies, self.weights):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
                logits = strategy(state_tensor).numpy()[0]

                # Convert to policy using regret matching
                strategy_policy = positive_regret_policy(
                    torch.FloatTensor(logits).unsqueeze(0),
                    torch.FloatTensor(legal_actions_mask).unsqueeze(0),
                ).numpy()[0]

                policy += strategy_policy * (weight / total_weight)

        return policy


class RandomPolicy:
    """Simple random policy for evaluation."""

    def action_probabilities(self, state, player_id=None):
        if state.is_terminal():
            return {}

        if player_id is None:
            player_id = state.current_player()

        legal_actions = state.legal_actions(player_id)
        prob = 1.0 / len(legal_actions)
        return {action: prob for action in legal_actions}


class DeepCFR:
    """Main Deep CFR algorithm."""

    def __init__(self, game, num_players=2, device="cpu"):
        self.game = game
        self.num_players = num_players
        self.device = device

        # Estimate input dimension from game state
        self.input_dim = self._estimate_input_dim()
        print(f"Estimated input dimension: {self.input_dim}")

        # Advantage networks for each player
        self.advantage_nets = [
            AdvantageNetwork(self.input_dim, 16, device) for _ in range(num_players)
        ]

        # Strategy buffers for each player
        self.strategy_buffers = [StrategyBuffer() for _ in range(num_players)]

        # Training history
        self.training_history = {
            "losses": [[] for _ in range(num_players)],
            "values": [[] for _ in range(num_players)],
            "buffer_sizes": [[] for _ in range(num_players)],
            "eval_rewards": [],
            "eval_scopas": [],  # Add this line
        }

    def _estimate_input_dim(self):
        """Estimate input dimension from a sample game state."""
        test_state = self.game.new_initial_state()
        features = self._state_to_features(test_state, 0)
        return len(features)

    def _state_to_features(self, state, player):
        """Convert OpenSpiel state to neural network features."""
        features = []

        try:
            info_str = state.information_state_string(player)

            if "H[" in info_str and "T[" in info_str:
                hand_part = info_str.split("H[")[1].split("]")[0]
                table_part = info_str.split("T[")[1].split("]")[0]

                # Parse hand cards
                hand_features = [0] * 16
                if hand_part:
                    for card_str in hand_part.split("-"):
                        try:
                            rank_str, suit_char = card_str[:-1], card_str[-1]
                            rank = int(rank_str)
                            suit_map = {"c": 0, "f": 1, "p": 2, "b": 3}
                            suit_idx = suit_map.get(suit_char, 0)
                            for card_idx, card_rank in enumerate(
                                MiniDeck.ranks[MiniDeck.suits[suit_idx]]
                            ):
                                if card_rank == rank:
                                    action_idx = suit_idx * 4 + card_idx
                                    if action_idx < 16:
                                        hand_features[action_idx] = 1
                                    break
                        except:
                            continue
                features.extend(hand_features)

                # Parse table cards
                table_features = [0] * 16
                if table_part:
                    for card_str in table_part.split("-"):
                        try:
                            rank_str, suit_char = card_str[:-1], card_str[-1]
                            rank = int(rank_str)
                            suit_map = {"c": 0, "f": 1, "p": 2, "b": 3}
                            suit_idx = suit_map.get(suit_char, 0)
                            for card_idx, card_rank in enumerate(
                                MiniDeck.ranks[MiniDeck.suits[suit_idx]]
                            ):
                                if card_rank == rank:
                                    action_idx = suit_idx * 4 + card_idx
                                    if action_idx < 16:
                                        table_features[action_idx] = 1
                                    break
                        except:
                            continue
                features.extend(table_features)

                # Add turn indicator and basic features
                features.extend([float(player == state.current_player()), 0.0])

            else:
                features = [0] * 34

        except Exception:
            features = [0] * 34

        return np.array(features, dtype=np.float32)

    def _get_legal_actions_mask(self, state, player):
        """Get legal actions mask for a state."""
        legal_actions = state.legal_actions(player)
        mask = np.zeros(16, dtype=np.float32)
        mask[legal_actions] = 1.0
        return mask

    def _external_sampling_cfr(self, state, player, depth=0, prob=1.0):
        """External sampling CFR."""
        if state.is_terminal():
            rewards = state.rewards()
            reward = (
                float(rewards[player])
                if hasattr(rewards, "__getitem__")
                else float(rewards)
            )
            return reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, probs = zip(*outcomes)
            action = np.random.choice(acts, p=probs)
            next_state = state.clone()
            next_state.apply_action(action)
            return self._external_sampling_cfr(next_state, player, depth + 1, prob)

        current_player = state.current_player()
        state_features = self._state_to_features(state, current_player)
        legal_actions_mask = self._get_legal_actions_mask(state, current_player)

        # Get advantages from network
        advantages = self.advantage_nets[current_player].get_advantages(
            state_features, legal_actions_mask
        )

        if advantages.ndim == 2 and advantages.shape[0] == 1:
            advantages = advantages[0]

        # Convert to policy using regret matching
        policy = positive_regret_policy(
            torch.FloatTensor(advantages).unsqueeze(0),
            torch.FloatTensor(legal_actions_mask).unsqueeze(0),
        ).numpy()[0]

        if current_player == player:
            # Update advantage network for traversing player
            value = 0.0
            counterfactual_values = np.zeros(16, dtype=np.float32)

            legal_actions_list = state.legal_actions(current_player)

            for action in legal_actions_list:
                child_state = state.clone()
                child_state.apply_action(action)

                action_value = self._external_sampling_cfr(
                    child_state, player, depth + 1, prob * policy[action]
                )
                value += policy[action] * action_value
                counterfactual_values[action] = action_value

            # Compute regrets
            regrets = counterfactual_values - value

            # Always add experience
            self.advantage_nets[current_player].add_experience(
                state_features, regrets, legal_actions_mask
            )

            return value
        else:
            # Sample action for opponent
            legal_actions = state.legal_actions(current_player)
            if len(legal_actions) == 0:
                return 0.0

            action_probs = policy[legal_actions]
            if action_probs.sum() == 0:
                action = np.random.choice(legal_actions)
            else:
                action = np.random.choice(
                    legal_actions, p=action_probs / action_probs.sum()
                )

            next_state = state.clone()
            next_state.apply_action(action)
            return self._external_sampling_cfr(
                next_state, player, depth + 1, prob * policy[action]
            )

    def evaluate_vs_random(self, num_episodes=100):
        """Evaluate the current strategy against random play."""
        total_reward = 0.0
        total_trained_scopas = 0
        total_random_scopas = 0
        random_policy = RandomPolicy()

        for episode in range(num_episodes):
            # Alternate seats for fairness
            if episode < num_episodes / 2:
                trained_seat = 0
                random_seat = 1
            else:
                trained_seat = 1
                random_seat = 0

            state = self.game.new_initial_state()

            while not state.is_terminal():
                current_player = state.current_player()

                if current_player == trained_seat:
                    # Use our trained policy
                    policy_probs = self.get_policy(state, current_player)
                    legal_actions = state.legal_actions(current_player)
                    action_probs = np.array([policy_probs[a] for a in legal_actions])

                    if np.any(np.isnan(action_probs)) or np.sum(action_probs) <= 0:
                        action_probs = np.ones(len(legal_actions)) / len(legal_actions)
                    else:
                        action_probs = action_probs / np.sum(action_probs)

                    action = np.random.choice(legal_actions, p=action_probs)
                else:
                    # Use random policy
                    action_probs = random_policy.action_probabilities(
                        state, current_player
                    )
                    actions, probs = zip(*action_probs.items())
                    action = np.random.choice(actions, p=probs)

                state.apply_action(action)

            rewards = state.rewards()
            total_reward += rewards[trained_seat]  # Only track trained agent's reward

            # Extract scopas and assign to correct agent type
            if hasattr(state, "env") and hasattr(state.env, "game"):
                game = state.env.game
                # Get scopas by agent type, not by fixed seat
                total_trained_scopas += game.players[trained_seat].scopas
                total_random_scopas += game.players[random_seat].scopas

        avg_reward = total_reward / num_episodes
        avg_trained_scopas = total_trained_scopas / num_episodes
        avg_random_scopas = total_random_scopas / num_episodes

        self.training_history["eval_rewards"].append(avg_reward)
        self.training_history["eval_scopas"].append(
            [avg_trained_scopas, avg_random_scopas]
        )

        return avg_reward, [avg_trained_scopas, avg_random_scopas]

    def train(self, iterations=100, advantage_epochs=10, eval_freq=5):
        """Train Deep CFR for specified iterations."""
        pbar = tqdm(total=iterations, desc="Deep CFR Training")

        for iteration in range(iterations):
            # Train advantage networks for each player
            iteration_losses = []
            iteration_values = []

            for player in range(self.num_players):
                # Perform external sampling CFR
                state = self.game.new_initial_state()
                value = self._external_sampling_cfr(state, player)

                # Train advantage network
                loss = self.advantage_nets[player].train(epochs=advantage_epochs)

                # Store history
                iteration_losses.append(loss)
                iteration_values.append(value)
                self.training_history["losses"][player].append(loss)
                self.training_history["values"][player].append(value)
                self.training_history["buffer_sizes"][player].append(
                    len(self.advantage_nets[player].buffer)
                )

            # Store current strategy
            if iteration > 0:
                for player in range(self.num_players):
                    strategy_net = FlexibleNet(
                        mode="mlp",
                        input_shape=(self.input_dim,),
                        output_dim=16,
                        mlp_hidden=HIDDEN,
                        mlp_act="relu",
                        mlp_norm="none",
                    )
                    strategy_net.load_state_dict(
                        self.advantage_nets[player].net.state_dict()
                    )
                    self.strategy_buffers[player].add_strategy(strategy_net, iteration)

            # Evaluate periodically
            if iteration % eval_freq == 0:
                eval_reward, eval_scopas = self.evaluate_vs_random(num_episodes=50)
                pbar.set_postfix(
                    {
                        "P0 loss": f"{iteration_losses[0]:.4f}",
                        "P1 loss": f"{iteration_losses[1]:.4f}",
                        "Eval vs Random": f"{eval_reward:.3f}",
                        "Scopas P0/P1": f"{eval_scopas[0]:.2f}/{eval_scopas[1]:.2f}",  # Add scopa info
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "P0 loss": f"{iteration_losses[0]:.4f}",
                        "P1 loss": f"{iteration_losses[1]:.4f}",
                        "P0 value": f"{iteration_values[0]:.3f}",
                    }
                )

            pbar.update(1)

        pbar.close()

    def get_policy(self, state, player):
        """Get average policy for a state."""
        state_features = self._state_to_features(state, player)
        legal_actions_mask = self._get_legal_actions_mask(state, player)

        return self.strategy_buffers[player].get_average_policy(
            state_features, legal_actions_mask
        )

    def plot_training_progress(self):
        """Plot training progress including scopas."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 24))

        # Plot losses
        for player in range(self.num_players):
            ax1.plot(self.training_history["losses"][player], label=f"Player {player}")
        ax1.set_title("Advantage Network Loss")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Plot values
        for player in range(self.num_players):
            ax2.plot(self.training_history["values"][player], label=f"Player {player}")
        ax2.set_title("Expected Values")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        # Plot buffer sizes
        for player in range(self.num_players):
            ax3.plot(
                self.training_history["buffer_sizes"][player], label=f"Player {player}"
            )
        ax3.set_title("Experience Buffer Size")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Buffer Size")
        ax3.legend()
        ax3.grid(True)

        # Plot evaluation rewards
        if self.training_history["eval_rewards"]:
            eval_iterations = [
                i * 5 for i in range(len(self.training_history["eval_rewards"]))
            ]
            ax4.plot(eval_iterations, self.training_history["eval_rewards"], "o-")
            ax4.set_title("Evaluation vs Random")
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("Average Reward")
            ax4.grid(True)

        # Plot scopas per player
        if self.training_history["eval_scopas"]:
            eval_iterations = [
                i * 5 for i in range(len(self.training_history["eval_scopas"]))
            ]
            trained_scopas = [
                scopa[0] for scopa in self.training_history["eval_scopas"]
            ]
            random_scopas = [scopa[1] for scopa in self.training_history["eval_scopas"]]

            # Fixed: Use consistent labels for trained vs random
            ax5.plot(
                eval_iterations,
                trained_scopas,
                "o-",
                label="Trained Agent",
                color="blue",
            )
            ax5.plot(
                eval_iterations, random_scopas, "o-", label="Random Agent", color="red"
            )
            ax5.set_title("Average Scopas per Game")
            ax5.set_xlabel("Iteration")
            ax5.set_ylabel("Scopas per Game")
            ax5.legend()
            ax5.grid(True)

            # Plot scopa difference (Trained - Random)
            scopa_diff = [
                scopa[0] - scopa[1] for scopa in self.training_history["eval_scopas"]
            ]
            ax6.plot(eval_iterations, scopa_diff, "o-", color="green")
            ax6.set_title("Scopa Difference (Trained - Random)")
            ax6.set_xlabel("Iteration")
            ax6.set_ylabel("Scopa Difference")
            ax6.axhline(y=0, color="black", linestyle="--", alpha=0.5)
            ax6.grid(True)

        plt.tight_layout()
        plt.savefig(
            "deep_cfr_training.png", dpi=150, bbox_inches="tight", transparent=True
        )
        plt.show()


# Training script
if __name__ == "__main__":
    # Load your game
    game = pyspiel.load_game("mini_scopa")

    # Set device
    device = ("cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                    else "cpu"))   
     
    # Initialize Deep CFR
    deep_cfr = DeepCFR(game, num_players=2, device=device)

    # Train
    print("Starting Deep CFR training...")
    deep_cfr.train(iterations=10, advantage_epochs=5, eval_freq=5)

    print("Training completed!")

    # Plot results
    print("Plotting training progress...")
    deep_cfr.plot_training_progress()

    # Final evaluation
    print("Running final evaluation vs random...")
    final_reward, final_scopas = deep_cfr.evaluate_vs_random(num_episodes=200)
    print(f"Final average reward vs random: {final_reward:.4f}")
    print(
        f"Final average scopas - Trained: {final_scopas[0]:.4f}, Random: {final_scopas[1]:.4f}"
    )
