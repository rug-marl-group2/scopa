import numpy as np
import torch
import pyspiel
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Add your DeepCFR import
sys.path.append(os.path.dirname(__file__))
from src.deep_cfr.deep_cfr import DeepCFR  # Replace with your actual script name

class ExperimentRunner:
    """Runs multiple Deep CFR experiments and aggregates results."""
    
    def __init__(self, num_trials=5, iterations=150, advantage_epochs=5, eval_freq=5):
        self.num_trials = num_trials
        self.iterations = iterations
        self.advantage_epochs = advantage_epochs
        self.eval_freq = eval_freq
        self.results = []
        self.aggregated_history = None
        
    def run_single_experiment(self, trial_id, device="cuda"):
        """Run a single Deep CFR training session."""
        print(f"\n=== Starting Trial {trial_id + 1}/{self.num_trials} ===")
        
        # Set random seeds for reproducibility (optional)
        torch.manual_seed(trial_id * 42)
        np.random.seed(trial_id * 42)
        
        # Load game and initialize Deep CFR
        game = pyspiel.load_game("mini_scopa")
        deep_cfr = DeepCFR(game, num_players=2, device=device)
        
        # Train
        deep_cfr.train(
            iterations=self.iterations,
            advantage_epochs=self.advantage_epochs,
            eval_freq=self.eval_freq
        )
        
        # Final evaluation
        final_reward, final_scopas = deep_cfr.evaluate_vs_random(num_episodes=200)
        
        # Store results
        trial_result = {
            'trial_id': trial_id,
            'final_reward': final_reward,
            'final_scopas': final_scopas,
            'training_history': deep_cfr.training_history,
            'final_losses': [deep_cfr.training_history['losses'][i][-1] for i in range(2)],
            'final_values': [deep_cfr.training_history['values'][i][-1] for i in range(2)]
        }
        
        print(f"Trial {trial_id + 1} completed - Final reward: {final_reward:.4f}")
        
        return trial_result
    
    def aggregate_results(self):
        """Aggregate results from all trials."""
        if not self.results:
            print("No results to aggregate!")
            return
        
        # Find the maximum number of evaluation points across all trials
        max_eval_points = max(len(result['training_history']['eval_rewards']) for result in self.results)
        max_iterations = self.iterations
        
        # Initialize aggregated history structure
        self.aggregated_history = {
            'losses': [[[] for _ in range(max_iterations)] for _ in range(2)],  # [player][iteration][trials]
            'values': [[[] for _ in range(max_iterations)] for _ in range(2)],
            'buffer_sizes': [[[] for _ in range(max_iterations)] for _ in range(2)],
            'eval_rewards': [[] for _ in range(max_eval_points)],
            'eval_scopas': [[] for _ in range(max_eval_points)]
        }
        
        # Aggregate training histories
        for result in self.results:
            history = result['training_history']
            
            # Aggregate per-iteration metrics
            for iteration in range(min(max_iterations, len(history['losses'][0]))):
                for player in range(2):
                    if iteration < len(history['losses'][player]):
                        self.aggregated_history['losses'][player][iteration].append(
                            history['losses'][player][iteration]
                        )
                    if iteration < len(history['values'][player]):
                        self.aggregated_history['values'][player][iteration].append(
                            history['values'][player][iteration]
                        )
                    if iteration < len(history['buffer_sizes'][player]):
                        self.aggregated_history['buffer_sizes'][player][iteration].append(
                            history['buffer_sizes'][player][iteration]
                        )
            
            # Aggregate evaluation metrics - handle different lengths
            eval_rewards = history['eval_rewards']
            eval_scopas = history['eval_scopas']
            
            for i in range(max_eval_points):
                if i < len(eval_rewards):
                    self.aggregated_history['eval_rewards'][i].append(eval_rewards[i])
                else:
                    # For trials with fewer eval points, use the last available value
                    self.aggregated_history['eval_rewards'][i].append(eval_rewards[-1] if eval_rewards else 0)
                
                if i < len(eval_scopas):
                    self.aggregated_history['eval_scopas'][i].append(eval_scopas[i])
                else:
                    # For trials with fewer eval points, use the last available value
                    self.aggregated_history['eval_scopas'][i].append(eval_scopas[-1] if eval_scopas else [0, 0])
        
        # Compute averages and standard deviations for the summary
        self.summary = {
            'final_rewards': [result['final_reward'] for result in self.results],
            'final_scopas_trained': [result['final_scopas'][0] for result in self.results],
            'final_scopas_random': [result['final_scopas'][1] for result in self.results],
            'final_losses_p0': [result['final_losses'][0] for result in self.results],
            'final_losses_p1': [result['final_losses'][1] for result in self.results],
        }
        
        # Compute statistics
        self.summary_stats = {
            'final_reward_mean': np.mean(self.summary['final_rewards']),
            'final_reward_std': np.std(self.summary['final_rewards']),
            'final_scopas_trained_mean': np.mean(self.summary['final_scopas_trained']),
            'final_scopas_trained_std': np.std(self.summary['final_scopas_trained']),
            'final_scopas_random_mean': np.mean(self.summary['final_scopas_random']),
            'final_scopas_random_std': np.std(self.summary['final_scopas_random']),
            'scopa_difference_mean': np.mean(self.summary['final_scopas_trained']) - np.mean(self.summary['final_scopas_random']),
        }
    
    def plot_aggregated_results(self):
        """Plot aggregated results across all trials."""
        if self.aggregated_history is None:
            self.aggregate_results()
        
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 18))
        
        # Plot 1: Losses with confidence intervals
        for player in range(2):
            # Get mean and std for each iteration
            mean_losses = []
            std_losses = []
            iterations_plotted = []
            
            for iteration in range(self.iterations):
                losses = self.aggregated_history['losses'][player][iteration]
                if losses:  # Only plot iterations where we have data
                    mean_losses.append(np.mean(losses))
                    std_losses.append(np.std(losses))
                    iterations_plotted.append(iteration)
            
            if iterations_plotted:
                ax1.plot(iterations_plotted, mean_losses, label=f'Player {player}')
                ax1.fill_between(iterations_plotted, 
                               np.array(mean_losses) - np.array(std_losses),
                               np.array(mean_losses) + np.array(std_losses), alpha=0.2)
        
        ax1.set_title('Advantage Network Loss (Mean ± STD)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Evaluation rewards with confidence intervals
        eval_rewards_data = self.aggregated_history['eval_rewards']
        if eval_rewards_data and any(eval_rewards_data):
            # Find actual evaluation points (where we have data)
            eval_points_with_data = []
            mean_rewards = []
            std_rewards = []
            
            for i, rewards in enumerate(eval_rewards_data):
                if rewards:  # Only include points with data
                    eval_points_with_data.append(i * self.eval_freq)
                    mean_rewards.append(np.mean(rewards))
                    std_rewards.append(np.std(rewards))
            
            if eval_points_with_data:
                ax2.plot(eval_points_with_data, mean_rewards, 'o-', label='Mean Reward')
                ax2.fill_between(eval_points_with_data, 
                               np.array(mean_rewards) - np.array(std_rewards),
                               np.array(mean_rewards) + np.array(std_rewards), alpha=0.3)
                ax2.set_title('Evaluation vs Random (Mean ± STD)')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Average Reward')
                ax2.legend()
                ax2.grid(True)
        
        # Plot 3: Scopas with confidence intervals
        eval_scopas_data = self.aggregated_history['eval_scopas']
        if eval_scopas_data and any(eval_scopas_data):
            # Find actual evaluation points (where we have data)
            eval_points_with_data = []
            mean_trained = []
            std_trained = []
            mean_random = []
            std_random = []
            
            for i, scopa_point in enumerate(eval_scopas_data):
                if scopa_point:  # Only include points with data
                    trained_scopas = [scopa[0] for scopa in scopa_point]
                    random_scopas = [scopa[1] for scopa in scopa_point]
                    
                    eval_points_with_data.append(i * self.eval_freq)
                    mean_trained.append(np.mean(trained_scopas))
                    std_trained.append(np.std(trained_scopas))
                    mean_random.append(np.mean(random_scopas))
                    std_random.append(np.std(random_scopas))
            
            if eval_points_with_data:
                ax3.plot(eval_points_with_data, mean_trained, 'o-', label='Trained Agent', color='blue')
                ax3.fill_between(eval_points_with_data, 
                               np.array(mean_trained) - np.array(std_trained),
                               np.array(mean_trained) + np.array(std_trained), alpha=0.2, color='blue')
                
                ax3.plot(eval_points_with_data, mean_random, 'o-', label='Random Agent', color='red')
                ax3.fill_between(eval_points_with_data, 
                               np.array(mean_random) - np.array(std_random),
                               np.array(mean_random) + np.array(std_random), alpha=0.2, color='red')
                
                ax3.set_title('Average Scopas per Game (Mean ± STD)')
                ax3.set_xlabel('Iteration')
                ax3.set_ylabel('Scopas per Game')
                ax3.legend()
                ax3.grid(True)
        
        # Plot 4: Scopa difference with confidence intervals
        if eval_scopas_data and any(eval_scopas_data):
            eval_points_with_data = []
            mean_diff = []
            std_diff = []
            
            for i, scopa_point in enumerate(eval_scopas_data):
                if scopa_point:
                    diffs = [scopa[0] - scopa[1] for scopa in scopa_point]
                    eval_points_with_data.append(i * self.eval_freq)
                    mean_diff.append(np.mean(diffs))
                    std_diff.append(np.std(diffs))
            
            if eval_points_with_data:
                ax4.plot(eval_points_with_data, mean_diff, 'o-', color='green', label='Scopa Difference')
                ax4.fill_between(eval_points_with_data, 
                               np.array(mean_diff) - np.array(std_diff),
                               np.array(mean_diff) + np.array(std_diff), alpha=0.3, color='green')
                ax4.set_title('Scopa Difference (Trained - Random)')
                ax4.set_xlabel('Iteration')
                ax4.set_ylabel('Scopa Difference')
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.legend()
                ax4.grid(True)
        
        # Plot 5: Final rewards distribution
        ax5.boxplot([self.summary['final_rewards'], 
                    self.summary['final_scopas_trained'], 
                    self.summary['final_scopas_random']],
                   labels=['Final Reward', 'Trained Scopas', 'Random Scopas'])
        ax5.set_title('Final Performance Distribution')
        ax5.set_ylabel('Value')
        ax5.grid(True)
        
        # Plot 6: Trial comparison
        trials = range(1, self.num_trials + 1)
        ax6.plot(trials, self.summary['final_rewards'], 'o-', label='Final Reward')
        ax6.plot(trials, self.summary['final_scopas_trained'], 's-', label='Trained Scopas')
        ax6.plot(trials, self.summary['final_scopas_random'], '^-', label='Random Scopas')
        ax6.set_title('Performance Across Trials')
        ax6.set_xlabel('Trial')
        ax6.set_ylabel('Value')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'deep_cfr_aggregated_results_{timestamp}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
    
    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'deep_cfr_experiment_results_{timestamp}.json'
        
        results_data = {
            'experiment_config': {
                'num_trials': self.num_trials,
                'iterations': self.iterations,
                'advantage_epochs': self.advantage_epochs,
                'eval_freq': self.eval_freq,
                'timestamp': timestamp
            },
            'summary_statistics': self.summary_stats,
            'individual_trials': [
                {
                    'trial_id': result['trial_id'],
                    'final_reward': result['final_reward'],
                    'final_scopas': result['final_scopas'],
                    'final_losses': result['final_losses']
                }
                for result in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not hasattr(self, 'summary_stats'):
            self.aggregate_results()
        
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Number of trials: {self.num_trials}")
        print(f"Final Reward: {self.summary_stats['final_reward_mean']:.4f} ± {self.summary_stats['final_reward_std']:.4f}")
        print(f"Trained Scopas: {self.summary_stats['final_scopas_trained_mean']:.4f} ± {self.summary_stats['final_scopas_trained_std']:.4f}")
        print(f"Random Scopas: {self.summary_stats['final_scopas_random_mean']:.4f} ± {self.summary_stats['final_scopas_random_std']:.4f}")
        print(f"Scopa Difference: {self.summary_stats['scopa_difference_mean']:.4f}")
        print("="*50)
    
    def run_all_experiments(self, device="cuda"):
        """Run all experiments and generate reports."""
        print(f"Starting {self.num_trials} Deep CFR experiments...")
        
        for trial in range(self.num_trials):
            result = self.run_single_experiment(trial, device)
            self.results.append(result)
        
        # Aggregate and display results
        self.aggregate_results()
        self.print_summary()
        self.plot_aggregated_results()
        self.save_results()

# Main execution
if __name__ == "__main__":
    # Configuration
    NUM_TRIALS = 10
    ITERATIONS = 250
    ADVANTAGE_EPOCHS = 5
    EVAL_FREQ = 5
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run experiments
    runner = ExperimentRunner(
        num_trials=NUM_TRIALS,
        iterations=ITERATIONS,
        advantage_epochs=ADVANTAGE_EPOCHS,
        eval_freq=EVAL_FREQ
    )
    
    runner.run_all_experiments(device=device)