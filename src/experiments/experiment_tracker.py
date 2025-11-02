"""
AI GENERATED (but works)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import  List, Optional
from dataclasses import dataclass
import pickle


@dataclass
class ExperimentMetrics:
    """Stores all metrics for a single experiment run."""
    
    # Training metrics
    iterations: List[int]
    
    # Evaluation vs Random (tracked during training)
    eval_iterations: List[int] = None
    eval_rewards: List[float] = None
    eval_scopas_trained: List[float] = None
    eval_scopas_random: List[float] = None
    eval_scopa_diff: List[float] = None
    
    # Exploitability tracking (for vanilla CFR)
    exploitability_iterations: Optional[List[int]] = None
    exploitability_values: Optional[List[float]] = None
    
    # Final evaluation metrics (after training complete)
    final_reward: float = 0.0
    final_scopa_trained: float = 0.0
    final_scopa_random: float = 0.0
    final_scopa_diff: float = 0.0
    
    # Training info
    num_info_sets: int = 0
    algorithm: str = "CFR"
    
    def __post_init__(self):
        if self.eval_iterations is None:
            self.eval_iterations = []
        if self.eval_rewards is None:
            self.eval_rewards = []
        if self.eval_scopas_trained is None:
            self.eval_scopas_trained = []
        if self.eval_scopas_random is None:
            self.eval_scopas_random = []
        if self.eval_scopa_diff is None:
            self.eval_scopa_diff = []
        if self.exploitability_iterations is None:
            self.exploitability_iterations = []
        if self.exploitability_values is None:
            self.exploitability_values = []


class ExperimentTracker:
    """Tracks metrics across multiple experiment runs."""
    
    def __init__(self, experiment_name: str, save_dir: str = "experiments/results"):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runs: List[ExperimentMetrics] = []
        
    def add_run(self, metrics: ExperimentMetrics):
        """Add a completed run to the tracker."""
        self.runs.append(metrics)
        
    def save(self):
        """Save all runs to disk in both pickle and readable formats."""
        # Save pickle for Python
        save_path = self.save_dir / f"{self.experiment_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(self.runs, f)
        print(f"Saved experiment results to {save_path}")
        
        # Save data in readable format (JSON)
        self.save_data_for_plotting()
    
    def save_data_for_plotting(self):
        """Save experiment data in easily readable format for plotting."""
        import json
        
        data_path = self.save_dir / f"{self.experiment_name}_data.json"
        
        # Prepare data structure
        plot_data = {
            'experiment_name': self.experiment_name,
            'algorithm': self.runs[0].algorithm if self.runs else "Unknown",
            'num_runs': len(self.runs),
            'runs': []
        }
        
        for i, run in enumerate(self.runs):
            run_data = {
                'run_id': i + 1,
                'eval_iterations': run.eval_iterations,
                'eval_rewards': run.eval_rewards,
                'eval_scopas_trained': run.eval_scopas_trained,
                'eval_scopas_random': run.eval_scopas_random,
                'eval_scopa_diff': run.eval_scopa_diff,
                'final_reward': run.final_reward,
                'final_scopa_trained': run.final_scopa_trained,
                'final_scopa_random': run.final_scopa_random,
                'final_scopa_diff': run.final_scopa_diff,
                'num_info_sets': run.num_info_sets
            }
            # Add exploitability data if available
            if run.exploitability_iterations and len(run.exploitability_iterations) > 0:
                run_data['exploitability_iterations'] = run.exploitability_iterations
                run_data['exploitability_values'] = run.exploitability_values
            plot_data['runs'].append(run_data)
        
        # Compute statistics across runs if multiple runs
        if len(self.runs) > 1:
            # Get statistics for each evaluation iteration
            eval_iters = self.runs[0].eval_iterations
            rewards_array = np.array([run.eval_rewards for run in self.runs])
            scopas_trained_array = np.array([run.eval_scopas_trained for run in self.runs])
            scopas_random_array = np.array([run.eval_scopas_random for run in self.runs])
            scopa_diff_array = np.array([run.eval_scopa_diff for run in self.runs])
            
            plot_data['statistics'] = {
                'eval_iterations': eval_iters,
                'rewards': {
                    'mean': rewards_array.mean(axis=0).tolist(),
                    'std': rewards_array.std(axis=0).tolist(),
                    'min': rewards_array.min(axis=0).tolist(),
                    'max': rewards_array.max(axis=0).tolist()
                },
                'scopas_trained': {
                    'mean': scopas_trained_array.mean(axis=0).tolist(),
                    'std': scopas_trained_array.std(axis=0).tolist()
                },
                'scopas_random': {
                    'mean': scopas_random_array.mean(axis=0).tolist(),
                    'std': scopas_random_array.std(axis=0).tolist()
                },
                'scopa_diff': {
                    'mean': scopa_diff_array.mean(axis=0).tolist(),
                    'std': scopa_diff_array.std(axis=0).tolist()
                },
                'final_metrics': {
                    'reward_mean': float(np.mean([r.final_reward for r in self.runs])),
                    'reward_std': float(np.std([r.final_reward for r in self.runs])),
                    'scopa_trained_mean': float(np.mean([r.final_scopa_trained for r in self.runs])),
                    'scopa_trained_std': float(np.std([r.final_scopa_trained for r in self.runs])),
                    'scopa_random_mean': float(np.mean([r.final_scopa_random for r in self.runs])),
                    'scopa_random_std': float(np.std([r.final_scopa_random for r in self.runs]))
                }
            }
        
        with open(data_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        
        print(f"Saved plotting data to {data_path}")
        
        # Also save as CSV for easy spreadsheet import
        self.save_data_as_csv()
    
    def save_data_as_csv(self):
        """Save data in CSV format for spreadsheet programs."""
        import csv
        
        # Save individual runs
        for i, run in enumerate(self.runs):
            csv_path = self.save_dir / f"{self.experiment_name}_run_{i+1}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Reward', 'Scopas_Trained', 'Scopas_Random', 'Scopa_Diff'])
                for j, iter_num in enumerate(run.eval_iterations):
                    writer.writerow([
                        iter_num,
                        run.eval_rewards[j],
                        run.eval_scopas_trained[j],
                        run.eval_scopas_random[j],
                        run.eval_scopa_diff[j]
                    ])
            
            # Save exploitability data separately if available
            if run.exploitability_iterations and len(run.exploitability_iterations) > 0:
                expl_csv_path = self.save_dir / f"{self.experiment_name}_run_{i+1}_exploitability.csv"
                with open(expl_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Iteration', 'Exploitability'])
                    for j, iter_num in enumerate(run.exploitability_iterations):
                        writer.writerow([iter_num, run.exploitability_values[j]])
        
        # Save statistics across runs if multiple
        if len(self.runs) > 1:
            stats_csv_path = self.save_dir / f"{self.experiment_name}_statistics.csv"
            eval_iters = self.runs[0].eval_iterations
            rewards_array = np.array([run.eval_rewards for run in self.runs])
            scopas_trained_array = np.array([run.eval_scopas_trained for run in self.runs])
            scopas_random_array = np.array([run.eval_scopas_random for run in self.runs])
            scopa_diff_array = np.array([run.eval_scopa_diff for run in self.runs])
            
            with open(stats_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Iteration',
                    'Reward_Mean', 'Reward_Std',
                    'Scopas_Trained_Mean', 'Scopas_Trained_Std',
                    'Scopas_Random_Mean', 'Scopas_Random_Std',
                    'Scopa_Diff_Mean', 'Scopa_Diff_Std'
                ])
                for i, iter_num in enumerate(eval_iters):
                    writer.writerow([
                        iter_num,
                        rewards_array[:, i].mean(), rewards_array[:, i].std(),
                        scopas_trained_array[:, i].mean(), scopas_trained_array[:, i].std(),
                        scopas_random_array[:, i].mean(), scopas_random_array[:, i].std(),
                        scopa_diff_array[:, i].mean(), scopa_diff_array[:, i].std()
                    ])
            
            print(f"Saved {len(self.runs)} run CSV files and statistics")
        else:
            print(f"Saved 1 run CSV file")
        
    def load(self):
        """Load runs from disk."""
        load_path = self.save_dir / f"{self.experiment_name}.pkl"
        with open(load_path, 'rb') as f:
            self.runs = pickle.load(f)
        print(f"Loaded {len(self.runs)} runs from {load_path}")
        
    def plot_all_metrics(self, save_prefix: Optional[str] = None):
        """Generate all plots for the experiment."""
        if not self.runs:
            print("No runs to plot!")
            return
            
        # Determine if we have multiple runs (MC-CFR) or single run (vanilla CFR)
        is_stochastic = len(self.runs) > 1
        algorithm = self.runs[0].algorithm
        
        if is_stochastic:
            self._plot_mccfr_metrics(save_prefix)
        else:
            self._plot_vanilla_cfr_metrics(save_prefix)
    
    def _plot_mccfr_metrics(self, save_prefix: Optional[str] = None):
        """Plot all MC-CFR metrics with multiple runs."""
        fig = plt.figure(figsize=(14, 10))
        
        # (a) Evaluation vs Random
        ax1 = plt.subplot(2, 2, 1)
        for run in self.runs:
            ax1.plot(run.eval_iterations, run.eval_rewards, alpha=0.3, color='green')
        
        # Plot mean and std
        max_evals = max(len(run.eval_iterations) for run in self.runs)
        eval_iters = self.runs[0].eval_iterations
        eval_rewards_array = []
        for run in self.runs:
            eval_rewards_array.append(run.eval_rewards)
        mean_rewards = np.mean(eval_rewards_array, axis=0)
        std_rewards = np.std(eval_rewards_array, axis=0)
        
        ax1.plot(eval_iters, mean_rewards, linewidth=2, color='green', label='Mean Reward')
        ax1.fill_between(eval_iters, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                         alpha=0.2, color='green')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('(a) Evaluation vs Random')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # (b) Average Scopas per Game
        ax2 = plt.subplot(2, 2, 2)
        for run in self.runs:
            ax2.plot(run.eval_iterations, run.eval_scopas_trained, alpha=0.3, color='blue')
            ax2.plot(run.eval_iterations, run.eval_scopas_random, alpha=0.3, color='red')
        
        # Plot mean
        scopas_trained_array = [run.eval_scopas_trained for run in self.runs]
        scopas_random_array = [run.eval_scopas_random for run in self.runs]
        mean_trained = np.mean(scopas_trained_array, axis=0)
        mean_random = np.mean(scopas_random_array, axis=0)
        
        ax2.plot(eval_iters, mean_trained, linewidth=2, color='blue', label='Trained Agent')
        ax2.plot(eval_iters, mean_random, linewidth=2, color='red', label='Random Agent')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Scopas')
        ax2.set_title('(b) Average Scopas per Game')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # (c) Scopa Difference
        ax3 = plt.subplot(2, 2, 3)
        for run in self.runs:
            ax3.plot(run.eval_iterations, run.eval_scopa_diff, alpha=0.3, color='purple')
        
        # Plot mean
        scopa_diff_array = [run.eval_scopa_diff for run in self.runs]
        mean_diff = np.mean(scopa_diff_array, axis=0)
        std_diff = np.std(scopa_diff_array, axis=0)
        
        ax3.plot(eval_iters, mean_diff, linewidth=2, color='purple', label='Mean Difference')
        ax3.fill_between(eval_iters, mean_diff - std_diff, mean_diff + std_diff, 
                         alpha=0.2, color='purple')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Scopa Difference')
        ax3.set_title('(c) Scopa Difference (Trained - Random)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # (d) Final Performance Distribution
        ax4 = plt.subplot(2, 2, 4)
        final_rewards = [run.final_reward for run in self.runs]
        final_trained_scopas = [run.final_scopa_trained for run in self.runs]
        final_random_scopas = [run.final_scopa_random for run in self.runs]
        
        metrics_data = [final_rewards, final_trained_scopas, final_random_scopas]
        metrics_labels = ['Final\nReward', 'Scopas\n(Trained)', 'Scopas\n(Random)']
        
        positions = np.arange(len(metrics_labels))
        means = [np.mean(data) for data in metrics_data]
        stds = [np.std(data) for data in metrics_data]
        
        bars = ax4.bar(positions, means, yerr=stds, capsize=5, alpha=0.7, 
                       color=['green', 'blue', 'red'])
        ax4.set_xticks(positions)
        ax4.set_xticklabels(metrics_labels)
        ax4.set_ylabel('Value')
        ax4.set_title('(d) Final Performance Distribution\n(Mean ± Std)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis to start from minimum value (not 0) to show differences better
        all_values = final_rewards + final_trained_scopas + final_random_scopas
        y_min = min(all_values) - 0.1
        y_max = max(means) + max(stds) + 0.15
        ax4.set_ylim(y_min, y_max)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax4.text(i, mean + std + 0.02, f'{mean:.3f}\n±{std:.3f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'{self.experiment_name} - MC-CFR Training Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_prefix:
            save_path = self.save_dir / f"{save_prefix}_mccfr_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def _plot_vanilla_cfr_metrics(self, save_prefix: Optional[str] = None):
        """Plot vanilla CFR metrics (single deterministic run)."""
        run = self.runs[0]
        
        fig = plt.figure(figsize=(15, 10))
        
        # (a) Evaluation vs Random
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(run.eval_iterations, run.eval_rewards, linewidth=2, color='green', 
                marker='o', markersize=4)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('(a) Evaluation vs Random')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # (b) Average Scopas per Game
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(run.eval_iterations, run.eval_scopas_trained, linewidth=2, 
                color='blue', marker='o', markersize=4, label='Trained Agent')
        ax2.plot(run.eval_iterations, run.eval_scopas_random, linewidth=2, 
                color='red', marker='o', markersize=4, label='Random Agent')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Scopas')
        ax2.set_title('(b) Average Scopas per Game')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # (c) Scopa Difference
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(run.eval_iterations, run.eval_scopa_diff, linewidth=2, 
                color='purple', marker='o', markersize=4)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Scopa Difference')
        ax3.set_title('(c) Scopa Difference (Trained - Random)')
        ax3.grid(True, alpha=0.3)
        
        # (d) Final Performance Metrics
        ax4 = plt.subplot(2, 3, 4)
        metrics_data = [run.final_reward, run.final_scopa_trained, run.final_scopa_random]
        metrics_labels = ['Final\nReward', 'Scopas\n(Trained)', 'Scopas\n(Random)']
        
        positions = np.arange(len(metrics_labels))
        colors = ['green', 'blue', 'red']
        
        ax4.bar(positions, metrics_data, alpha=0.7, color=colors)
        ax4.set_xticks(positions)
        ax4.set_xticklabels(metrics_labels)
        ax4.set_ylabel('Value')
        ax4.set_title('(d) Final Performance Metrics')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, val in enumerate(metrics_data):
            ax4.text(i, val + 0.05, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # (e) Exploitability or Convergence Analysis
        ax5 = plt.subplot(2, 3, 5)
        if run.exploitability_iterations and len(run.exploitability_iterations) > 0:
            # Plot exploitability if available
            ax5.plot(run.exploitability_iterations, run.exploitability_values, linewidth=2, 
                    color='darkblue', marker='o', markersize=4)
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Exploitability (NashConv)')
            ax5.set_title('(e) Exploitability over Training')
            ax5.grid(True, alpha=0.3)
            # Set y-axis to log scale if values span multiple orders of magnitude
            if max(run.exploitability_values) / min(run.exploitability_values) > 10:
                ax5.set_yscale('log')
        else:
            # Show improvement over iterations if no exploitability
            rewards_improvement = np.array(run.eval_rewards) - run.eval_rewards[0]
            ax5.plot(run.eval_iterations, rewards_improvement, linewidth=2, 
                    color='darkgreen', marker='o', markersize=4)
            ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Reward Improvement')
            ax5.set_title('(e) Reward Improvement from Start')
            ax5.grid(True, alpha=0.3)
        
        # (f) Info Sets Growth
        ax6 = plt.subplot(2, 3, 6)
        ax6.text(0.5, 0.5, f'Total Info Sets Learned:\n{run.num_info_sets:,}', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.text(0.5, 0.3, f'Final Reward: {run.final_reward:.4f}\n'
                          f'Final Scopa Diff: {run.final_scopa_diff:.4f}',
                ha='center', va='center', fontsize=12)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('(f) Training Summary')
        
        plt.suptitle(f'{self.experiment_name} - Vanilla CFR Training Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_prefix:
            save_path = self.save_dir / f"{save_prefix}_vanilla_cfr_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print a summary of all runs."""
        if not self.runs:
            print("No runs to summarize!")
            return
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*70}")
        print(f"Algorithm: {self.runs[0].algorithm}")
        print(f"Number of runs: {len(self.runs)}")
        
        if len(self.runs) > 1:
            # Multiple runs - show statistics
            final_rewards = [run.final_reward for run in self.runs]
            final_scopas_trained = [run.final_scopa_trained for run in self.runs]
            final_scopas_random = [run.final_scopa_random for run in self.runs]
            final_scopa_diff = [run.final_scopa_diff for run in self.runs]
            
            print(f"\nFinal Reward:        {np.mean(final_rewards):.4f} ± {np.std(final_rewards):.4f}")
            print(f"Trained Scopas:      {np.mean(final_scopas_trained):.4f} ± {np.std(final_scopas_trained):.4f}")
            print(f"Random Scopas:       {np.mean(final_scopas_random):.4f} ± {np.std(final_scopas_random):.4f}")
            print(f"Scopa Difference:    {np.mean(final_scopa_diff):.4f} ± {np.std(final_scopa_diff):.4f}")
            
        else:
            # Single run
            run = self.runs[0]
            print(f"\nFinal Reward:        {run.final_reward:.4f}")
            print(f"Trained Scopas:      {run.final_scopa_trained:.4f}")
            print(f"Random Scopas:       {run.final_scopa_random:.4f}")
            print(f"Scopa Difference:    {run.final_scopa_diff:.4f}")
            print(f"Info Sets Learned:   {run.num_info_sets:,}")
        
        print(f"{'='*70}\n")

