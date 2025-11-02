"""
AI GENERATED (but works)
"""
"""
Run MC-CFR experiment on MiniScopa with comprehensive metric tracking.
MC-CFR is stochastic, so multiple runs are performed.
"""

import sys
sys.path.append('..')

import pyspiel
from envs import openspiel_mini_scopa
from algorithms.mc_cfr import MCCFRTrainer, RandomPolicy, evaluate_agent
from experiments.experiment_tracker import ExperimentTracker, ExperimentMetrics
import numpy as np


def evaluate_policy_quick(game, trained_policy, opponent_policy, num_episodes=500):
    """Quick evaluation for intermediate checkpoints."""
    total_reward = 0
    total_scopas_trained = 0
    total_scopas_random = 0
    
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
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action = np.random.choice(actions, p=probs)
                state.apply_action(action)
        
        total_reward += state.rewards()[agent_seat]
        
        # Track scopas
        if hasattr(state, 'env') and hasattr(state.env, 'game'):
            try:
                players = state.env.game.players
                total_scopas_trained += players[agent_seat].scopas
                total_scopas_random += players[1 - agent_seat].scopas
            except:
                pass
    
    return (total_reward / num_episodes, 
            total_scopas_trained / num_episodes,
            total_scopas_random / num_episodes)


def run_single_mccfr_experiment(
    run_id,
    iterations=500,
    eval_interval=5,
    final_eval_episodes=5000
):
    """
    Run a single MC-CFR experiment.
    
    Args:
        run_id: Identifier for this run
        iterations: Number of MC-CFR training iterations
        eval_interval: Evaluate policy every N iterations
        final_eval_episodes: Number of episodes for final evaluation
    """
    print(f"\n{'='*70}")
    print(f"MC-CFR RUN #{run_id}")
    print(f"{'='*70}")
    
    # Initialize
    game = pyspiel.load_game("mini_scopa")
    trainer = MCCFRTrainer(game=game)
    random_policy = RandomPolicy(game)
    
    # Metrics storage
    metrics = ExperimentMetrics(
        iterations=list(range(iterations)),
        algorithm="MC-CFR"
    )
    
    # Training loop with periodic evaluation
    print("Training and evaluation:")
    for t in range(iterations):
        # One iteration of MC-CFR
        trainer.iteration()
        
        # Evaluate periodically
        if (t + 1) % eval_interval == 0:
            mccfr_policy = trainer.tabular_policy()
            avg_reward, avg_scopas_trained, avg_scopas_random = evaluate_policy_quick(
                game, mccfr_policy, random_policy, num_episodes=500
            )
            
            metrics.eval_iterations.append(t + 1)
            metrics.eval_rewards.append(avg_reward)
            metrics.eval_scopas_trained.append(avg_scopas_trained)
            metrics.eval_scopas_random.append(avg_scopas_random)
            metrics.eval_scopa_diff.append(avg_scopas_trained - avg_scopas_random)
            
            print(f"  Iter {t+1:3d}: Reward={avg_reward:+.4f}, "
                  f"Scopas(T/R)={avg_scopas_trained:.3f}/{avg_scopas_random:.3f}, "
                  f"Diff={avg_scopas_trained - avg_scopas_random:+.3f}")
    
    # Final comprehensive evaluation
    print(f"\nFinal evaluation ({final_eval_episodes} episodes)...")
    mccfr_policy = trainer.tabular_policy()
    final_reward, _, scopa_stats = evaluate_agent(
        game, mccfr_policy, random_policy, num_episodes=final_eval_episodes
    )
    
    metrics.final_reward = final_reward
    metrics.final_scopa_trained = scopa_stats['trained_avg']
    metrics.final_scopa_random = scopa_stats['opponent_avg']
    metrics.final_scopa_diff = scopa_stats['difference']
    metrics.num_info_sets = len(trainer.info_sets)
    
    print(f"\nRun #{run_id} Final Results:")
    print(f"  Reward:           {final_reward:+.4f}")
    print(f"  Scopas (Trained): {metrics.final_scopa_trained:.4f}")
    print(f"  Scopas (Random):  {metrics.final_scopa_random:.4f}")
    print(f"  Scopa Diff:       {metrics.final_scopa_diff:+.4f}")
    print(f"  Info Sets:        {metrics.num_info_sets:,}")
    
    return metrics


def run_mccfr_experiments(
    num_runs=5,
    iterations=500,
    eval_interval=5,
    final_eval_episodes=5000
):
    """
    Run multiple MC-CFR experiments to capture variance.
    
    Args:
        num_runs: Number of independent runs
        iterations: Number of MC-CFR training iterations per run
        eval_interval: Evaluate policy every N iterations
        final_eval_episodes: Number of episodes for final evaluation
    """
    print("="*70)
    print("MC-CFR EXPERIMENTS ON MINISCOPA")
    print("="*70)
    print(f"Number of runs: {num_runs}")
    print(f"Iterations per run: {iterations}")
    print(f"Evaluation interval: {eval_interval}")
    print(f"Final evaluation episodes: {final_eval_episodes}")
    print("="*70)
    
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_name="MiniScopa_MCCFR",
        save_dir="experiments/results"
    )
    
    # Run experiments
    for run_id in range(1, num_runs + 1):
        metrics = run_single_mccfr_experiment(
            run_id=run_id,
            iterations=iterations,
            eval_interval=eval_interval,
            final_eval_episodes=final_eval_episodes
        )
        tracker.add_run(metrics)
    
    # Save results
    tracker.save()
    
    # Print summary
    tracker.print_summary()
    
    # Generate plots
    print("\nGenerating comprehensive plots...")
    tracker.plot_all_metrics(save_prefix="miniscopa_mccfr")
    
    print("\n✓ All experiments complete!")
    
    return tracker


def main():
    # Run multiple MC-CFR experiments
    tracker = run_mccfr_experiments(
        num_runs=10,
        iterations=500,
        eval_interval=5,
        final_eval_episodes=5000
    )


if __name__ == "__main__":
    main()

