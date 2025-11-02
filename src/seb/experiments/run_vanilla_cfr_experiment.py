"""
AI GENERATED (but works :))
"""

import sys
sys.path.append('..')

import pyspiel
from envs import openspiel_mini_scopa
from algorithms.vanilla_cfr import CFRTrainer, RandomPolicy, evaluate_agent
from experiments.experiment_tracker import ExperimentTracker, ExperimentMetrics
from open_spiel.python.algorithms import exploitability
import numpy as np


def evaluate_policy_quick(game, policy, opponent_policy, num_episodes=500):
    """Quick evaluation for intermediate checkpoints."""
    total_reward = 0
    total_scopas_trained = 0
    total_scopas_random = 0
    
    for episode in range(num_episodes):
        if episode < num_episodes / 2:
            agent_seat = 0
            policies = [policy, opponent_policy]
        else:
            agent_seat = 1
            policies = [opponent_policy, policy]
        
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


def run_vanilla_cfr_experiment(
    iterations=500,
    eval_interval=5,
    final_eval_episodes=5000,
):
    print("="*70)
    print("VANILLA CFR EXPERIMENT ON MINISCOPA")
    print("="*70)
    print(f"Iterations: {iterations}")
    print(f"Evaluation interval: {eval_interval}")
    print(f"Final evaluation episodes: {final_eval_episodes}")
    
    print("="*70)
    
    # Initialize
    game = pyspiel.load_game("mini_scopa")
    trainer = CFRTrainer(game=game)
    random_policy = RandomPolicy(game)
    
    # Metrics storage
    metrics = ExperimentMetrics(
        iterations=list(range(iterations)),
        algorithm="Vanilla CFR"
    )
    
    # Training loop with periodic evaluation
    print("\nTraining and evaluation:")
    for t in range(iterations):
        # One iteration of CFR
        for player_id in range(game.num_players()):
            initial_state = game.new_initial_state()
            trainer._cfr_recursive(initial_state, player_id, 1.0, 1.0)
        
       
        # Evaluate periodically
        if (t + 1) % eval_interval == 0:
            cfr_policy = trainer.get_openspiel_policy()
            avg_reward, avg_scopas_trained, avg_scopas_random = evaluate_policy_quick(
                game, cfr_policy, random_policy, num_episodes=500
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
    cfr_policy = trainer.get_openspiel_policy()
    final_reward, _, scopa_stats = evaluate_agent(
        game, cfr_policy, random_policy, num_episodes=final_eval_episodes
    )
    
    metrics.final_reward = final_reward
    metrics.final_scopa_trained = scopa_stats['trained_avg']
    metrics.final_scopa_random = scopa_stats['opponent_avg']
    metrics.final_scopa_diff = scopa_stats['difference']
    metrics.num_info_sets = len(trainer.info_set_map)
    
    print(f"\nFinal Results:")
    print(f"  Reward:           {final_reward:+.4f}")
    print(f"  Scopas (Trained): {metrics.final_scopa_trained:.4f}")
    print(f"  Scopas (Random):  {metrics.final_scopa_random:.4f}")
    print(f"  Scopa Diff:       {metrics.final_scopa_diff:+.4f}")
    print(f"  Info Sets:        {metrics.num_info_sets:,}")
    
    return metrics


def main():
    # Create experiment tracker
    tracker = ExperimentTracker(
        experiment_name="MiniScopa_VanillaCFR",
        save_dir="experiments/results"
    )
    
    # Run single experiment (vanilla CFR is deterministic)
    metrics = run_vanilla_cfr_experiment(
        iterations=500,
        eval_interval=5,
        final_eval_episodes=5000
    )
    
    # Add to tracker
    tracker.add_run(metrics)
    
    # Save results
    tracker.save()
    
    # Print summary
    tracker.print_summary()
    
    # Generate plots
    print("\nGenerating plots...")
    tracker.plot_all_metrics(save_prefix="miniscopa_vanilla_cfr")
    
    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()

