"""
Monte Carlo CFR on Team Mini Scopa (2v2)

Applies Monte Carlo CFR to find TMEcor (Team-Maxmin Equilibrium with Correlation)
in the TPI-converted 2v2 Team Mini Scopa game.

The TPI conversion transforms the team game into a 2-player zero-sum game
where each team is represented as a single coordinator.
"""

import sys
sys.path.insert(0, '.')

import pyspiel
from envs import openspiel_team_mini_scopa
from algorithms.mc_cfr import MCCFRTrainer, RandomPolicy, evaluate_agent
import matplotlib.pyplot as plt

def plot_results(exploitability_history, avg_reward_history):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Exploitability plot
    if exploitability_history and len(exploitability_history) > 0:
        iterations = [x[0] for x in exploitability_history]
        exploit = [x[1] for x in exploitability_history]
        
        ax1.plot(iterations, exploit, linewidth=2, color='blue', label='Exploitability')
        ax1.set_title('MCCFR Convergence on Team Mini Scopa (2v2)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Exploitability', fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Exploitability not computed', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('MCCFR Training on Team Mini Scopa (2v2)', fontsize=14, fontweight='bold')
    
    # Average reward plot
    ax2.plot(avg_reward_history, linewidth=2, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax2.set_title('MCCFR vs Random Policy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Games Played', fontsize=12)
    ax2.set_ylabel('Average Reward (Team 0)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("results/team_mini_scopa_mccfr_example.png", dpi=150)
    print("✓ Plot saved to results/team_mini_scopa_mccfr_example.png")

def print_section(title):
    """Print formatted section header"""
    print("=" * 70)
    print(title)
    print("=" * 70)

def main():
    print_section("MCCFR on Team Mini Scopa (2v2)")
    print("Finding Team-Maxmin Equilibrium with Correlation (TMEcor)")
    
    # Load TPI-converted game
    print_section("Loading TPI Game")
    try:
        game = pyspiel.load_game("team_mini_scopa_tpi")
        print("✓ TPI Game loaded")
        print(f"  - Coordinators (teams): {game.num_players()}")
        print(f"  - Original players: 4 (2 per team)")
        print(f"  - Action space: {game.num_distinct_actions()}")
    except Exception as e:
        print(f"✗ Failed to load game: {e}")
        return None
    
    print_section("Key Properties of TMEcor")
    print("  1. Ex ante coordination: Teams agree on strategy before play")
    print("  2. No communication: Team members can't talk during game")
    print("  3. Asymmetric info: Teammates don't see each other's hands")
    print("  4. Correlated strategies: Team uses joint normal-form strategy")
    print("  5. Maxmin: Each team maximizes worst-case payoff")
    
    print_section("Training MCCFR")
    print("Algorithm: Monte Carlo Counterfactual Regret Minimization")
    print("  - Samples game tree instead of full traversal")
    print("  - Converges to Nash Equilibrium (TMEcor)")
    print("  - By Theorem 4.6 (Carminati et al. 2022):")
    print("    NE in TPI game ≡ TMEcor in original team game")
    print()
    
    # Train MCCFR
    trainer = MCCFRTrainer(game=game)
    
    # Note: Exploitability computation may be slow/infeasible for large games
    # Set compute_exploitability=False for faster training
    hist = trainer.train(
        iterations=1000,
        eval_interval=50,
        compute_exploitability=False,  # Set to True if you want exploitability
        use_custom_exploitability=True,
        approximate=False
    )
    
    print_section("Training Complete")
    print(f"✓ MCCFR training finished")
    print(f"  - Info sets learned: {len(trainer.info_sets)}")
    
    # Get learned policy
    print_section("Evaluating Policy")
    print("Testing MCCFR policy vs random baseline...")
    
    mc_cfr_policy = trainer.tabular_policy()
    random_policy = RandomPolicy(game)
    
    # Evaluate
    avg_reward, avg_reward_history = evaluate_agent(
        game,
        mc_cfr_policy,
        random_policy,
        num_episodes=1000
    )
    
    print(f"✓ Evaluation complete")
    print(f"  - Average reward (Team 0): {avg_reward:.4f}")
    print(f"  - Games played: {len(avg_reward_history)}")
    
    # Plot results
    print_section("Generating Plots")
    plot_results(hist, avg_reward_history)
    
    print_section("Results Summary")
    print(f"Info sets learned: {len(trainer.info_sets)}")
    print(f"Average reward vs random: {avg_reward:.4f}")
    
    if avg_reward > 0.1:
        print("\n✓ MCCFR policy significantly beats random play!")
    elif avg_reward > -0.1:
        print("\n≈ MCCFR policy roughly equal to random play")
    else:
        print("\n✗ MCCFR policy underperforms random play (may need more training)")
    
    print("\nNext Steps:")
    print("  - Train for more iterations (5000-10000)")
    print("  - Enable exploitability computation (if feasible)")
    print("  - Compare with vanilla CFR")
    print("  - Test Deep CFR once PyTorch is installed")
    
    return {
        'trainer': trainer,
        'policy': mc_cfr_policy,
        'avg_reward': avg_reward,
        'history': hist
    }

if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "=" * 70)
        print("MCCFR training completed successfully!")
        print("=" * 70)

