import pyspiel
from envs import openspiel_mini_scopa
from algorithms.mc_cfr import MCCFRTrainer, RandomPolicy, evaluate_agent
import matplotlib.pyplot as plt

def plot_results(avg_reward_history):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_reward_history, linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    plt.title('CFR Performance on Mini Scopa', fontsize=14, fontweight='bold')
    plt.xlabel('Games Played', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mini_scopa_mccfr_example.png", dpi=150)

def main():
    game = pyspiel.load_game("mini_scopa")
   
    
    trainer = MCCFRTrainer(game=game)
    hist = trainer.train(iterations=5000)
    
    mc_cfr_policy = trainer.tabular_policy()
    random_policy = RandomPolicy(game)
    
    avg_reward, avg_reward_history, scopa_stats = evaluate_agent(
        game,
        mc_cfr_policy,
        random_policy,
        num_episodes=5000
    )
    
    plot_results(avg_reward_history)
    print(f"  Info sets learned: {len(trainer.info_sets)}")
    print(f"  Average reward: {avg_reward:.4f}")
    
    if scopa_stats.get('data_collected', False):
        print(f"\n  Scopa Statistics:")
        print(f"    Trained agent avg scopas/game:  {scopa_stats['trained_avg']:.4f}")
        print(f"    Random agent avg scopas/game:   {scopa_stats['opponent_avg']:.4f}")
        print(f"    Difference (Trained - Random):  {scopa_stats['difference']:+.4f}")
    
    return avg_reward

if __name__ == "__main__":
    main()

