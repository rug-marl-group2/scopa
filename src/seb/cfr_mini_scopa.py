import pyspiel
from envs import openspiel_mini_scopa
from algorithms.vanilla_cfr import CFRTrainer, RandomPolicy, evaluate_agent
import matplotlib.pyplot as plt

def plot( avg_reward_history, scopa_stats=None):

        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(avg_reward_history, linewidth=1.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        plt.title('Average Reward vs. Random Agent')
        plt.xlabel('Games Played')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        scopa_hist = scopa_stats['history']
        plt.plot(scopa_hist['trained'], label='Trained Agent', linewidth=1.5, alpha=0.8)
        plt.plot(scopa_hist['opponent'], label='Random Agent', linewidth=1.5, alpha=0.8)
        plt.title('Average Scopas per Game')
        plt.xlabel('Games Played')
        plt.ylabel('Average Scopas')
        plt.legend()
        plt.grid(True)
    
        plt.tight_layout()
        plt.savefig("cfr_miniscopa_final_performance.png", dpi=150)
        plt.show()

def main():
    game = pyspiel.load_game("mini_scopa")
   
    trainer = CFRTrainer(game=game)
    history = trainer.train(
        steps=500,
        eval_interval=5,
        compute_exploitability=False 
    )
    
    cfr_policy = trainer.get_openspiel_policy()
    random_policy = RandomPolicy(game)
    
    avg_reward, avg_reward_history, scopa_stats = evaluate_agent(
        game,
        cfr_policy,
        random_policy,
        num_episodes=500
    )
    
    plot(avg_reward_history, scopa_stats)
    print(f"  Info sets learned: {len(trainer.info_set_map)}")
    print(f"  Average reward: {avg_reward:.4f}")
    print(f"  Trained agent avg scopas/game:  {scopa_stats['trained_avg']:.4f}")
    print(f"  Random agent avg scopas/game:   {scopa_stats['opponent_avg']:.4f}")

    return avg_reward

if __name__ == "__main__":
    result = main()

