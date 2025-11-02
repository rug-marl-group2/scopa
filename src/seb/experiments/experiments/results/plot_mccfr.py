import json
import matplotlib.pyplot as plt
import numpy as np

with open('MiniScopa_MCCFR_data.json', 'r') as f:
    data = json.load(f)

stats = data['statistics']
eval_iters = stats['eval_iterations']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
mean = np.array(stats['rewards']['mean'])
std = np.array(stats['rewards']['std'])
ax.plot(eval_iters, mean, linewidth=2, color='green', label='Mean Reward')
ax.fill_between(eval_iters, mean - std, mean + std, alpha=0.2, color='green')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Average Reward')
ax.set_title('Evaluation vs Random')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
mean_trained = np.array(stats['scopas_trained']['mean'])
std_trained = np.array(stats['scopas_trained']['std'])
mean_random = np.array(stats['scopas_random']['mean'])
std_random = np.array(stats['scopas_random']['std'])
ax.plot(eval_iters, mean_trained, linewidth=2, color='blue', label='Trained Agent')
ax.fill_between(eval_iters, mean_trained - std_trained, mean_trained + std_trained, alpha=0.2, color='blue')
ax.plot(eval_iters, mean_random, linewidth=2, color='red', label='Random Agent')
ax.fill_between(eval_iters, mean_random - std_random, mean_random + std_random, alpha=0.2, color='red')
ax.set_xlabel('Iteration')
ax.set_ylabel('Average Scopas')
ax.set_title('Average Scopas per Game')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
mean_diff = np.array(stats['scopa_diff']['mean'])
std_diff = np.array(stats['scopa_diff']['std'])
ax.plot(eval_iters, mean_diff, linewidth=2, color='purple', label='Mean Difference')
ax.fill_between(eval_iters, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color='purple')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Scopa Difference')
ax.set_title('Scopa Difference (Trained - Random)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mccfr_plots.png', dpi=300, bbox_inches='tight')
plt.show()

