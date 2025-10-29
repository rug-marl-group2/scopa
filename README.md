## 🚀 Quickstart

### 1) Create & activate a virtual environment (Python ≥3.10)

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate          # (Linux/macOS)
# .\.venv\Scripts\activate         # (Windows PowerShell)
```

### 2) Install dependencies

```bash
pip intall -r requirements.txt
```
### 3) Run 1v1 Team Deep CFR with ipython

```bash
ipython src/scripts/train_scopa1v1.py -- \
  --mode mlp \
  --iters 2 \
  --traversals_per_player 16 \
  --regret_steps 10 \
  --policy_steps 10
```

### Optional: Use Job Scripts

```bash
# Launch a training job (edit the script to adjust args/resources)
./jobs/launch_dcfr.sh

# Quick test job / smoke test
./jobs/test_dcfr.sh
```

## 🧠 Methodology

This project implements a **general Deep Counterfactual Regret Minimization (Deep CFR)** framework for **imperfect-information games** such as *Scopa*.  
The system is designed to support both the original **2 v 2** multiplayer setup and a **1 v 1 team-based abstraction**, using PyTorch and PettingZoo.

---

### 🎯 Overview

Deep CFR is a neural approximation of the classical **Counterfactual Regret Minimization (CFR)** algorithm.  
It learns equilibrium strategies in extensive-form games by iteratively:

1. **Traversing the game tree** to sample information sets.  
2. **Computing regrets** for actions that would have led to better outcomes.  
3. **Updating neural networks** to generalize these regrets across unseen states.  
4. **Deriving policies** proportional to the positive cumulative regrets.  

Over many iterations, average regret → 0 ⇒ policies approach an **equilibrium** (Nash in zero-sum, CCE otherwise).

---

### ⚙️ Training Flow

Each Deep CFR iteration proceeds as follows:

| Phase | Component | Description |
|:------|:-----------|:------------|
| **1. Traversal** | `ExternalSamplingTraverser` | Simulates partial game rollouts (“external sampling”) to collect advantage and policy samples for each agent. |
| **2. Regret Update** | `RegretNet` + MSE loss | Learns to predict the instantaneous regret / advantage for every legal action. |
| **3. Policy Update** | `PolicyNet` + cross-entropy | Learns the average strategy (probability of each action under the current policy). |
| **4. Evaluation** | `evaluate_vs_random`, `evaluate_selfplay` | Tests agent strength and equilibrium stability. |
| **5. Logging + Checkpoint** | `RunLogger` | Saves metrics, TensorBoard logs, and model checkpoints. |

All training data are stored in **reservoir-sampled replay buffers** (`RegretMemory`, `PolicyMemory`), ensuring an unbiased sample of all previously visited information sets.

---

### 🧩 Architecture Overview

Each agent *i* maintains two neural networks:

- **RegretNet \(R_i(o)\)** → predicts *action-advantages* (instantaneous regrets).  
- **PolicyNet \(P_i(o)\)** → predicts *average-policy logits*.

Both networks share the same flexible architecture (`FlexibleNet`) that can be configured as a pure MLP or Conv + MLP stack depending on the observation shape.

---

### 🧮 Algorithmic Abstraction

#### 🔸 Original Scopa (2 v 2)

- **Four players**: `P0 P1 P2 P3`.  
- **Teams**: A = (0, 2), B = (1, 3).  
- **Mixed-motive game** → partially cooperative within team, competitive across teams.  
- **Learning goal**: minimize team-level cumulative regret.  
- **Solution concept**: Coarse Correlated Equilibrium (CCE).  

#### 🔸 1 v 1 Team Abstraction

To stabilize learning, the two teammates are merged into a **single meta-agent** controlling both seats:

| Aspect | 4-Player Scopa | 1 v 1 Abstraction |
|:--------|:----------------|:-----------------|
| Agents | 4 individual players | 2 meta-agents (teams) |
| Game type | general-sum | zero-sum |
| Training | decentralized per seat | centralized per team |
| Objective | minimize total regret (CCE) | minimax (Team Nash Equilibrium) |
| Evaluation | each seat acts independently | same network reused for both teammates |

During training, both teammates share one set of networks (Regret / Policy), learning a **joint strategy** for the team.  
During execution, each seat acts independently using its **local observation** — the policy is shared but execution is **decentralized** (CTDE = Centralized Training, Decentralized Execution).

This abstraction restores the **zero-sum** structure and allows Deep CFR to approximate a proper **Nash Equilibrium** between the two teams.

---

### 🧩 Evaluation

Two complementary evaluation modes are used:

| Function | Purpose | Description |
|:----------|:---------|:-------------|
| `evaluate_vs_random()` | Skill test | Plays the learned policy (even seats / Team A) vs random opponents (odd seats / Team B). Reports winrate and average score difference. |
| `evaluate_selfplay()` | Stability test | Runs all players with their current policies — an equilibrium proxy (balanced ≈ 0.5 winrate). |

During evaluation, **no centralized controller** is used:  
each seat queries its shared team network independently based on its current observation.

---

### 🧭 Theoretical Guarantees

- In **two-player zero-sum** form (1 v 1 teams): Deep CFR → Nash Equilibrium as average regret → 0.  
- In **multi-player general-sum** form (4-seat Scopa): converges to a Coarse Correlated Equilibrium.  

---

### ⚖️ Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|:-------------|:---------------|
| Theoretically grounded (CFR framework). | Large sample requirements for convergence. |
| Scalable with neural approximation. | Function approximation introduces bias. |
| Stable self-play training (no oscillation). | Non-zero-sum (4-player) → weaker equilibrium notion (CCE). |
| CTDE framework enables coordinated play. | Centralized training may reveal teammate info unavailable in real play. |

---

### 🔄 Summary Flow

1. **Traverse** → collect (obs, mask, adv, policy samples).  
2. **Train Regret** → minimize MSE of predicted advantages.  
3. **Train Policy** → minimize cross-entropy vs average strategy.  
4. **Evaluate** → vs random and self-play diagnostics.  
5. **Repeat** until average regret ≈ 0 → equilibrium policy.

---

**Key Takeaway:**  
> The 1 v 1 team abstraction transforms Scopa from a 4-player cooperative/competitive card game into a two-player zero-sum imperfect-information match.  
> Deep CFR then learns approximate team-level Nash strategies through centralized training and decentralized execution.
