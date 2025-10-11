"""
This script trains a Counterfactual Regret Minimization (CFR) agent for Scopone Scientifico
using a JAX-based implementation. It logs training metrics, optionally evaluates performance, saves 
the resulting policy, and previews gameplay using both random and learned strategies.        

Example Usage:
    python src/train_cfr.py --iters 1500 --log_every 10 --eval_every 50 --eval_eps 32 --eval_policy current --save_kind full --max_infosets 250000
    #python src/train_cfr.py --iters 2500 --log_every 10 --eval_every 100 --eval_eps 64 --eval_policy current --save_kind current --branch_topk 1 --max_infosets 250000

Sanity Check usage:
    python src/train_cfr.py --iters 200 --batch_size 2 --eval_every 0 \
    --branch_topk 3 --max_branch_actions 4 --pw_alpha 0.3 --pw_tail 2 \
    --prune_threshold -0.05 --prune_warmup 16 --prune_reactivation 8 \
    --subset_cache_size 2048 --rollout_depth 12 --rollout_samples 4 \
    --obs_key_mode compact --max_infosets 20000 --table_dtype float16

Laptop train usage:
    python src/train_cfr.py --iters 5000 --batch_size 8 --eval_every 500 --eval_eps 64 --eval_policy avg \
    --branch_topk 5 --max_branch_actions 6 --pw_alpha 0.3 --pw_tail 3 \
    --prune_threshold -0.02 --prune_warmup 32 --prune_reactivation 8 \
    --subset_cache_size 8192 --rollout_depth 14 --rollout_samples 8 \
    --obs_key_mode full --max_infosets 250000 --table_dtype float16 --save_kind avg

Stronger baseline usage:
    python src/train_cfr.py --iters 20000 --batch_size 32 --eval_every 1000 --eval_eps 128 --eval_policy avg \
    --branch_topk 6 --max_branch_actions 8 --pw_alpha 0.3 --pw_tail 4 \
    --prune_threshold 0.0 --prune_warmup 48 --prune_reactivation 16 \
    --subset_cache_size 32768 --rollout_depth 16 --rollout_samples 12 \
    --obs_key_mode full --max_infosets 1000000 --table_dtype float16 --save_kind full


"""


import argparse
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys
import random
import time
from typing import Optional



from tlogger import TLogger
import env as env_mod
from env import MaScopaEnv
from cfr_jax import CFRTrainer

def make_env(tlog, render_mode=None):
    return MaScopaEnv(render_mode=render_mode, tlogger=tlog)

def card_str_list(cards):
    return [str(c) for c in cards]


def action_to_card(player, action_idx: int):
    for c in player.hand:
        ind = (c.rank - 1) + {
            'cuori': 0,
            'picche': 10,
            'fiori': 20,
            'bello': 30
        }[c.suit]
        if ind == action_idx:
            return c
    return None


def preview_game(env, policy: str, trainer: Optional[CFRTrainer]):
    steps = 0
    print(f"\n=== Preview game ({policy}) ===")
    env.reset()
    while True:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        seat = env.agent_name_mapping[agent]
        player = env.game.players[seat]
        obs = env.observations[agent]
        mask = env.infos[agent]["action_mask"]

        if policy == "random" or trainer is None:
            legal = [i for i, m in enumerate(mask) if m == 1]
            action = random.choice(legal)
        else:
            action = trainer.act_from_obs(seat, obs)
            if mask[action] == 0:
                legal = [i for i, m in enumerate(mask) if m == 1]
                action = legal[0]

        card = action_to_card(player, action)
        capture_desc = ""
        if card is not None:
            if card.rank == 1:
                capture_desc = "(Ace: captures all table)"
            else:
                isin, comb = env.game.card_in_table(card)
                if isin:
                    capture_desc = f"(captures {[str(c) for c in comb]})"

        print(f"Step {steps:02d} | {agent} plays {card} {capture_desc}")
        print(f"  Hand: {card_str_list(player.hand)}")
        print(f"  Table before: {card_str_list(env.game.table)}")

        env.step(action)

        print(f"  Table after:  {card_str_list(env.game.table)}")
        steps += 1

    print("Result scores:", env.roundScores())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4, help="Deals per iteration for MCCFR mini-batch")
    parser.add_argument("--rm_plus", action="store_true", default=True, help="Use RM+ (regret clamping)")
    parser.add_argument("--seed", type=int, default=time.time_ns())
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--branch_topk", type=int, default=5, help="Limit traverser branching to top-K actions by current policy; 0 disables pruning (full branching, correct CFR)")
    parser.add_argument("--max_branch_actions", type=int, default=4, help="Hard cap on traverser actions explored per infoset (0 = unlimited)")
    parser.add_argument("--pw_alpha", type=float, default=0.3, help="Progressive widening growth exponent")
    parser.add_argument("--pw_tail", type=int, default=3, help="Base number of tail actions sampled alongside top-K")
    parser.add_argument("--prune_threshold", type=float, default=-0.05, help="Regret pruning threshold (actions below are skipped)")
    parser.add_argument("--prune_warmup", type=int, default=32, help="Number of visits before regret pruning engages")
    parser.add_argument("--prune_reactivation", type=int, default=8, help="Force reconsideration of pruned actions every N visits")
    parser.add_argument("--subset_cache_size", type=int, default=8192, help="LRU cache size for subset-sum table solves (0 disables)")
    parser.add_argument("--rollout_depth", type=int, default=14, help="Depth at which to switch to random rollout evaluation (0 = disabled)")
    parser.add_argument("--rollout_samples", type=int, default=6, help="Number of rollouts used when depth cutoff triggers")
    parser.add_argument("--preview_random", type=int, default=0, help="Show this many random games after training")
    parser.add_argument("--preview_policy", type=int, default=0, help="Show this many policy games after training")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate and log game metrics every N iters (0 to disable)")
    parser.add_argument("--eval_eps", type=int, default=64, help="Episodes per evaluation mode")
    parser.add_argument("--eval_policy", type=str, default="current", choices=["avg", "current"], help="Which policy to use for evaluation")
    parser.add_argument("--max_infosets", type=int, default=0, help="Cap the number of stored infosets (0 = unlimited)")
    parser.add_argument("--obs_key_mode", type=str, default="full", choices=["full", "compact", "hand_table"], help="How much of the observation to hash for infoset keys")
    parser.add_argument("--table_dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type for regret/strategy tables to reduce memory")
    parser.add_argument("--save_path", type=str, default="", help="Path to save trained model/policy (.pkl)")
    parser.add_argument("--save_kind", type=str, default="avg", choices=["avg", "full"], help="Save average policy or full trainer state")
    args = parser.parse_args()

    # Create logger first so CFR can log metrics
    tlog = TLogger(log_dir="scopa/runs/scopa_cfr/"+time.strftime("%Y-%m-%d-%H-%M-%S"))
    import numpy as np  # local import to keep file lean
    dtype_map = {"float16": np.float16, "float32": np.float32}
    trainer = CFRTrainer(seed=args.seed,
                         tlogger=tlog,
                         branch_topk=args.branch_topk if args.branch_topk > 0 else None,
                         max_infosets=(args.max_infosets if args.max_infosets and args.max_infosets > 0 else None),
                         dtype=dtype_map.get(args.table_dtype, np.float16),
                         obs_key_mode=args.obs_key_mode,
                         rm_plus=args.rm_plus,
                         progressive_widening=True,
                         pw_alpha=args.pw_alpha,
                         pw_tail=args.pw_tail,
                         regret_prune=True,
                         prune_threshold=args.prune_threshold,
                         prune_warmup=args.prune_warmup,
                         prune_reactivation=args.prune_reactivation,
                         subset_cache_size=args.subset_cache_size,
                         max_branch_actions=args.max_branch_actions,
                         rollout_depth=args.rollout_depth,
                         rollout_samples=args.rollout_samples
                         )
    trainer.train(
        iterations=args.iters,
        seed=args.seed,
        verbose=args.verbose,
        log_every=args.log_every,
        eval_every=args.eval_every if args.eval_every > 0 else None,
        eval_episodes=args.eval_eps,
        eval_use_avg_policy=(args.eval_policy == "avg"),
        batch_size=args.batch_size
    )

    # Save trained model/policy (default inside run dir)
    save_path = args.save_path if args.save_path else os.path.join(tlog.get_log_dir(), f"policy_{args.save_kind}.pkl")
    try:
        trainer.save(save_path, kind=args.save_kind)
        print(f"Saved {args.save_kind} checkpoint to: {save_path}")
    except Exception as e:
        print(f"WARNING: failed to save checkpoint to {save_path}: {e}")

    # Build env for previews and use the same logger so you can monitor
    env = make_env(tlog)

    # Optional: show a few random games to understand dynamics
    for _ in range(max(0, args.preview_random)):
        preview_game(env, policy="random", trainer=None)

    # Optional: show a few games with the learned policy
    for _ in range(max(0, args.preview_policy)):
        preview_game(env, policy="policy", trainer=trainer)

    tlog.close()


if __name__ == "__main__":
    main()
