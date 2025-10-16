"""
This script trains a Counterfactual Regret Minimization (CFR) agent for Scopone Scientifico
using a JAX-based implementation. It logs training metrics, optionally evaluates performance, saves 
the resulting policy, and previews gameplay using both random and learned strategies.        

Example Usage:
    python scopa/src/train_cfr.py --iters 10 --log_every 1 --exploit_every 2 --exploit_eps 16 --exploit_policy avg --eval_every 4 --eval_eps 64 --eval_policy current --save_kind full --max_infosets 250000
"""


import argparse
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import sys
import random
import json
import time
from typing import Optional



from tlogger import TLogger
import env as env_mod
from env import env as make_env
from cfr_jax import CFRTrainer
from kuhn_cfr import KuhnCFRTrainer


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


def preview_game(env, policy: str, trainer: Optional[CFRTrainer], dump_dir: Optional[str] = None, label: str = "", episode_idx: int = 0):
    steps = 0
    print(f"\n=== Preview game ({policy}) ===")
    env.reset()
    transcript = []
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    while True:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            break
        seat = env.agent_name_mapping[agent]
        player = env.game.players[seat]
        obs = env.observations[agent]
        mask = env.infos[agent]["action_mask"]
        legal = [i for i, m in enumerate(mask) if m == 1]
        action_source = "random"
        fallback_used = False

        if policy == "random" or trainer is None:
            action = random.choice(legal)
        else:
            action_source = "policy"
            action = trainer.act_from_obs(seat, obs)
            if mask[action] == 0:
                fallback_used = True
                action = legal[0] if legal else int(action)

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

        step_record = {
            "step": steps,
            "agent": agent,
            "seat": seat,
            "hand": card_str_list(player.hand),
            "table_before": card_str_list(env.game.table),
            "legal_actions": legal,
            "action_index": int(action),
            "action_source": action_source,
            "fallback_used": fallback_used,
            "action_card": str(card) if card is not None else None,
            "capture_description": capture_desc,
            "action_mask": list(mask),
            "observation": obs.tolist() if hasattr(obs, "tolist") else obs,
        }

        env.step(action)

        step_record["table_after"] = card_str_list(env.game.table)
        transcript.append(step_record)

        print(f"  Table after:  {card_str_list(env.game.table)}")
        steps += 1

    result_scores = list(env.roundScores())
    print("Result scores:", result_scores)
    if dump_dir:
        episode_label = label or policy
        filename = f"{episode_label}_preview_{episode_idx:03d}.json"
        payload = {
            "policy": policy,
            "episode_index": int(episode_idx),
            "steps": transcript,
            "result_scores": result_scores,
        }
        with open(os.path.join(dump_dir, filename), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="scopa", choices=["scopa", "kuhn"], help="Game domain to train CFR on")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4, help="Deals per iteration for MCCFR mini-batch")
    parser.add_argument("--traversals_per_deal", type=int, default=0, help="Number of target seats sampled per deal (<=0 traverses all seats)")
    parser.add_argument("--rm_plus", action="store_true", default=False, help="Use RM+ (regret clamping)")
    parser.add_argument("--reward_mode", type=str, default="team", choices=["team", "selfish"], help="Select team-shared or selfish utility shaping")
    parser.add_argument("--seed", type=int, default=time.time_ns())
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--branch_topk", type=int, default=0, help="Initial top-K restriction during traversal; 0 disables pruning (full branching)")
    parser.add_argument("--branch_topk_decay", type=float, default=0, help="Multiplicative decay applied to branch_topk after each iteration")
    parser.add_argument("--branch_topk_min", type=int, default=0, help="Lower bound for branch_topk during decay")
    parser.add_argument("--max_branch_actions", type=int, default=0, help="Hard cap on traverser actions explored per infoset (0 = unlimited)")
    parser.add_argument("--pw_alpha", type=float, default=0.3, help="Progressive widening growth exponent")
    parser.add_argument("--pw_tail", type=int, default=0, help="Base number of tail actions sampled alongside top-K")
    parser.add_argument("--prune_threshold", type=float, default=-0.05, help="Regret pruning threshold (actions below are skipped)")
    parser.add_argument("--prune_warmup", type=int, default=32, help="Number of visits before regret pruning engages")
    parser.add_argument("--prune_reactivation", type=int, default=2, help="Force reconsideration of pruned actions every N visits")
    parser.add_argument("--subset_cache_size", type=int, default=8192, help="LRU cache size for subset-sum table solves (0 disables)")
    parser.add_argument("--rollout_depth", type=int, default=0, help="Depth at which to switch to random rollout evaluation (0 = disabled)")
    parser.add_argument("--rollout_samples", type=int, default=6, help="Number of rollouts used when depth cutoff triggers")
    parser.add_argument("--preview_random", type=int, default=0, help="Show this many random games after training")
    parser.add_argument("--preview_policy", type=int, default=0, help="Show this many policy games after training")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate and log game metrics every N iters (0 to disable)")
    parser.add_argument("--eval_eps", type=int, default=64, help="Episodes per evaluation mode")
    parser.add_argument("--eval_policy", type=str, default="current", choices=["avg", "current"], help="Which policy to use for evaluation")
    parser.add_argument("--exploit_every", type=int, default=0, help="Compute exploitability every N iterations (0 disables)")
    parser.add_argument("--exploit_eps", type=int, default=32, help="Deals sampled when estimating exploitability")
    parser.add_argument("--exploit_policy", type=str, default="avg", choices=["avg", "current"], help="Policy snapshot used for exploitability checks")
    parser.add_argument("--max_infosets", type=int, default=0, help="Cap the number of stored infosets (0 = unlimited)")
    parser.add_argument("--obs_key_mode", type=str, default="full", choices=["full", "compact", "hand_table"], help="How much of the observation to hash for infoset keys")
    parser.add_argument("--table_dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type for regret/strategy tables to reduce memory")
    parser.add_argument("--save_path", type=str, default="", help="Path to save trained model/policy (.pkl)")
    parser.add_argument("--save_kind", type=str, default="avg", choices=["avg", "full"], help="Save average policy or full trainer state")
    parser.add_argument("--best_save_path", type=str, default="", help="Path to save best checkpoint (defaults inside run dir)")
    parser.add_argument("--debug_topk", type=int, default=0, help="Dump top-K infosets each log interval (0 disables)")
    parser.add_argument("--debug_dump_dir", type=str, default="", help="Directory for debug dumps (defaults to run_dir/debug)")
    parser.add_argument("--preview_dump_dir", type=str, default="", help="Directory to store preview transcripts (defaults to run_dir/previews)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_root = os.path.join(repo_root, "runs", f"{args.game}_cfr")
    os.makedirs(run_root, exist_ok=True)
    run_dir = os.path.join(run_root, time.strftime("%Y-%m-%d-%H-%M-%S"))

    debug_dir = args.debug_dump_dir if args.debug_dump_dir else os.path.join(run_dir, "debug")
    if args.debug_topk <= 0:
        debug_dir = None

    preview_dump_dir = args.preview_dump_dir if args.preview_dump_dir else (os.path.join(run_dir, "previews") if (args.preview_random or args.preview_policy) else None)
    if preview_dump_dir:
        os.makedirs(preview_dump_dir, exist_ok=True)

    # Create logger first so CFR can log metrics
    tlog = TLogger(log_dir=run_dir)

    if args.game == "kuhn":
        trainer = KuhnCFRTrainer(seed=args.seed, tlogger=tlog, rm_plus=args.rm_plus)
        best_save_path = args.best_save_path if args.best_save_path else os.path.join(tlog.get_log_dir(), "policy_avg_best.pkl")
        trainer.train(
            iterations=args.iters,
            seed=args.seed,
            verbose=args.verbose,
            log_every=args.log_every,
            eval_every=args.eval_every if args.eval_every > 0 else None,
            eval_episodes=args.eval_eps,
            eval_use_avg_policy=(args.eval_policy == "avg"),
            batch_size=args.batch_size,
            best_save_path=best_save_path,
            best_save_kind=args.save_kind,
            traversals_per_deal=args.traversals_per_deal,
            branch_topk_decay=args.branch_topk_decay,
            branch_topk_min=args.branch_topk_min,
            exploit_every=(args.exploit_every if args.exploit_every > 0 else None),
            exploit_episodes=(args.exploit_eps if args.exploit_eps > 0 else None),
            exploit_use_avg_policy=(args.exploit_policy == "avg"),
            debug_topk=args.debug_topk,
            debug_dir=debug_dir
        )
        final_metrics = trainer.evaluate(episodes=args.eval_eps, seed=args.seed + 1234567, use_avg_policy=(args.eval_policy == "avg"))
        final_value = float(final_metrics.get("player0_value", 0.0))
        if best_save_path and final_value > trainer.best_vs_random_win_rate:
            trainer.best_vs_random_win_rate = final_value
            try:
                trainer.save(best_save_path)
                print(f"Saved Kuhn policy as best checkpoint to: {best_save_path} (player0_value={final_value:+.3f})")
            except Exception as exc:
                print(f"WARNING: failed to save best Kuhn checkpoint to {best_save_path}: {exc}")
        final_path = args.save_path if args.save_path else os.path.join(tlog.get_log_dir(), "policy_avg.pkl")
        try:
            trainer.save(final_path)
            print(f"Saved Kuhn policy to: {final_path} (player0_value={final_value:+.3f})")
        except Exception as exc:
            print(f"WARNING: failed to save Kuhn policy to {final_path}: {exc}")
        if args.preview_random or args.preview_policy:
            print("Preview games are not supported for the Kuhn poker environment.")
        tlog.close()
        return

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
                         rollout_samples=args.rollout_samples,
                         reward_mode=args.reward_mode
                         )
    best_save_path = args.best_save_path if args.best_save_path else os.path.join(tlog.get_log_dir(), f"policy_{args.save_kind}_best.pkl")

    trainer.train(
        iterations=args.iters,
        seed=args.seed,
        verbose=args.verbose,
        log_every=args.log_every,
        eval_every=args.eval_every if args.eval_every > 0 else None,
        eval_episodes=args.eval_eps,
        eval_use_avg_policy=(args.eval_policy == "avg"),
        batch_size=args.batch_size,
        best_save_path=best_save_path,
        best_save_kind=args.save_kind,
        traversals_per_deal=args.traversals_per_deal,
        branch_topk_decay=args.branch_topk_decay,
        branch_topk_min=args.branch_topk_min,
        exploit_every=(args.exploit_every if args.exploit_every > 0 else None),
        exploit_episodes=(args.exploit_eps if args.exploit_eps > 0 else None),
        exploit_use_avg_policy=(args.exploit_policy == "avg"),
        debug_topk=args.debug_topk,
        debug_dir=debug_dir
    )

    # Save trained model/policy (default inside run dir)
    save_path = args.save_path if args.save_path else os.path.join(tlog.get_log_dir(), f"policy_{args.save_kind}.pkl")
    final_eval = trainer.evaluate(episodes=args.eval_eps, seed=args.seed + 1234567, selfplay=False,
                                  use_avg_policy=(args.eval_policy == "avg"))
    final_win0 = float(final_eval.get("win_rate_team0", 0.0))
    if best_save_path and final_win0 > trainer.best_vs_random_win_rate:
        trainer.best_vs_random_win_rate = final_win0
        try:
            trainer.save(best_save_path, kind=args.save_kind)
            print(f"Saved final policy as new best checkpoint to: {best_save_path} (win_rate_team0={final_win0:.3f})")
        except Exception as e:
            print(f"WARNING: failed to save best checkpoint to {best_save_path}: {e}")
    try:
        trainer.save(save_path, kind=args.save_kind)
        print(f"Saved {args.save_kind} checkpoint to: {save_path}")
    except Exception as e:
        print(f"WARNING: failed to save checkpoint to {save_path}: {e}")

    # Build env for previews and use the same logger so you can monitor
    env = make_env(tlog)

    # Optional: show a few random games to understand dynamics
    preview_counter = 0
    for _ in range(max(0, args.preview_random)):
        preview_game(env, policy="random", trainer=None, dump_dir=preview_dump_dir, label=f"{args.game}_random", episode_idx=preview_counter)
        preview_counter += 1

    # Optional: show a few games with the learned policy
    for _ in range(max(0, args.preview_policy)):
        preview_game(env, policy="policy", trainer=trainer, dump_dir=preview_dump_dir, label=f"{args.game}_policy", episode_idx=preview_counter)
        preview_counter += 1

    tlog.close()


if __name__ == "__main__":
    main()
