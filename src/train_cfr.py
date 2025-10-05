XLA_PYTHON_CLIENT_PREALLOCATE=False

import argparse
import os
import sys
import random
import time
from typing import Optional



from tlogger import TLogger
import env as env_mod
from env import env as make_env
from cfr_jax import CFRTrainer


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--branch_topk", type=int, default=3, help="Limit traverser branching to top-K actions by current policy (speed boost)")
    parser.add_argument("--preview_random", type=int, default=1, help="Show this many random games after training")
    parser.add_argument("--preview_policy", type=int, default=1, help="Show this many policy games after training")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate and log game metrics every N iters (0 to disable)")
    parser.add_argument("--eval_eps", type=int, default=64, help="Episodes per evaluation mode")
    parser.add_argument("--eval_policy", type=str, default="avg", choices=["avg", "current"], help="Which policy to use for evaluation")
    parser.add_argument("--save_path", type=str, default="", help="Path to save trained model/policy (.pkl)")
    parser.add_argument("--save_kind", type=str, default="avg", choices=["avg", "full"], help="Save average policy or full trainer state")
    args = parser.parse_args()

    # Create logger first so CFR can log metrics
    tlog = TLogger(log_dir="scopa/runs/scopa_cfr/"+time.strftime("%Y-%m-%d-%H-%M-%S"))
    trainer = CFRTrainer(seed=args.seed, tlogger=tlog, branch_topk=args.branch_topk)
    trainer.train(
        iterations=args.iters,
        seed=args.seed,
        verbose=args.verbose,
        log_every=args.log_every,
        eval_every=args.eval_every if args.eval_every > 0 else None,
        eval_episodes=args.eval_eps,
        eval_use_avg_policy=(args.eval_policy == "avg"),
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
