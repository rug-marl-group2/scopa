"""Train Scopa agents with centralized training and decentralized execution."""
import argparse
import os
import time
from typing import Tuple

from tlogger import TLogger
from env import env as make_env
from ctde_trainer import CTDETrainer

import numpy as np


def parse_hidden(value: str) -> Tuple[int, ...]:
    cleaned = value.strip()
    if not cleaned:
        return tuple()
    return tuple(int(part) for part in cleaned.split(',') if part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2500, help="Training epochs (outer loops)")
    parser.add_argument("--episodes_per_epoch", type=int, default=16, help="Episodes sampled per epoch")
    parser.add_argument("--seed", type=int, default=232412)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.8)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--actor_hidden", type=str, default="256,128", help="Comma-separated hidden sizes for actor MLP")
    parser.add_argument("--critic_hidden", type=str, default="256,128", help="Comma-separated hidden sizes for critic MLP")
    parser.add_argument("--target_tau", type=float, default=0.01, help="Soft-update rate for target networks")
    parser.add_argument("--target_interval", type=int, default=10, help="Gradient steps between target-network updates")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate and log every N epochs (0 disables)")
    parser.add_argument("--eval_episodes", type=int, default=32, help="Episodes per evaluation run")
    parser.add_argument("--eval_policy", type=str, default="target", choices=["target", "online"], help="Actor parameters to use during evaluation")
    parser.add_argument("--eval_vs_random", action="store_true", help="Also evaluate against random opponents")
    parser.add_argument("--log_dir", type=str, default="", help="Custom log directory (defaults to timestamped run)")
    parser.add_argument("--save_path", type=str, default="", help="Path to save final trainer checkpoint")
    parser.add_argument("--best_save_path", type=str, default="", help="Path to save best checkpoint (defaults inside run dir)")
    args = parser.parse_args()

    actor_hidden = parse_hidden(args.actor_hidden)
    critic_hidden = parse_hidden(args.critic_hidden)

    default_log_root = os.path.join("scopa", "runs", "scopa_ctde")
    os.makedirs(default_log_root, exist_ok=True)
    if args.log_dir:
        log_dir = args.log_dir
    else:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(default_log_root, timestamp)

    tlog = TLogger(log_dir=log_dir)

    def env_factory():
        return make_env(tlog)

    trainer = CTDETrainer(env_fn=env_factory,
                          seed=args.seed,
                          actor_hidden=actor_hidden if actor_hidden else (256, 128),
                          critic_hidden=critic_hidden if critic_hidden else (256, 128),
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          gamma=args.gamma,
                          epsilon_start=args.epsilon_start,
                          epsilon_end=args.epsilon_end,
                          epsilon_decay=args.epsilon_decay,
                          target_tau=args.target_tau,
                          target_update_interval=args.target_interval,
                          tlogger=tlog)
    final_save_path = args.save_path if args.save_path else os.path.join(log_dir, "ctde_final.pkl")
    best_save_path = args.best_save_path if args.best_save_path else os.path.join(log_dir, "ctde_best.pkl")
    actor_source = "target" if args.eval_policy == "target" else "online"

    try:
        trainer.train(args.epochs, args.episodes_per_epoch,
                      eval_every=args.eval_every,
                      eval_episodes=args.eval_episodes,
                      eval_use_target=(args.eval_policy == "target"),
                      eval_vs_random=args.eval_vs_random)
    finally:
        try:
            final_metrics = trainer.evaluate(episodes=args.eval_episodes, epsilon=0.0,
                                             use_target=(args.eval_policy == "target"), incl_vs_random=True)
        except Exception as exc:
            print(f"WARNING: final evaluation failed: {exc}")
            final_metrics = {}

        win_rate = float(final_metrics.get("vs_random/win_rate", 0.0))
        if np.isfinite(win_rate) and win_rate > trainer.best_vs_random_win_rate:
            trainer.best_vs_random_win_rate = win_rate
            try:
                trainer.save(best_save_path, checkpoint_type="best", actor_source=actor_source)
                print(f"Saved final model as new best checkpoint to: {best_save_path} (win_rate_team0={win_rate:.3f})")
            except Exception as exc:
                print(f"WARNING: failed to save best checkpoint to {best_save_path}: {exc}")

        try:
            trainer.save(final_save_path, checkpoint_type="final", actor_source=actor_source)
            print(f"Saved final checkpoint to: {final_save_path}")
        except Exception as exc:
            print(f"WARNING: failed to save final checkpoint to {final_save_path}: {exc}")


if __name__ == "__main__":
    main()
