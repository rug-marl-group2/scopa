"""Train Scopa agents with centralized training and decentralized execution."""
import argparse
import os
import time
from typing import Tuple

from tlogger import TLogger
from env import env as make_env
from ctde_trainer import CTDETrainer


def parse_hidden(value: str) -> Tuple[int, ...]:
    cleaned = value.strip()
    if not cleaned:
        return tuple()
    return tuple(int(part) for part in cleaned.split(',') if part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (outer loops)")
    parser.add_argument("--episodes_per_epoch", type=int, default=16, help="Episodes sampled per epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=0.2)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--actor_hidden", type=str, default="256,128", help="Comma-separated hidden sizes for actor MLP")
    parser.add_argument("--critic_hidden", type=str, default="256,128", help="Comma-separated hidden sizes for critic MLP")
    parser.add_argument("--target_tau", type=float, default=0.01, help="Soft-update rate for target networks")
    parser.add_argument("--target_interval", type=int, default=1, help="Gradient steps between target-network updates")
    parser.add_argument("--log_dir", type=str, default="", help="Custom log directory (defaults to timestamped run)")
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
    try:
        trainer.train(args.epochs, args.episodes_per_epoch)
    finally:
        tlog.close()


if __name__ == "__main__":
    main()
