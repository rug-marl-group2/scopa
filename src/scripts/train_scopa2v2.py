"""
Deep CFR training script for 4-player imperfect-information card game Scopone Scientifico.

Example usages:

Test run:
ipython src/scripts/train_scopa2v2.py -- --mode mlp --iters 2 --traversals_per_player 16 --regret_steps 10 --policy_steps 10

Wide MLP, more traversals, lower LR:
ipython src/scripts/train_scopa2v2.py -- --mode mlp --mlp_hidden 1024,512 --traversals_per_player 1024 --lr_regret 0.0005 --lr_policy 0.0005 --iters 200

Conv2D + MLP, default params:
ipython src/scripts/train_scopa2v2.py -- --mode conv2d_mlp --iters 100
"""

import argparse

import numpy as np
import torch

import src.games.scopa as env_mod
from src.deep_cfr.buffers import PolicyMemory, RegretMemory
from src.deep_cfr.loggers import RunLogger
from src.deep_cfr.nets import FlexibleNet
from src.deep_cfr.trainer import DeepCFRTrainer


def parse_int_list(s: str):
    """
    Parse a comma-separated string into a list of ints.

    :param s: Comma-separated string, e.g., "512,256"
    :return: List of integers, e.g., [512, 256]
    """
    # "512,256" -> [512, 256]; "" or None -> []
    if s is None or s == "":
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_float_list(s: str):
    """
    Parse a comma-separated string into a list of floats.

    :param s: Comma-separated string, e.g., "0.1,0.01"
    :return: List of floats, e.g., [0.1, 0.01]
    """

    if s is None or s == "":
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]


def build_argparser():
    """Build the argument parser for Deep CFR training."""

    p = argparse.ArgumentParser("Deep CFR trainer (Scopa/2v2)")

    # --- device/seed ---
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    p.add_argument("--seed", type=int, default=1234)

    # --- model mode ---
    p.add_argument("--mode", type=str, default="mlp", choices=["mlp", "conv2d_mlp"])

    # --- MLP config ---
    p.add_argument("--in_dim", type=int, default=200, help="Flattened obs size (5x40)")
    p.add_argument("--mlp_hidden", type=parse_int_list, default="512,256")
    p.add_argument(
        "--mlp_act",
        type=str,
        default="relu",
        choices=["relu", "gelu", "silu", "tanh", "elu", "lrelu", "none"],
    )
    p.add_argument(
        "--mlp_norm", type=str, default="none", choices=["none", "batch", "layer"]
    )
    p.add_argument("--mlp_dropout", type=float, default=0.0)
    p.add_argument("--mlp_residual", action="store_true")

    # --- Conv config (if conv2d_mlp) ---
    p.add_argument("--conv_input_C", type=int, default=1)
    p.add_argument("--conv_input_H", type=int, default=5)
    p.add_argument("--conv_input_W", type=int, default=40)
    p.add_argument("--conv_channels", type=parse_int_list, default="32,64")
    p.add_argument("--conv_kernels", type=parse_int_list, default="3,3")
    p.add_argument("--conv_strides", type=parse_int_list, default="1,1")
    p.add_argument("--conv_paddings", type=parse_int_list, default="1,1")
    p.add_argument("--conv_act", type=str, default="relu")
    p.add_argument(
        "--conv_norm", type=str, default="none", choices=["none", "batch", "layer"]
    )
    p.add_argument("--conv_dropout2d", type=float, default=0.0)
    p.add_argument("--conv_residual", action="store_true")

    # --- IO ---
    p.add_argument("--num_actions", type=int, default=40)

    # --- buffers ---
    p.add_argument("--regret_mem", type=int, default=100_000)
    p.add_argument("--policy_mem", type=int, default=100_000)

    # --- optim ---
    p.add_argument("--lr_regret", type=float, default=1e-3)
    p.add_argument("--lr_policy", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # --- schedule ---
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--traversals_per_player", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--regret_steps", type=int, default=300)
    p.add_argument("--policy_steps", type=int, default=300)

    # --- eval/log/save ---
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--results_dir", type=str, default="results/deep_cfr")
    p.add_argument("--no_tb", action="store_true", help="Disable TensorBoard")

    return p


def make_env():
    # No logger during training; feel free to wire TLogger here.
    return env_mod.env(tlogger=None, render_mode=None)


def build_nets(args, device):
    if args.mode == "mlp":
        # Input is flattened 4x40 -> 160 by default
        regret_nets = [
            FlexibleNet(
                mode="mlp",
                input_shape=(args.in_dim,),
                output_dim=args.num_actions,
                mlp_hidden=args.mlp_hidden,
                mlp_act=args.mlp_act,
                mlp_norm=args.mlp_norm,
                mlp_dropout=args.mlp_dropout,
                mlp_residual=args.mlp_residual,
            ).to(device)
            for _ in range(4)
        ]
        policy_nets = [
            FlexibleNet(
                mode="mlp",
                input_shape=(args.in_dim,),
                output_dim=args.num_actions,
                mlp_hidden=args.mlp_hidden,
                mlp_act=args.mlp_act,
                mlp_norm=args.mlp_norm,
                mlp_dropout=args.mlp_dropout,
                mlp_residual=args.mlp_residual,
            ).to(device)
            for _ in range(4)
        ]
    else:
        # conv2d_mlp: treat obs (4,40) as (C=1,H=4,W=40) by default
        input_shape = (args.conv_input_C, args.conv_input_H, args.conv_input_W)
        regret_nets = [
            FlexibleNet(
                mode="conv2d_mlp",
                input_shape=input_shape,
                output_dim=args.num_actions,
                conv_channels=args.conv_channels,
                conv_kernels=args.conv_kernels,
                conv_strides=args.conv_strides,
                conv_paddings=args.conv_paddings,
                conv_act=args.conv_act,
                conv_norm=args.conv_norm,
                conv_dropout2d=args.conv_dropout2d,
                conv_residual=args.conv_residual,
                mlp_hidden=args.mlp_hidden,
                mlp_act=args.mlp_act,
                mlp_norm=args.mlp_norm,
                mlp_dropout=args.mlp_dropout,
                mlp_residual=args.mlp_residual,
            ).to(device)
            for _ in range(4)
        ]
        policy_nets = [
            FlexibleNet(
                mode="conv2d_mlp",
                input_shape=input_shape,
                output_dim=args.num_actions,
                conv_channels=args.conv_channels,
                conv_kernels=args.conv_kernels,
                conv_strides=args.conv_strides,
                conv_paddings=args.conv_paddings,
                conv_act=args.conv_act,
                conv_norm=args.conv_norm,
                conv_dropout2d=args.conv_dropout2d,
                conv_residual=args.conv_residual,
                mlp_hidden=args.mlp_hidden,
                mlp_act=args.mlp_act,
                mlp_norm=args.mlp_norm,
                mlp_dropout=args.mlp_dropout,
                mlp_residual=args.mlp_residual,
            ).to(device)
            for _ in range(4)
        ]
    return regret_nets, policy_nets


def main():
    args = build_argparser().parse_args()
    device = torch.device(
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device in ["cuda", "cpu"] else "cpu")
    )
    torch.manual_seed(args.seed)

    # logger
    logger = RunLogger(
        logdir=args.logdir,
        results_dir=args.results_dir,
        use_tb=(not args.no_tb),
        run_name=None,
    )
    logger.save_args(args)

    # Nets & buffers
    regret_nets, policy_nets = build_nets(args, device)
    regret_mems = [
        RegretMemory(capacity=args.regret_mem, device=str(device), seed=args.seed + i)
        for i in range(4)
    ]
    policy_mems = [
        PolicyMemory(
            capacity=args.policy_mem, device=str(device), seed=args.seed + 100 + i
        )
        for i in range(4)
    ]

    # Trainer
    trainer = DeepCFRTrainer(
        make_env_fn=make_env,
        regret_nets=regret_nets,
        policy_nets=policy_nets,
        regret_mems=regret_mems,
        policy_mems=policy_mems,
        device=str(device),
        lr_regret=args.lr_regret,
        lr_policy=args.lr_policy,
        weight_decay=args.weight_decay,
        logger=logger,
    )

    trainer.run(
        iters=args.iters,
        traversals_per_player=args.traversals_per_player,
        batch_size=args.batch_size,
        regret_steps=args.regret_steps,
        policy_steps=args.policy_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
