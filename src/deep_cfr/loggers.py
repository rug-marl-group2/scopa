"""
Deep CFR logging utilities for 4-player imperfect-information games.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch


class RunLogger:
    """
    Logger for Deep CFR training runs.
    Also logs to TensorBoard if available.

    :param logdir: base directory for logs
    :param results_dir: base directory for results
    :param use_tb: whether to use TensorBoard logging
    :param run_name: optional name for this run; if None, use timestamp
    """

    def __init__(
        self,
        logdir: str = "runs",
        results_dir: str = "results",
        use_tb: bool = True,
        run_name: Optional[str] = None,
    ):
        self.run_id = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.logdir = Path(logdir) / self.run_id
        self.results = Path(results_dir) / self.run_id
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.logdir))
            except Exception:
                self.tb = None
        self.csv_file = self.results / "metrics.csv"
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="") as f:
                csv.writer(f).writerow(["iter", "split", "metric", "value"])

    def log_scalar(self, tag: str, value: float, step: int, split: str = "train"):
        """
        Log a scalar metric.
        :param tag: metric name
        :param value: metric value
        :param step: training iteration or step
        :param split: data split (e.g., 'train', 'eval')
        """
        if self.tb:
            self.tb.add_scalar(f"{split}/{tag}", value, step)
        with open(self.csv_file, "a", newline="") as f:
            csv.writer(f).writerow([step, split, tag, value])

    def save_args(self, args: Any):
        with open(self.results / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    def save_checkpoint(
        self,
        policy_nets: List[torch.nn.Module],
        regret_nets: List[torch.nn.Module],
        iter_id: int,
    ):
        ckpt_dir = self.results / f"ckpt_{iter_id:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(policy_nets):
            torch.save(net.state_dict(), ckpt_dir / f"policy_{i}.pt")
        for i, net in enumerate(regret_nets):
            torch.save(net.state_dict(), ckpt_dir / f"regret_{i}.pt")

    def plot_history(self, history: Dict[str, List[float]]):
        # e.g., history keys: 'regret_loss/seat0', 'policy_loss/seat0', 'winrate_vs_random', ...
        for key, arr in history.items():
            if not arr:
                continue
            x = list(range(1, len(arr) + 1))
            plt.figure()
            plt.plot(x, arr)
            plt.xlabel("evaluation step")
            plt.ylabel(key)
            plt.title(key)
            out = self.results / f"{key.replace('/', '_')}.png"
            plt.savefig(out, bbox_inches="tight")
            plt.close()

    def close(self):
        if self.tb:
            self.tb.flush()
            self.tb.close()
