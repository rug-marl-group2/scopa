"""
Deep CFR trainer for 4-player imperfect-information games.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from deep_cfr.buffers import PolicyMemory, RegretMemory
from deep_cfr.evaluator import evaluate_selfplay, evaluate_vs_random
from deep_cfr.loggers import RunLogger
from deep_cfr.nets import FlexibleNet, masked_softmax
from deep_cfr.traverser import ExternalSamplingTraverser


class DeepCFRTrainer:
    """
    Orchestrates Deep CFR:
      - data collection via ExternalSamplingTraverser
      - training RegretNet/PolicyNet per seat
      - periodic evaluation + logging + checkpointing

    :param make_env_fn: function that creates a new environment instance
    :param regret_nets: List of 4 FlexibleNet regret networks, one per seat
    :param policy_nets: List of 4 FlexibleNet policy networks, one per seat
    :param regret_mems: List of 4 RegretMemory buffers, one per seat
    :param policy_mems: List of 4 PolicyMemory buffers, one per seat
    :param device: torch device for net inference/training
    :param lr_regret: learning rate for regret nets
    :param lr_policy: learning rate for policy nets
    :param weight_decay: weight decay (L2 regularization) for optimizers
    :param logger: optional RunLogger for logging training progress
    :param grad_clip: optional max norm for gradient clipping
    """

    def __init__(
        self,
        make_env_fn: Callable[[], any],
        regret_nets: List[FlexibleNet],
        policy_nets: List[FlexibleNet],
        regret_mems: List[RegretMemory],
        policy_mems: List[PolicyMemory],
        device: str = "cpu",
        lr_regret: float = 1e-3,
        lr_policy: float = 1e-3,
        weight_decay: float = 0.0,
        logger: Optional[RunLogger] = None,
        grad_clip: Optional[float] = None,
    ):
        self.make_env_fn = make_env_fn
        self.regret_nets = regret_nets
        self.policy_nets = policy_nets
        self.regret_mems = regret_mems
        self.policy_mems = policy_mems
        self.device = device
        self.grad_clip = grad_clip

        self.regret_opts = [
            optim.Adam(n.parameters(), lr=lr_regret, weight_decay=weight_decay)
            for n in regret_nets
        ]
        self.policy_opts = [
            optim.Adam(n.parameters(), lr=lr_policy, weight_decay=weight_decay)
            for n in policy_nets
        ]

        for n in self.regret_nets + self.policy_nets:
            n.to(self.device)

        self.logger = logger
        self.history: Dict[str, list] = {
            "regret_loss/seat0": [],
            "regret_loss/seat1": [],
            "regret_loss/seat2": [],
            "regret_loss/seat3": [],
            "policy_loss/seat0": [],
            "policy_loss/seat1": [],
            "policy_loss/seat2": [],
            "policy_loss/seat3": [],
            "winrate_vs_random": [],
            "scorediff_vs_random": [],
            "winrate_selfplay": [],
            "scorediff_selfplay": [],
        }
        self._global_iter = 0

    # ---------------- data collection ----------------
    def collect(self, traversals_per_seat: int, seed_offset: int = 0):
        t0 = time.perf_counter()
        traverser = ExternalSamplingTraverser(
            regret_nets=self.regret_nets,
            policy_nets=self.policy_nets,
            regret_mems=self.regret_mems,
            policy_mems=self.policy_mems,
            device=self.device,
            strict_illegal=True,
        )

        total = 4 * traversals_per_seat
        done = 0

        for seat in range(4):
            for k in range(traversals_per_seat):
                env = self.make_env_fn()
                env.reset(seed=seed_offset + 1000 * seat + k)
                traverser.traverse(env, target_seat=seat)
                done += 1

                # heartbeat: first, every 10, and last
                if done == 1 or done % 10 == 0 or done == total:
                    now = time.perf_counter()
                    dt = now - t0
                    rate = done / max(dt, 1e-9)
                    eta = (total - done) / max(rate, 1e-9)
                    sys.stdout.write(
                        f"\r[collect] iter={self._global_iter} "
                        f"seat={seat} k={k + 1}/{traversals_per_seat} "
                        f"done={done}/{total}  {rate:.1f} trav/s  ETA={eta:.1f}s"
                    )
                    sys.stdout.flush()

        print()  # end progress line
        t1 = time.perf_counter()
        print(
            f"[collect] iter={self._global_iter} traversals={total} time={t1 - t0:.2f}s  "
            f"buffers: R={[len(b.buf) for b in self.regret_mems]} "
            f"P={[len(b.buf) for b in self.policy_mems]}",
            flush=True,
        )

    # ---------------- losses ----------------
    @staticmethod
    def _mse_masked(
        pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        denom = mask.sum().clamp_min(1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    @staticmethod
    def _xent_masked_weighted(
        logits: torch.Tensor,
        target_probs: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        probs = masked_softmax(logits, mask)
        logp = (probs.clamp_min(1e-8)).log()
        loss_per = -(target_probs * logp * mask).sum(dim=-1) / mask.sum(
            dim=-1
        ).clamp_min(1.0)
        return (loss_per * weights).mean()

    # ---------------- single-seat train steps ----------------
    def train_regret(self, seat: int, batch_size: int, steps: int):
        """
        Train the RegretNet for a given seat.
        :param seat: seat index (0-3)
        :param batch_size: training batch size
        :param steps: number of gradient steps to take
        """
        net, opt = self.regret_nets[seat], self.regret_opts[seat]
        mem = self.regret_mems[seat]
        net.train()
        last_loss = None
        for _ in range(steps):
            if len(mem.buf) == 0:
                break
            infos, masks, advs = mem.sample(batch_size)
            infos, masks, advs = (
                infos.to(self.device),
                masks.to(self.device),
                advs.to(self.device),
            )
            preds = net(infos)
            loss = self._mse_masked(preds, advs, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            last_loss = loss.item()
        if last_loss is not None:
            key = f"regret_loss/seat{seat}"
            self.history[key].append(last_loss)
            if self.logger:
                self.logger.log_scalar(
                    key, last_loss, step=self._global_iter, split="train"
                )

    def train_policy(self, seat: int, batch_size: int, steps: int):
        """
        Train the PolicyNet for a given seat.
        :param seat: seat index (0-3)
        :param batch_size: training batch size
        :param steps: number of gradient steps to take
        """
        net, opt = self.policy_nets[seat], self.policy_opts[seat]
        mem = self.policy_mems[seat]
        net.train()
        last_loss = None
        for _ in range(steps):
            if len(mem.buf) == 0:
                break
            infos, masks, probs, weights = mem.sample(batch_size)
            infos, masks, probs = (
                infos.to(self.device),
                masks.to(self.device),
                probs.to(self.device),
            )
            weights = weights.to(self.device)
            logits = net(infos)
            loss = self._xent_masked_weighted(logits, probs, masks, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            last_loss = loss.item()
        if last_loss is not None:
            key = f"policy_loss/seat{seat}"
            self.history[key].append(last_loss)
            if self.logger:
                self.logger.log_scalar(
                    key, last_loss, step=self._global_iter, split="train"
                )

    # ---------------- quick eval every iteration ----------------
    def _mini_eval(self, n_games_small: int = 30):
        """
        Quick evaluation after each iteration to monitor progress.
        :param n_games_small: number of games to play for quick eval
        """
        wr, sd = evaluate_vs_random(
            self.policy_nets,
            self.make_env_fn,
            n_games=n_games_small,
            device=self.device,
        )
        sp_wr, sp_sd = evaluate_selfplay(
            self.policy_nets,
            self.make_env_fn,
            n_games=n_games_small,
            device=self.device,
        )
        print(
            f"[mini_eval] vs_random: winrate={wr:.3f} diff={sd:.3f} | "
            f"selfplay: winrate={sp_wr:.3f} diff={sp_sd:.3f}",
            flush=True,
        )

    # ---------------- main loop ----------------
    def run(
        self,
        iters: int = 200,
        traversals_per_seat: int = 512,
        batch_size: int = 512,
        regret_steps: int = 400,
        policy_steps: int = 400,
        eval_every: int = 0,
        save_every: int = 0,
    ):
        """
        Run the Deep CFR training loop.
        :param iters: number of Deep CFR iterations
        :param traversals_per_seat: number of external sampling traversals per seat per iteration
        :param batch_size: training batch size for both regret and policy nets
        :param regret_steps: number of gradient steps for regret nets per iteration
        :param policy_steps: number of gradient steps for policy nets per iteration
        :param eval_every: frequency of full evaluation (0 = no eval)
        :param save_every: frequency of checkpoint saving (0 = no saving)
        """
        for it in range(1, iters + 1):
            self._global_iter = it
            print(f"\n=== Iter {it}/{iters} ===", flush=True)
            t_iter0 = time.perf_counter()

            # 1) collect
            self.collect(traversals_per_seat, seed_offset=10_000 * it)

            # 2) train regret nets
            t0 = time.perf_counter()
            for seat in range(4):
                self.train_regret(seat, batch_size, regret_steps)
            t1 = time.perf_counter()
            print(
                f"[train_regret] seats=4 steps={regret_steps} time={t1 - t0:.2f}s",
                flush=True,
            )

            # 3) train policy nets
            t0 = time.perf_counter()
            for seat in range(4):
                self.train_policy(seat, batch_size, policy_steps)
            t1 = time.perf_counter()
            print(
                f"[train_policy] seats=4 steps={policy_steps} time={t1 - t0:.2f}s",
                flush=True,
            )

            # quick diagnostics
            print(
                "[loss] regret: "
                + ", ".join(
                    (
                        f"s{i}={self.history[f'regret_loss/seat{i}'][-1]:.4f}"
                        if self.history[f"regret_loss/seat{i}"]
                        else f"s{i}=None"
                    )
                    for i in range(4)
                )
            )
            print(
                "[loss] policy: "
                + ", ".join(
                    (
                        f"s{i}={self.history[f'policy_loss/seat{i}'][-1]:.4f}"
                        if self.history[f"policy_loss/seat{i}"]
                        else f"s{i}=None"
                    )
                    for i in range(4)
                )
            )
            print(
                f"[buffers] R={[len(b.buf) for b in self.regret_mems]} P={[len(b.buf) for b in self.policy_mems]}"
            )

            # 4) eval: full eval when scheduled, otherwise mini-eval
            if eval_every and it % eval_every == 0:
                t0 = time.perf_counter()
                wr, sd = evaluate_vs_random(
                    self.policy_nets, self.make_env_fn, n_games=200, device=self.device
                )
                sp_wr, sp_sd = evaluate_selfplay(
                    self.policy_nets, self.make_env_fn, n_games=200, device=self.device
                )
                t1 = time.perf_counter()
                print(
                    f"[eval] vs_random: winrate={wr:.3f} diff={sd:.3f} | "
                    f"selfplay: winrate={sp_wr:.3f} diff={sp_sd:.3f} "
                    f"time={t1 - t0:.2f}s",
                    flush=True,
                )
                if self.logger:
                    self.logger.log_scalar(
                        "winrate_vs_random", wr, step=it, split="eval"
                    )
                    self.logger.log_scalar(
                        "scorediff_vs_random", sd, step=it, split="eval"
                    )
                    self.logger.log_scalar(
                        "winrate_selfplay", sp_wr, step=it, split="eval"
                    )
                    self.logger.log_scalar(
                        "scorediff_selfplay", sp_sd, step=it, split="eval"
                    )
            else:
                self._mini_eval(n_games_small=30)

            # 5) save artifacts
            if save_every and it % save_every == 0 and self.logger:
                self.logger.save_checkpoint(
                    self.policy_nets, self.regret_nets, iter_id=it
                )
                self.logger.plot_history(self.history)
                print("[save] checkpoint + plots", flush=True)

            t_iter1 = time.perf_counter()
            print(f"[iter_done] total_time={t_iter1 - t_iter0:.2f}s", flush=True)
