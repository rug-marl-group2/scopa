# src/deep_cfr/trainer.py  (parametric n-player version)

from __future__ import annotations
import sys, time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.deep_cfr.buffers import PolicyMemory, RegretMemory
from src.deep_cfr.evaluator import evaluate_selfplay, evaluate_vs_random
from src.deep_cfr.loggers import RunLogger
from src.deep_cfr.nets import FlexibleNet, masked_softmax
from src.deep_cfr.traverser import ExternalSamplingTraverser


class DeepCFRTrainer:
    """
    Deep CFR trainer (n-player, imperfect-information).

    - Data collection via ExternalSamplingTraverser (external sampling CFR)
    - Training regret/policy heads per player
    - Optional eval and checkpointing

    :param make_env_fn: Function that creates a new environment instance.
    :param regret_nets: List of FlexibleNet instances for each player's regret network.
    :param policy_nets: List of FlexibleNet instances for each player's policy network.
    :param regret_mems: List of RegretMemory instances for each player.
    :param policy_mems: List of PolicyMemory instances for each player.
    :param device: Device to run training on ("cpu" or "cuda").
    :param lr_regret: Learning rate for regret networks.
    :param lr_policy: Learning rate for policy networks.
    :param weight_decay: Weight decay (L2 regularization) for optimizers.
    :param logger: Optional RunLogger instance for logging and checkpointing.
    :param grad_clip: Optional gradient clipping value.
    """

    def __init__(
        self,
        make_env_fn: Callable[[], any],
        regret_nets: List[FlexibleNet],
        policy_nets: List[FlexibleNet],
        regret_mems: List[RegretMemory],
        policy_mems: List[PolicyMemory],
        device: str = "cuda",
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
        self.logger = logger

        # --- n-player inference & checks
        self.n_players = len(self.regret_nets)
        assert self.n_players >= 2, "Need at least 2 players."
        assert len(self.policy_nets) == self.n_players
        assert len(self.regret_mems) == self.n_players
        assert len(self.policy_mems) == self.n_players

        # --- opts
        self.regret_opts = [
            optim.Adam(n.parameters(), lr=lr_regret, weight_decay=weight_decay)
            for n in self.regret_nets
        ]
        self.policy_opts = [
            optim.Adam(n.parameters(), lr=lr_policy, weight_decay=weight_decay)
            for n in self.policy_nets
        ]
        for n in self.regret_nets + self.policy_nets:
            n.to(self.device)

        # --- history dict
        self.history: Dict[str, list] = {
            **{f"regret_loss/player{i}": [] for i in range(self.n_players)},
            **{f"policy_loss/player{i}": [] for i in range(self.n_players)},
            "winrate_vs_random": [],
            "scorediff_vs_random": [],
            "winrate_selfplay": [],
            "scorediff_selfplay": [],
        }
        self._global_iter = 0

    # ===================== Data collection (external sampling) =====================
    def collect(self, traversals_per_player: int, seed_offset: int = 0):
        t0 = time.perf_counter()
        traverser = ExternalSamplingTraverser(
            regret_nets=self.regret_nets,
            policy_nets=self.policy_nets,
            regret_mems=self.regret_mems,
            policy_mems=self.policy_mems,
            device=self.device,
            strict_illegal=True,  # rely on env illegal-mask repair
        )

        total = self.n_players * traversals_per_player
        done = 0

        for player in range(self.n_players):
            for k in range(traversals_per_player):
                env = self.make_env_fn()
                env.reset(seed=seed_offset + 1000 * player + k)
                # Traverser must interpret "target_seat=player" as "player index"
                traverser.traverse(env, target_seat=player)
                done += 1

                if done == 1 or done % 10 == 0 or done == total:
                    dt = max(time.perf_counter() - t0, 1e-9)
                    rate = done / dt
                    eta = (total - done) / max(rate, 1e-9)
                    sys.stdout.write(
                        f"\r[collect] iter={self._global_iter} "
                        f"player={player} k={k + 1}/{traversals_per_player} "
                        f"done={done}/{total}  {rate:.1f} trav/s  ETA={eta:.1f}s"
                    )
                    sys.stdout.flush()

        print()
        print(
            f"[collect] iter={self._global_iter} traversals={total}  "
            f"R_buf={[len(b.buf) for b in self.regret_mems]}  "
            f"P_buf={[len(b.buf) for b in self.policy_mems]}",
            flush=True,
        )

    # ===================== Loss helpers =====================
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

    # ===================== Per-player train steps =====================
    def train_regret(self, player: int, batch_size: int, steps: int):
        net, opt = self.regret_nets[player], self.regret_opts[player]
        mem = self.regret_mems[player]
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
            key = f"regret_loss/player{player}"
            self.history[key].append(last_loss)
            if self.logger:
                self.logger.log_scalar(
                    key, last_loss, step=self._global_iter, split="train"
                )

    def train_policy(self, player: int, batch_size: int, steps: int):
        net, opt = self.policy_nets[player], self.policy_opts[player]
        mem = self.policy_mems[player]
        net.train()
        last_loss = None
        for _ in range(steps):
            if len(mem.buf) == 0:
                break
            infos, masks, probs, weights = mem.sample(batch_size)
            infos, masks, probs, weights = (
                infos.to(self.device),
                masks.to(self.device),
                probs.to(self.device),
                weights.to(self.device),
            )
            logits = net(infos)
            loss = self._xent_masked_weighted(logits, probs, masks, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            last_loss = loss.item()
        if last_loss is not None:
            key = f"policy_loss/player{player}"
            self.history[key].append(last_loss)
            if self.logger:
                self.logger.log_scalar(
                    key, last_loss, step=self._global_iter, split="train"
                )

    # ===================== Quick eval (n-player aware) =====================
    def _mini_eval(self, n_games_small: int = 30):
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
            f"[mini_eval] vs_random: winrate={wr:.3f} diff={sd:.3f} | selfplay: winrate={sp_wr:.3f} diff={sp_sd:.3f}",
            flush=True,
        )

    # ===================== Main loop =====================
    def run(
        self,
        iters: int = 200,
        traversals_per_player: int = 512,
        batch_size: int = 512,
        regret_steps: int = 400,
        policy_steps: int = 400,
        eval_every: int = 0,
        save_every: int = 0,
    ):
        for it in range(1, iters + 1):
            self._global_iter = it
            print(f"\n=== Iter {it}/{iters} (players={self.n_players}) ===", flush=True)
            t_iter0 = time.perf_counter()

            # 1) collect
            self.collect(traversals_per_player, seed_offset=10_000 * it)

            # 2) train regret nets
            t0 = time.perf_counter()
            for pid in range(self.n_players):
                self.train_regret(pid, batch_size, regret_steps)
            print(
                f"[train_regret] players={self.n_players} steps={regret_steps} time={time.perf_counter() - t0:.2f}s",
                flush=True,
            )

            # 3) train policy nets
            t0 = time.perf_counter()
            for pid in range(self.n_players):
                self.train_policy(pid, batch_size, policy_steps)
            print(
                f"[train_policy] players={self.n_players} steps={policy_steps} time={time.perf_counter() - t0:.2f}s",
                flush=True,
            )

            # Diagnostics
            print(
                "[loss] regret: "
                + ", ".join(
                    (
                        f"p{i}={self.history[f'regret_loss/player{i}'][-1]:.4f}"
                        if self.history[f"regret_loss/player{i}"]
                        else f"p{i}=None"
                    )
                    for i in range(self.n_players)
                )
            )
            print(
                "[loss] policy: "
                + ", ".join(
                    (
                        f"p{i}={self.history[f'policy_loss/player{i}'][-1]:.4f}"
                        if self.history[f"policy_loss/player{i}"]
                        else f"p{i}=None"
                    )
                    for i in range(self.n_players)
                )
            )
            print(
                f"[buffers] R={[len(b.buf) for b in self.regret_mems]}  P={[len(b.buf) for b in self.policy_mems]}"
            )

            # 4) eval
            if eval_every and it % eval_every == 0:
                t0 = time.perf_counter()
                wr, sd = evaluate_vs_random(
                    self.policy_nets, self.make_env_fn, n_games=200, device=self.device
                )
                sp_wr, sp_sd = evaluate_selfplay(
                    self.policy_nets, self.make_env_fn, n_games=200, device=self.device
                )
                print(
                    f"[eval] vs_random: winrate={wr:.3f} diff={sd:.3f} | selfplay: winrate={sp_wr:.3f} diff={sp_sd:.3f}  time={time.perf_counter() - t0:.2f}s",
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

            # 5) save
            if save_every and it % save_every == 0 and self.logger:
                self.logger.save_checkpoint(
                    self.policy_nets, self.regret_nets, iter_id=it
                )
                self.logger.plot_history(self.history)
                print("[save] checkpoint + plots", flush=True)

            print(
                f"[iter_done] total_time={time.perf_counter() - t_iter0:.2f}s",
                flush=True,
            )
