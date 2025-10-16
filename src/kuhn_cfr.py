"""
Lightweight CFR trainer for Kuhn Poker.

This module provides a minimal implementation that mirrors the basic
interfaces used by ``train_cfr.py`` so the existing logging and checkpoint
machinery can be reused when testing the CFR loop on a toy game.
"""

from __future__ import annotations

import pickle
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np


ACTION_LABELS = ("p", "b")  # pass/check, bet/call
TERMINAL_HISTORIES = ("pp", "bb", "bp", "pbp", "pbb")


@dataclass
class KuhnNode:
    """Stores regrets and average strategy totals for a single infoset."""

    regret_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    strategy_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def current_strategy(self, rm_plus: bool) -> np.ndarray:
        regrets = np.maximum(self.regret_sum, 0.0) if rm_plus else self.regret_sum.copy()
        normalizer = regrets.sum()
        if normalizer <= 0.0:
            return np.full(2, 0.5, dtype=np.float64)
        return regrets / normalizer

    def get_strategy(self, reach_prob: float, rm_plus: bool) -> np.ndarray:
        strategy = self.current_strategy(rm_plus)
        self.strategy_sum += reach_prob * strategy
        return strategy

    def average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total <= 0.0:
            return np.full(2, 0.5, dtype=np.float64)
        return self.strategy_sum / total

    def to_dict(self) -> Dict[str, Iterable[float]]:
        return {
            "regret_sum": self.regret_sum.tolist(),
            "strategy_sum": self.strategy_sum.tolist(),
            "avg_strategy": self.average_strategy().tolist(),
        }


class KuhnCFRTrainer:
    """Tabular CFR for Kuhn Poker with an interface similar to the Scopa trainer."""

    def __init__(self, seed: int = 0, tlogger: Optional[object] = None, rm_plus: bool = True):
        self.tlogger = tlogger
        self.rm_plus = bool(rm_plus)
        self.rng = np.random.default_rng(seed)
        self.info_sets: Dict[str, KuhnNode] = {}
        self.iteration = 0
        self.best_avg_value = float("-inf")
        self.best_policy_snapshot: Optional[Dict[str, Dict[str, Iterable[float]]]] = None
        self._iter_stats: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def _begin_iteration_stats(self) -> None:
        self._iter_stats = {}

    def _stat_add(self, key: str, value: float) -> None:
        if self._iter_stats is None:
            return
        self._iter_stats[key] = self._iter_stats.get(key, 0.0) + float(value)

    def _stat_record_max(self, key: str, value: float) -> None:
        if self._iter_stats is None:
            return
        v = float(value)
        cur = self._iter_stats.get(key)
        if cur is None or v > cur:
            self._iter_stats[key] = v

    def _finalize_iteration_stats(self, iteration: int, emit: bool) -> None:
        stats = self._iter_stats
        self._iter_stats = None
        if not emit or self.tlogger is None or stats is None:
            return
        writer = self.tlogger.writer
        calls = stats.get("cfr_calls", 0.0)
        if calls > 0:
            writer.add_scalar("KuhnDebug/cfr_calls", calls, iteration)
            depth_avg = stats.get("depth_total", 0.0) / calls
            writer.add_scalar("KuhnDebug/avg_depth", depth_avg, iteration)
        writer.add_scalar("KuhnDebug/max_depth", stats.get("max_depth", 0.0), iteration)
        terminals = stats.get("terminal_calls", 0.0)
        writer.add_scalar("KuhnDebug/terminal_calls", terminals, iteration)
        if terminals > 0:
            term_sum = stats.get("terminal_sum", 0.0)
            term_sumsq = stats.get("terminal_sumsq", 0.0)
            term_mean = term_sum / terminals
            variance = max(term_sumsq / terminals - term_mean * term_mean, 0.0)
            writer.add_scalar("KuhnDebug/terminal_mean", term_mean, iteration)
            writer.add_scalar("KuhnDebug/terminal_std", variance ** 0.5, iteration)
        writer.add_scalar("KuhnDebug/nodes_touched", stats.get("nodes_touched", 0.0), iteration)

    def dump_top_infosets(self, iteration: int, top_k: int, path: str) -> None:
        if top_k <= 0:
            return
        entries = []
        for key, node in self.info_sets.items():
            regrets = np.asarray(node.regret_sum, dtype=np.float64)
            abs_regrets = np.abs(regrets)
            max_reg = float(abs_regrets.max())
            mean_reg = float(abs_regrets.mean())
            strategy = node.average_strategy()
            entropy = float(-np.sum(np.where(strategy > 1e-12, strategy * np.log2(np.clip(strategy, 1e-12, 1.0)), 0.0)))
            entries.append(
                {
                    "infoset": key,
                    "max_regret": max_reg,
                    "mean_abs_regret": mean_reg,
                    "strategy": strategy.tolist(),
                    "entropy": entropy,
                }
            )
        if not entries:
            return
        entries.sort(key=lambda item: item["max_regret"], reverse=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "iteration": iteration,
            "top_k": top_k,
            "entries": entries[:top_k],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # Core CFR recursion
    # ------------------------------------------------------------------
    def _cfr(self, cards: np.ndarray, history: str, reach_p0: float, reach_p1: float) -> float:
        self._stat_add("cfr_calls", 1.0)
        depth = len(history)
        self._stat_add("depth_total", float(depth))
        self._stat_record_max("max_depth", float(depth))
        self._stat_add("nodes_touched", 1.0)
        if history in TERMINAL_HISTORIES:
            payoff = self._terminal_utility(history, cards)
            self._stat_add("terminal_calls", 1.0)
            self._stat_add("terminal_sum", float(payoff))
            self._stat_add("terminal_sumsq", float(payoff * payoff))
            return payoff

        plays = len(history)
        player = plays % 2
        card = int(cards[player])
        info_key = f"{card}:{history}"
        node = self.info_sets.setdefault(info_key, KuhnNode())
        reach = reach_p0 if player == 0 else reach_p1
        strategy = node.get_strategy(reach, self.rm_plus)

        util = np.zeros(2, dtype=np.float64)
        node_util = 0.0
        for action_idx, action in enumerate(ACTION_LABELS):
            next_history = history + action
            if player == 0:
                util[action_idx] = -self._cfr(cards, next_history, reach_p0 * strategy[action_idx], reach_p1)
            else:
                util[action_idx] = -self._cfr(cards, next_history, reach_p0, reach_p1 * strategy[action_idx])
            node_util += strategy[action_idx] * util[action_idx]

        regret = util - node_util
        opp_reach = reach_p1 if player == 0 else reach_p0
        self._stat_add("regret_updates", 1.0)
        node.regret_sum += opp_reach * regret
        return node_util

    @staticmethod
    def _terminal_utility(history: str, cards: np.ndarray) -> float:
        card0, card1 = int(cards[0]), int(cards[1])
        if history == "pp":
            return 1.0 if card0 > card1 else -1.0
        if history == "bb":
            return 2.0 if card0 > card1 else -2.0
        if history == "bp":
            return 1.0
        if history == "pbp":
            return -1.0
        if history == "pbb":
            return 2.0 if card0 > card1 else -2.0
        raise ValueError(f"Unknown terminal history: {history}")

    # ------------------------------------------------------------------
    # Public API consumed by train_cfr.py
    # ------------------------------------------------------------------
    def train(
        self,
        iterations: int,
        seed: Optional[int] = None,
        verbose: bool = False,
        log_every: int = 100,
        eval_every: Optional[int] = None,
        eval_episodes: int = 1000,
        eval_use_avg_policy: bool = True,
        batch_size: int = 1,
        best_save_path: Optional[str] = None,
        best_save_kind: str = "avg",
        traversals_per_deal: int = 1,
        branch_topk_decay: float = 0.0,
        branch_topk_min: int = 0,
        exploit_every: Optional[int] = None,
        exploit_episodes: Optional[int] = None,
        exploit_use_avg_policy: bool = True,
        debug_topk: Optional[int] = None,
        debug_dir: Optional[str] = None,
    ) -> None:
        # Unused parameters are accepted for compatibility with the Scopa trainer.
        del seed, batch_size, best_save_kind, traversals_per_deal
        del branch_topk_decay, branch_topk_min, exploit_every, exploit_episodes, exploit_use_avg_policy

        debug_topk = int(debug_topk) if (debug_topk is not None and int(debug_topk) > 0) else None
        debug_dump_dir = None
        if debug_topk:
            if debug_dir:
                debug_dump_dir = debug_dir
            elif self.tlogger is not None:
                debug_dump_dir = os.path.join(self.tlogger.get_log_dir(), "debug")
            else:
                debug_dump_dir = os.path.join(os.getcwd(), "kuhn_debug")
            os.makedirs(debug_dump_dir, exist_ok=True)

        for it in range(1, iterations + 1):
            self._begin_iteration_stats()
            cards = np.array([1, 2, 3], dtype=np.int32)
            self.rng.shuffle(cards)
            self._stat_add("deals", 1.0)
            self._cfr(cards, "", 1.0, 1.0)
            self.iteration += 1

            if verbose and it % max(log_every, 1) == 0:
                print(f"[Kuhn] iter {it:06d} | infosets={len(self.info_sets)} | avg_value={self.expected_value(eval_use_avg_policy):+.4f}")

            should_emit = self.tlogger is not None and it % max(log_every, 1) == 0
            if should_emit:
                avg_value = self.expected_value(eval_use_avg_policy)
                avg_regret = self.average_abs_regret()
                self.tlogger.writer.add_scalar("Kuhn/expected_value", float(avg_value), it)
                self.tlogger.writer.add_scalar("Kuhn/avg_abs_regret", float(avg_regret), it)
                self.tlogger.writer.add_scalar("Kuhn/num_infosets", float(len(self.info_sets)), it)
                if debug_topk:
                    dump_path = os.path.join(debug_dump_dir, f"infosets_{it:06d}.json") if debug_dump_dir else None
                    if dump_path is not None:
                        self.dump_top_infosets(it, debug_topk, dump_path)
            self._finalize_iteration_stats(it, emit=should_emit)

            if eval_every is not None and eval_every > 0 and it % eval_every == 0:
                metrics = self.evaluate(episodes=eval_episodes, use_avg_policy=eval_use_avg_policy)
                value = float(metrics.get("player0_value", 0.0))
                if verbose:
                    print(f"[Kuhn] eval@{it}: player0_value={value:+.4f}")
                if self.tlogger is not None:
                    self.tlogger.writer.add_scalar("Eval/player0_value", value, it)
                if best_save_path is not None and value > self.best_avg_value:
                    self.best_avg_value = value
                    self.best_policy_snapshot = self.export_policy()
                    try:
                        self.save(best_save_path)
                    except Exception as exc:
                        print(f"WARNING: failed to save Kuhn policy to {best_save_path}: {exc}")

    def average_abs_regret(self) -> float:
        if not self.info_sets:
            return 0.0
        total = 0.0
        count = 0
        for node in self.info_sets.values():
            total += float(np.abs(node.regret_sum).sum())
            count += node.regret_sum.size
        return total / float(count)

    def expected_value(self, use_avg_policy: bool = True) -> float:
        metrics = self.evaluate(episodes=2048, use_avg_policy=use_avg_policy)
        return float(metrics.get("player0_value", 0.0))

    def evaluate(
        self,
        episodes: int = 1000,
        seed: Optional[int] = None,
        selfplay: bool = True,
        use_avg_policy: bool = True,
    ) -> Dict[str, float]:
        if episodes <= 0:
            return {"player0_value": 0.0}
        rng = np.random.default_rng(seed if seed is not None else self.rng.integers(1, 1 << 32))
        total = 0.0
        for _ in range(episodes):
            cards = np.array([1, 2, 3], dtype=np.int32)
            rng.shuffle(cards)
            total += self._simulate_episode(cards, rng, use_avg_policy)
        avg = total / float(episodes)
        return {"player0_value": float(avg), "selfplay": float(bool(selfplay))}

    def _simulate_episode(self, cards: np.ndarray, rng: np.random.Generator, use_avg_policy: bool) -> float:
        history = ""
        while history not in TERMINAL_HISTORIES:
            player = len(history) % 2
            card = int(cards[player])
            info_key = f"{card}:{history}"
            node = self.info_sets.get(info_key)
            if node is None:
                strategy = np.full(2, 0.5, dtype=np.float64)
            else:
                strategy = node.average_strategy() if use_avg_policy else node.current_strategy(self.rm_plus)
            action_idx = int(rng.choice(2, p=strategy))
            history += ACTION_LABELS[action_idx]
        return self._terminal_utility(history, cards)

    def save(self, path: str) -> None:
        data = {"policy": self.export_policy(), "iteration": self.iteration}
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def export_policy(self) -> Dict[str, Dict[str, Iterable[float]]]:
        return {infoset: node.to_dict() for infoset, node in self.info_sets.items()}

    # Compatibility placeholders ------------------------------------------------
    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:  # pragma: no cover - unused
        raise NotImplementedError("Kuhn trainer does not integrate with Scopa observations.")

    @property
    def best_vs_random_win_rate(self) -> float:
        return self.best_avg_value

    @best_vs_random_win_rate.setter
    def best_vs_random_win_rate(self, value: float) -> None:
        self.best_avg_value = value
