from typing import Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import os
import pickle
import time
import json

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import trange
import hashlib


# --------------------------
# Numpy-based Environment
# --------------------------

NUM_SUITS = 4
NUM_RANKS = 10
NUM_CARDS = NUM_SUITS * NUM_RANKS

# Index metadata
CARD_RANKS = np.concatenate([np.arange(1, NUM_RANKS + 1, dtype=np.int32) for _ in range(NUM_SUITS)])
CARD_SUITS = np.concatenate([np.full((NUM_RANKS,), si, dtype=np.int32) for si in range(NUM_SUITS)])
TEAM0_MASK = np.array([1, 0, 1, 0], dtype=np.int32)
TEAM1_MASK = np.array([0, 1, 0, 1], dtype=np.int32)
TEAM_INDEX = np.array([0, 1, 0, 1], dtype=np.int32)
PRIMIERA_PRIORITY = np.array([0, 2, 0, 0, 0, 1, 3, 4, 0, 0, 0], dtype=np.int32)


class NState:
    """Lightweight NumPy state for CFR traversals."""
    def __init__(self):
        self.hands = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.table = np.zeros((NUM_CARDS,), dtype=np.int8)
        self.captures = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.history = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.scopas = np.zeros((4,), dtype=np.int32)
        self.cur_player = np.int32(0)
        self.last_capture_player = np.int32(-1)




@dataclass
class ActionDiff:
    seat: int
    action_idx: int
    hand_prev: int
    history_prev: int
    table_indices: np.ndarray
    table_prev: np.ndarray
    captures_indices: np.ndarray
    scopa_delta: int
    last_capture_prev: int
    cur_player_prev: int


@dataclass
class EvaluationSchedule:
    """Utility container that governs periodic evaluation runs."""

    every: Optional[int]
    episodes: int
    use_avg_policy: bool

    def is_enabled(self) -> bool:
        return self.every is not None and self.every > 0 and self.episodes > 0

    def should_run(self, iteration: int) -> bool:
        return self.is_enabled() and iteration % int(self.every) == 0

    def resolved_episodes(self) -> int:
        return max(int(self.episodes), 1)


@dataclass
class ExploitabilitySchedule:
    """Schedule controlling exploitability estimation frequency."""

    every: Optional[int]
    episodes: Optional[int]
    use_avg_policy: bool

    def is_enabled(self) -> bool:
        return self.every is not None and self.every > 0

    def should_run(self, iteration: int) -> bool:
        return self.is_enabled() and iteration % int(self.every) == 0

    def resolved_episodes(self, fallback: int) -> int:
        eps = self.episodes if self.episodes is not None and self.episodes > 0 else fallback
        return max(int(eps), 1)


@dataclass
class ExploitabilityConfig:
    """Configuration for Monte Carlo exploitability estimation."""

    policy_rollouts: int = 4
    best_response_rollouts: int = 4
    opponent_samples: int = 1

    def sanitized(self) -> "ExploitabilityConfig":
        return ExploitabilityConfig(
            policy_rollouts=max(int(self.policy_rollouts), 1),
            best_response_rollouts=max(int(self.best_response_rollouts), 1),
            opponent_samples=max(int(self.opponent_samples), 1),
        )


def _apply_action_core(
    st: NState,
    action_idx: int,
    subset_mask: Optional[np.ndarray] = None,
    mask_resolver: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    record_diff: bool = False,
) -> Tuple[Optional[ActionDiff], bool]:
    """Shared Scopa action application used by env step and CFR traversal."""

    seat = int(st.cur_player)
    diff: Optional[ActionDiff] = None
    if record_diff:
        diff = ActionDiff(
            seat=seat,
            action_idx=int(action_idx),
            hand_prev=int(st.hands[seat, action_idx]),
            history_prev=int(st.history[seat, action_idx]),
            table_indices=np.empty(0, dtype=np.int32),
            table_prev=np.empty(0, dtype=np.int8),
            captures_indices=np.empty(0, dtype=np.int32),
            scopa_delta=0,
            last_capture_prev=int(st.last_capture_player),
            cur_player_prev=int(st.cur_player),
        )

    st.hands[seat, action_idx] = 0
    st.history[seat, action_idx] = 1
    rank = int(CARD_RANKS[action_idx])
    scopa = False

    if rank == 1:
        table_indices = np.nonzero(st.table)[0]
        if table_indices.size > 0:
            if diff is not None:
                diff.table_indices = table_indices
                diff.table_prev = st.table[table_indices].copy()
            st.table[table_indices] = 0
        captured = np.zeros((NUM_CARDS,), dtype=np.int8)
        if table_indices.size > 0:
            captured[table_indices] = 1
        captured[action_idx] = 1
        cap_idx = np.nonzero(captured)[0]
        if cap_idx.size > 0:
            if diff is not None:
                diff.captures_indices = cap_idx
            st.captures[seat, cap_idx] += 1
        st.last_capture_player = np.int32(seat)
    else:
        mask = subset_mask
        if mask is None and mask_resolver is not None:
            mask = mask_resolver(st.table, rank)
        mask_bool = None if mask is None else np.asarray(mask, dtype=bool)
        if mask_bool is not None and mask_bool.any():
            table_indices = np.nonzero(mask_bool)[0]
            if diff is not None:
                diff.table_indices = table_indices
                diff.table_prev = st.table[table_indices].copy()
            st.table[table_indices] = 0
            captured = mask_bool.astype(np.int8)
            captured[action_idx] = 1
            cap_idx = np.nonzero(captured)[0]
            if cap_idx.size > 0:
                if diff is not None:
                    diff.captures_indices = cap_idx
                st.captures[seat, cap_idx] += 1
            scopa = bool(st.table.sum() == 0)
            if scopa:
                st.scopas[seat] += 1
                if diff is not None:
                    diff.scopa_delta = 1
            st.last_capture_player = np.int32(seat)
        else:
            if diff is not None:
                diff.table_indices = np.array([action_idx], dtype=np.int32)
                diff.table_prev = np.array([st.table[action_idx]], dtype=np.int8)
            st.table[action_idx] = 1

    st.cur_player = np.int32((seat + 1) % 4)
    return diff, scopa

def np_init_state(rng: Generator) -> NState:
    st = NState()
    perm = rng.permutation(NUM_CARDS)
    for seat in range(4):
        idx = perm[seat * 10:(seat + 1) * 10]
        st.hands[seat, idx] = 1
    st.cur_player = np.int32(rng.integers(0, 4))
    return st


def np_clone_state(st: NState) -> NState:
    clone = NState()
    clone.hands[...] = st.hands
    clone.table[...] = st.table
    clone.captures[...] = st.captures
    clone.history[...] = st.history
    clone.scopas[...] = st.scopas
    clone.cur_player = np.int32(st.cur_player)
    clone.last_capture_player = np.int32(st.last_capture_player)
    return clone


def np_legal_mask(st: NState) -> np.ndarray:
    return st.hands[int(st.cur_player)]


def np_build_obs(st: NState, seat: int) -> np.ndarray:
    friend = (seat + 2) % 4
    enemy1 = (seat + 1) % 4
    enemy2 = (seat + 3) % 4
    out = np.zeros((6, NUM_CARDS), dtype=np.float32)
    out[0] = st.hands[seat]
    out[1] = st.table
    out[2] = np.clip(st.captures[seat] + st.captures[friend], 0, 1)
    out[3] = st.history[enemy1]
    out[4] = st.history[enemy2]
    out[5] = st.history[friend]
    return out



_EMPTY_DP_HAS = np.zeros((NUM_RANKS + 1,), dtype=bool)
_EMPTY_DP_HAS[0] = True
_EMPTY_DP_MASK = np.zeros((NUM_RANKS + 1, NUM_CARDS), dtype=bool)


def _subset_sum_dp_numpy(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if table.sum() <= 0:
        return _EMPTY_DP_HAS.copy(), _EMPTY_DP_MASK.copy()
    dp_has = np.zeros((NUM_RANKS + 1,), dtype=bool)
    dp_has[0] = True
    dp_mask = np.zeros((NUM_RANKS + 1, NUM_CARDS), dtype=bool)
    table_indices = np.nonzero(table)[0]
    for idx in table_indices:
        rank = int(CARD_RANKS[idx])
        if rank <= 0 or rank > NUM_RANKS:
            continue
        for s in range(NUM_RANKS, rank - 1, -1):
            if (not dp_has[s]) and dp_has[s - rank]:
                dp_has[s] = True
                row = dp_mask[s - rank].copy()
                row[idx] = True
                dp_mask[s] = row
    return dp_has, dp_mask


def _subset_sum_mask(table: np.ndarray, target_rank: int) -> np.ndarray:
    if target_rank <= 0 or target_rank > NUM_RANKS:
        return np.zeros((NUM_CARDS,), dtype=bool)
    dp_has, dp_mask = _subset_sum_dp_numpy(table)
    if target_rank >= dp_has.size or not bool(dp_has[target_rank]):
        return np.zeros((NUM_CARDS,), dtype=bool)
    return np.asarray(dp_mask[target_rank], dtype=bool).copy()


def np_step(st: NState, action_idx: int) -> tuple[NState, bool]:
    _, scopa = _apply_action_core(
        st,
        int(action_idx),
        mask_resolver=_subset_sum_mask,
        record_diff=False,
    )
    return st, scopa


def np_is_terminal(st: NState) -> bool:
    return int(st.hands.sum()) == 0


def np_final_captures(st: NState) -> np.ndarray:
    captures = st.captures.copy()
    if int(st.last_capture_player) >= 0:
        seat = int(st.last_capture_player)
        captures[seat] = np.clip(captures[seat] + st.table, 0, 1)
    return captures


def _distribute_point(team_points: np.ndarray, player_points: np.ndarray, team_idx: int, seats: tuple[int, int], totals: float, counts: np.ndarray, value: float = 1.0) -> None:
    """Add `value` reward to a team and share to individual seats."""
    team_points[team_idx] += value
    if totals > 0.0:
        weights = [counts[seat] / float(totals) for seat in seats]
    else:
        weights = [1.0 / len(seats)] * len(seats)
    for seat, w in zip(seats, weights):
        player_points[seat] += value * float(w)


def np_round_scores(st: NState) -> tuple[np.ndarray, np.ndarray]:
    """Return (team_points, player_points) for a terminal state."""
    captures = np_final_captures(st)

    team0_caps = (captures.T @ TEAM0_MASK).astype(np.int32)
    team1_caps = (captures.T @ TEAM1_MASK).astype(np.int32)
    card_counts = captures.sum(axis=1).astype(np.int32)
    team_card_counts = np.array([
        int(card_counts[0] + card_counts[2]),
        int(card_counts[1] + card_counts[3]),
    ], dtype=np.int32)

    is_bello = (CARD_SUITS == 3).astype(np.int32)
    coin_counts = (captures * is_bello).sum(axis=1).astype(np.int32)
    team_coin_counts = np.array([
        int(coin_counts[0] + coin_counts[2]),
        int(coin_counts[1] + coin_counts[3]),
    ], dtype=np.int32)

    player_points = np.zeros(4, dtype=np.float32)
    team_points = np.zeros(2, dtype=np.float32)

    # Scopa points accrue to the player who completed them.
    scopa_counts = st.scopas.astype(np.float32)
    player_points += scopa_counts
    team_points[0] += float(scopa_counts[0] + scopa_counts[2])
    team_points[1] += float(scopa_counts[1] + scopa_counts[3])

    # Most cards bonus.
    if team_card_counts[0] > team_card_counts[1]:
        _distribute_point(team_points, player_points, 0, (0, 2), float(team_card_counts[0]), card_counts)
    elif team_card_counts[1] > team_card_counts[0]:
        _distribute_point(team_points, player_points, 1, (1, 3), float(team_card_counts[1]), card_counts)

    # Most coins bonus.
    if team_coin_counts[0] > team_coin_counts[1]:
        _distribute_point(team_points, player_points, 0, (0, 2), float(team_coin_counts[0]), coin_counts)
    elif team_coin_counts[1] > team_coin_counts[0]:
        _distribute_point(team_points, player_points, 1, (1, 3), float(team_coin_counts[1]), coin_counts)

    # Sette bello goes to the player who captured it.
    sette_bello_idx = 3 * NUM_RANKS + (7 - 1)
    sette_owner = np.nonzero(captures[:, sette_bello_idx])[0]
    if sette_owner.size > 0:
        seat = int(sette_owner[0])
        team_idx = TEAM_INDEX[seat]
        team_points[team_idx] += 1.0
        player_points[seat] += 1.0

    # Primiera bonus: distribute according to each player's contribution.
    player_primiera = np.zeros(4, dtype=np.float32)
    pri = PRIMIERA_PRIORITY[1:]
    for seat in range(4):
        cards = captures[seat].reshape(NUM_SUITS, NUM_RANKS)
        player_primiera[seat] = float((cards * pri).max(axis=1).sum())
    team_primiera = np.array([
        float(player_primiera[0] + player_primiera[2]),
        float(player_primiera[1] + player_primiera[3]),
    ], dtype=np.float32)
    if team_primiera[0] > team_primiera[1]:
        _distribute_point(team_points, player_points, 0, (0, 2), float(team_primiera[0]), player_primiera)
    elif team_primiera[1] > team_primiera[0]:
        _distribute_point(team_points, player_points, 1, (1, 3), float(team_primiera[1]), player_primiera)

    return team_points, player_points


def np_evaluate_round(st: NState) -> tuple[int, int]:
    team_points, _ = np_round_scores(st)
    if team_points[0] > team_points[1]:
        return 1, -1
    if team_points[1] > team_points[0]:
        return -1, 1
    return 0, 0


# --------------------------
# CFR Trainer (NumPy)
# --------------------------

def regret_matching(regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Convert cumulative regrets to a stochastic policy over legal actions."""
    pos = np.maximum(regrets, 0.0) * legal_mask
    s = pos.sum()
    if s > 0:
        return pos / s
    # Uniform over legal
    denom = max(int(legal_mask.sum()), 1)
    return legal_mask.astype(np.float32) / denom


class CFRTrainer:
    """NumPy CFR trainer with hashed infosets and memory guards."""
    def __init__(self, seed: int = 42, tlogger: Optional[object] = None, branch_topk: Optional[int] = None,
                 max_infosets: Optional[int] = None, obs_key_mode: str = "full",
                 dtype: np.dtype = np.float16, rm_plus: bool = True, rng: Optional[Generator] = None,
                 progressive_widening: bool = True, pw_alpha: float = 0.3, pw_tail: int = 3,
                 regret_prune: bool = True, prune_threshold: float = -0.05,
                 prune_warmup: int = 32, prune_reactivation: int = 8,
                 subset_cache_size: int = 8192, max_branch_actions: int = 0,
                 rollout_depth: int = 0, rollout_samples: int = 0, reward_mode: str = "team",
                 exploit_config: Optional[ExploitabilityConfig] = None):
        self.tlogger = tlogger
        self.rm_plus = bool(rm_plus)
        bt = branch_topk  # limit branching at traverser nodes for speed
        if bt is not None and int(bt) <= 0:
            bt = None
        self.branch_topk = bt
        self.progressive_widening = bool(progressive_widening) and (self.branch_topk is not None)
        self.pw_alpha = max(float(pw_alpha), 0.0)
        self.pw_tail = max(int(pw_tail), 0)
        self.regret_prune = bool(regret_prune)
        self.prune_threshold = float(prune_threshold)
        self.prune_warmup = max(int(prune_warmup), 0)
        self.prune_reactivation = max(int(prune_reactivation), 1)
        self._infoset_visits: Dict[Tuple[int, bytes], int] = {}
        self._subset_cache: OrderedDict[bytes, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._subset_cache_cap = max(int(subset_cache_size), 0)
        mba = int(max_branch_actions) if (max_branch_actions and max_branch_actions > 0) else None
        if mba is not None and self.branch_topk is not None:
            self.branch_topk = min(self.branch_topk, mba)
        self.max_branch_actions = mba
        rd = int(rollout_depth) if (rollout_depth and rollout_depth > 0) else None
        self.rollout_depth = rd
        self.rollout_samples = max(int(rollout_samples), 1) if rd is not None else 0
        self._max_rollout_steps = NUM_CARDS * 2
        mode = str(reward_mode).lower()
        if mode not in ("team", "selfish"):
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")
        self.reward_mode = mode
        base_exploit_cfg = exploit_config.sanitized() if exploit_config is not None else ExploitabilityConfig()
        self.exploit_config = base_exploit_cfg.sanitized()

        # Memory controls
        self.max_infosets = int(max_infosets) if (max_infosets is not None and max_infosets > 0) else None
        self._last_seen: Dict[Tuple[int, bytes], int] = {}
        self._clock: int = 0
        self.obs_key_mode = str(obs_key_mode)
        table_dtype = np.dtype(dtype)
        self.dtype = table_dtype
        # Float16 accumulators are numerically unstable once reach weights get large
        # (sample_reach can be < 1e-6), so promote to at least float32 for storage.
        accum_dtype = np.dtype(np.float32)
        if table_dtype.itemsize >= accum_dtype.itemsize:
            accum_dtype = table_dtype
        self.accum_dtype = accum_dtype
        if rng is None:
            self.rng = Generator(PCG64(seed))
        elif hasattr(rng, "choice"):
            self.rng = rng
        else:
            try:
                entropy = np.asarray(rng, dtype=np.uint64).tobytes()
                if not entropy:
                    raise ValueError("empty entropy")
                digest = hashlib.sha1(entropy).digest()
                seed_material = int.from_bytes(digest[:8], "little")
            except Exception:
                seed_material = int(seed)
            self.rng = Generator(PCG64(seed_material & ((1 << 64) - 1)))

        # infoset -> arrays of size 40 (global card index space)
        self.cum_regret: Dict[Tuple[int, bytes], np.ndarray] = {}
        self.cum_strategy: Dict[Tuple[int, bytes], np.ndarray] = {}
        self.best_vs_random_win_rate: float = float("-inf")
        self._iter_stats: Optional[Dict[str, float]] = None

    def _begin_iteration_stats(self) -> None:
        """Reset per-iteration counters for debugging metrics."""
        self._iter_stats = {}

    def _stat_add(self, key: str, value: float) -> None:
        if self._iter_stats is None:
            return
        self._iter_stats[key] = self._iter_stats.get(key, 0.0) + float(value)

    def _stat_record_max(self, key: str, value: float) -> None:
        if self._iter_stats is None:
            return
        val = float(value)
        current = self._iter_stats.get(key)
        if current is None or val > current:
            self._iter_stats[key] = val

    def _finalize_iteration_stats(self, iteration: int, emit: bool = True) -> None:
        stats = self._iter_stats
        self._iter_stats = None
        if stats is None:
            return
        if not emit or self.tlogger is None:
            return
        writer = self.tlogger.writer
        branch_calls = stats.get("branch_calls", 0.0)
        total_legal = stats.get("branch_legal_total", 0.0)
        total_post_prune = stats.get("branch_post_prune_total", 0.0)
        total_selected = stats.get("branch_selected_total", 0.0)
        width_total = stats.get("branch_width_cap_total", 0.0)
        if branch_calls > 0:
            writer.add_scalar("CFRDebug/branch_calls", branch_calls, iteration)
            writer.add_scalar("CFRDebug/branch_avg_legal", total_legal / branch_calls, iteration)
            writer.add_scalar("CFRDebug/branch_avg_after_prune", total_post_prune / branch_calls, iteration)
            writer.add_scalar("CFRDebug/branch_avg_selected", total_selected / branch_calls, iteration)
            writer.add_scalar("CFRDebug/branch_avg_width_cap", width_total / branch_calls if branch_calls > 0 else 0.0, iteration)
        if total_legal > 0:
            pruned = max(total_legal - total_post_prune, 0.0)
            writer.add_scalar("CFRDebug/branch_prune_rate", pruned / total_legal, iteration)
        writer.add_scalar("CFRDebug/branch_pruned_actions", stats.get("branch_pruned_actions", 0.0), iteration)
        tail_samples = stats.get("branch_tail_samples", 0.0)
        if branch_calls > 0:
            writer.add_scalar("CFRDebug/branch_tail_per_call", tail_samples / branch_calls, iteration)
        if "branch_selected_max" in stats:
            writer.add_scalar("CFRDebug/branch_selected_max", stats["branch_selected_max"], iteration)
        if "branch_legal_max" in stats:
            writer.add_scalar("CFRDebug/branch_legal_max", stats["branch_legal_max"], iteration)
        hits = stats.get("subset_cache_hits", 0.0)
        misses = stats.get("subset_cache_misses", 0.0)
        cache_total = hits + misses
        if cache_total > 0:
            writer.add_scalar("CFRDebug/subset_cache_hit_rate", hits / cache_total, iteration)
        writer.add_scalar("CFRDebug/subset_cache_hits", hits, iteration)
        writer.add_scalar("CFRDebug/subset_cache_misses", misses, iteration)
        util_count = stats.get("utility_count", 0.0)
        if util_count > 0:
            util_sum = stats.get("utility_sum", 0.0)
            util_sumsq = stats.get("utility_sumsq", 0.0)
            mean = util_sum / util_count
            variance = max(util_sumsq / util_count - mean * mean, 0.0)
            writer.add_scalar("CFRDebug/traversal_util_mean", mean, iteration)
            writer.add_scalar("CFRDebug/traversal_util_var", variance, iteration)
            writer.add_scalar("CFRDebug/traversal_util_std", variance ** 0.5, iteration)
        seat0_count = stats.get("utility_count_seat0", 0.0)
        if seat0_count > 0:
            seat0_sum = stats.get("utility_sum_seat0", 0.0)
            seat0_sumsq = stats.get("utility_sumsq_seat0", 0.0)
            seat0_mean = seat0_sum / seat0_count
            seat0_var = max(seat0_sumsq / seat0_count - seat0_mean * seat0_mean, 0.0)
            writer.add_scalar("CFRDebug/traversal_util_mean_seat0", seat0_mean, iteration)
            writer.add_scalar("CFRDebug/traversal_util_std_seat0", seat0_var ** 0.5, iteration)
        mccfr_calls = stats.get("mccfr_calls", 0.0)
        if mccfr_calls > 0:
            depth_total = stats.get("mccfr_depth_total", 0.0)
            writer.add_scalar("CFRDebug/mccfr_calls", mccfr_calls, iteration)
            writer.add_scalar("CFRDebug/mccfr_avg_depth", depth_total / mccfr_calls, iteration)
        writer.add_scalar("CFRDebug/mccfr_rollout_calls", stats.get("mccfr_rollout_calls", 0.0), iteration)
        writer.add_scalar("CFRDebug/mccfr_terminal", stats.get("mccfr_terminal", 0.0), iteration)
        if "mccfr_max_depth" in stats:
            writer.add_scalar("CFRDebug/mccfr_max_depth", stats["mccfr_max_depth"], iteration)
    def _get_strategy(self, infoset_key, legal_mask: np.ndarray) -> np.ndarray:
        """Fetch/init tables for the given infoset and compute policy."""
        if infoset_key not in self.cum_regret:
            self.cum_regret[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.accum_dtype)
            self.cum_strategy[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.accum_dtype)
            self._touch_key(infoset_key)
            self._evict_if_needed(avoid_key=infoset_key)
        else:
            self._touch_key(infoset_key)
        # Use float32 math for stability
        return regret_matching(np.asarray(self.cum_regret[infoset_key], dtype=np.float32), legal_mask.astype(np.float32))
    
    def _peek_strategy(self, infoset_key: Tuple[int, bytes], legal_mask: np.ndarray) -> np.ndarray:
        """Return the current policy without creating state for unseen infosets."""
        if infoset_key in self.cum_regret:
            regrets = np.asarray(self.cum_regret[infoset_key], dtype=np.float32)
            return regret_matching(regrets, legal_mask.astype(np.float32))
        legal = legal_mask.astype(np.float32)
        total = float(legal.sum())
        if total <= 0.0:
            return legal
        return legal / total
    

    def _get_subset_dp(self, table: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._subset_cache_cap <= 0 or table.sum() <= 0:
            return _subset_sum_dp_numpy(table)
        key_bytes = np.packbits(table.astype(np.uint8), bitorder="little").tobytes()
        cached = self._subset_cache.get(key_bytes)
        if cached is not None:
            self._subset_cache.move_to_end(key_bytes)
            self._stat_add("subset_cache_hits", 1.0)
            return cached
        self._stat_add("subset_cache_misses", 1.0)
        dp_has, dp_mask = _subset_sum_dp_numpy(table)
        self._subset_cache[key_bytes] = (dp_has, dp_mask)
        if len(self._subset_cache) > self._subset_cache_cap:
            self._subset_cache.popitem(last=False)
        return dp_has, dp_mask

    def _subset_sum_mask_cached(self, table: np.ndarray, target_rank: int) -> np.ndarray:
        if target_rank <= 0 or target_rank > NUM_RANKS:
            return np.zeros((NUM_CARDS,), dtype=bool)
        dp_has, dp_mask = self._get_subset_dp(table)
        if target_rank >= dp_has.size or not bool(dp_has[target_rank]):
            return np.zeros((NUM_CARDS,), dtype=bool)
        return np.asarray(dp_mask[target_rank], dtype=bool).copy()

    def _subset_sum_masks_cached(self, table: np.ndarray, actions: np.ndarray) -> Dict[int, np.ndarray]:
        if actions.size == 0:
            return {}
        dp_has, dp_mask = self._get_subset_dp(table)
        cache: Dict[int, np.ndarray] = {}
        for a in actions:
            idx = int(a)
            rank = int(CARD_RANKS[idx])
            if rank <= 0 or rank > NUM_RANKS:
                continue
            if bool(dp_has[rank]):
                cache[idx] = dp_mask[rank]
        return cache

    def _increment_visit(self, infoset_key: Tuple[int, bytes]) -> int:
        count = self._infoset_visits.get(infoset_key, 0) + 1
        self._infoset_visits[infoset_key] = count
        return count

    def _apply_regret_pruning(self, infoset_key: Tuple[int, bytes], actions: np.ndarray, visit_count: int) -> np.ndarray:
        if not self.regret_prune or actions.size == 0 or visit_count <= self.prune_warmup:
            return actions
        if visit_count % self.prune_reactivation == 0:
            return actions
        regrets = np.asarray(self.cum_regret[infoset_key], dtype=np.float32)[actions]
        keep_mask = regrets > self.prune_threshold
        if keep_mask.any():
            return actions[keep_mask]
        return actions

    def _sample_weighted_subset(self, choices: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
        if choices.size == 0 or k <= 0:
            return np.empty((0,), dtype=choices.dtype)
        probs = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            probs = np.full(choices.size, 1.0 / float(choices.size), dtype=np.float64)
        else:
            probs /= total
            adjust = 1.0 - float(probs.sum())
            if abs(adjust) > 1e-15:
                probs[-1] += adjust
            probs = np.clip(probs, 0.0, 1.0)
            norm = float(probs.sum())
            if norm <= 0.0 or not np.isfinite(norm):
                probs = np.full(choices.size, 1.0 / float(choices.size), dtype=np.float64)
            else:
                probs /= norm
        idx = self.rng.choice(choices.size, size=min(k, choices.size), replace=False, p=probs)
        return choices[np.atleast_1d(idx)]
    
    def _select_branch_actions(self, infoset_key: Tuple[int, bytes], legal_actions: np.ndarray, sigma: np.ndarray, visit_count: int) -> np.ndarray:
        actions = legal_actions
        legal_count = int(actions.size)
        self._stat_add("branch_calls", 1.0)
        self._stat_add("branch_legal_total", float(legal_count))
        self._stat_record_max("branch_legal_max", float(legal_count))
        if legal_count == 0:
            return actions
        before_prune = actions.size
        pruned = self._apply_regret_pruning(infoset_key, actions, visit_count)
        if pruned.size > 0:
            removed = max(before_prune - pruned.size, 0)
            if removed > 0:
                self._stat_add("branch_pruned_actions", float(removed))
            actions = pruned
        self._stat_add("branch_post_prune_total", float(actions.size))
        effective_limit = self.branch_topk
        if effective_limit is None and self.max_branch_actions is not None:
            effective_limit = self.max_branch_actions
        width_cap = actions.size
        if effective_limit is None or actions.size <= effective_limit:
            selected = actions
        else:
            probs_masked = sigma[actions]
            width_cap = min(actions.size, effective_limit)
            if self.progressive_widening:
                extra = int(np.power(max(visit_count - 1, 0), self.pw_alpha))
                width_cap = min(actions.size, effective_limit + extra)
            if self.max_branch_actions is not None:
                width_cap = min(width_cap, self.max_branch_actions)
            top_idx = np.argpartition(probs_masked, -width_cap)[-width_cap:]
            selected = actions[top_idx]
            if self.progressive_widening and width_cap < actions.size:
                tail_mask = np.ones(actions.size, dtype=bool)
                tail_mask[top_idx] = False
                tail_actions = actions[tail_mask]
                if tail_actions.size > 0:
                    samples = min(tail_actions.size, max(self.pw_tail, 1) + int(np.log1p(visit_count)))
                    sampled = self._sample_weighted_subset(tail_actions, probs_masked[tail_mask], samples)
                    if sampled.size > 0:
                        self._stat_add("branch_tail_samples", float(sampled.size))
                        selected = np.concatenate([selected, sampled.astype(selected.dtype)])
        self._stat_add("branch_width_cap_total", float(width_cap))
        selected = np.unique(selected.astype(np.int32))
        if width_cap < selected.size:
            downsampled = self._sample_weighted_subset(selected, sigma[selected], width_cap)
            if downsampled.size > 0:
                selected = downsampled.astype(np.int32)
            else:
                selected = selected[:width_cap]

        if self.max_branch_actions is not None and selected.size > self.max_branch_actions:
            downsampled = self._sample_weighted_subset(selected, sigma[selected], self.max_branch_actions)
            if downsampled.size > 0:
                selected = downsampled.astype(np.int32)
            else:
                selected = selected[:self.max_branch_actions]
        self._stat_add("branch_selected_total", float(selected.size))
        self._stat_record_max("branch_selected_max", float(selected.size))
        return np.sort(selected.astype(np.int32))

    def _apply_action_inplace(self, st: NState, action_idx: int, subset_mask: Optional[np.ndarray] = None) -> ActionDiff:
        diff, _ = _apply_action_core(
            st,
            int(action_idx),
            subset_mask=subset_mask,
            mask_resolver=self._subset_sum_mask_cached,
            record_diff=True,
        )
        assert diff is not None  # for type checkers; record_diff=True guarantees diff
        return diff

    def _undo_action(self, st: NState, diff: ActionDiff) -> None:
        seat = diff.seat
        st.cur_player = np.int32(diff.cur_player_prev)
        st.last_capture_player = np.int32(diff.last_capture_prev)
        st.hands[seat, diff.action_idx] = diff.hand_prev
        st.history[seat, diff.action_idx] = diff.history_prev
        if diff.table_indices.size > 0:
            st.table[diff.table_indices] = diff.table_prev
        if diff.captures_indices.size > 0:
            st.captures[seat, diff.captures_indices] -= 1
        if diff.scopa_delta:
            st.scopas[seat] -= diff.scopa_delta

    def _estimate_rollout_value(self, st: NState, target_seat: int, rng: Generator) -> float:
        samples = max(self.rollout_samples, 1) if self.rollout_samples else 1
        policy_cache: Dict[Tuple[int, bytes], np.ndarray] = {}
        total = 0.0
        for _ in range(samples):
            trace: list[ActionDiff] = []
            steps = 0
            while not np_is_terminal(st) and steps < self._max_rollout_steps:
                cur = int(st.cur_player)
                obs = np_build_obs(st, cur)
                legal_mask = (obs[0] > 0).astype(np.int32)
                legal_actions = np.nonzero(legal_mask)[0]
                if legal_actions.size == 0:
                    break
                infoset_key = (cur, self._obs_key(obs))
                sigma = policy_cache.get(infoset_key)
                if sigma is None:
                    sigma = self._peek_strategy(infoset_key, legal_mask).copy()
                    policy_cache[infoset_key] = sigma
                action = int(self._safe_sample(sigma, legal_mask, rng))
                diff = self._apply_action_inplace(st, action)
                trace.append(diff)
                steps += 1
            total += self._evaluate_utility(st, target_seat)
            while trace:
                self._undo_action(st, trace.pop())
        return total / float(samples)

    def _safe_sample(self, p_arr: np.ndarray, legal_mask: np.ndarray, rng: Optional[Generator] = None,
                     return_prob: bool = False) -> Union[int, Tuple[int, float]]:
        """Sample an action robustly over the legal set.

        - Works on the restricted legal index set to avoid tiny normalization errors.
        - Falls back to uniform legal if probabilities are degenerate.
        - Optionally returns the probability used for sampling the chosen action.
        """
        if rng is None:
            rng = self.rng
        lm = (legal_mask > 0).astype(np.int32)
        legal_idx = np.nonzero(lm)[0]
        if legal_idx.size == 0:
            return (0, 1.0) if return_prob else 0

        p = np.asarray(p_arr, dtype=np.float64)[legal_idx]
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            probs = np.full(legal_idx.size, 1.0 / float(legal_idx.size), dtype=np.float64)
        else:
            p /= s
            adjust = 1.0 - float(p.sum())
            if abs(adjust) > 1e-15:
                p[-1] += adjust
            p = np.clip(p, 0.0, 1.0)
            s2 = float(p.sum())
            if s2 <= 0.0 or not np.isfinite(s2):
                probs = np.full(legal_idx.size, 1.0 / float(legal_idx.size), dtype=np.float64)
            else:
                p /= s2
                probs = p

        choice_idx = int(rng.choice(legal_idx.size, p=probs))
        choice = int(legal_idx[choice_idx])
        if return_prob:
            return choice, float(probs[choice_idx])
        return choice

    def _evaluate_utility(self, st: NState, seat: int) -> float:
        team_points, player_points = np_round_scores(st)
        team_idx = TEAM_INDEX[seat]
        other_team = 1 - team_idx
        team_advantage = float(team_points[team_idx] - team_points[other_team])
        if self.reward_mode == "team":
            sign = 0.0
            if team_advantage > 0.0:
                sign = 1.0
            elif team_advantage < 0.0:
                sign = -1.0
            return sign

        team_total = float(team_points[team_idx])
        if team_total > 0.0:
            weight = float(player_points[seat]) / team_total
        else:
            weight = 0.5
        return team_advantage * weight

    def _obs_key(self, obs: np.ndarray) -> bytes:
        """Bit-pack observation planes to bytes (exactly like SavedPolicy).
        Full: 4x40 -> 20 bytes; compact: 3x40 -> 15B; hand_table: 2x40 -> 10B.
        """
        mode = self.obs_key_mode
        if mode == "hand_table":
            obs_slice = obs[:2]
        elif mode == "compact":
            obs_slice = obs[:3]
        else:
            obs_slice = obs
        planes = (np.asarray(obs_slice, dtype=np.float32) > 0.5).astype(np.uint8)
        packed = []
        for plane in planes:
            b = np.packbits(plane, bitorder="little")
            packed.append(b[:5].tobytes())  # 40 bits -> 5 bytes
        return b"".join(packed)

    def _infoset(self, st: NState, seat: int) -> Tuple[Tuple[int, bytes], np.ndarray]:
        """Return (infoset_key, legal_mask) for seat in state."""
        obs = np_build_obs(st, seat)
        key = (seat, self._obs_key(obs))
        legal = (obs[0] > 0).astype(np.int32)
        return key, legal

    def _touch_key(self, key: Tuple[int, bytes]) -> None:
        self._clock += 1
        self._last_seen[key] = self._clock

    def _evict_if_needed(self, avoid_key: Optional[Tuple[int, bytes]] = None) -> None:
        if self.max_infosets is None:
            return
        while len(self.cum_regret) > self.max_infosets:
            # Evict least recently touched key (simple LRU)
            to_evict = None
            min_seen = None
            for k, seen in self._last_seen.items():
                if avoid_key is not None and k == avoid_key:
                    continue
                if min_seen is None or seen < min_seen:
                    min_seen = seen
                    to_evict = k
            if to_evict is None:
                break
            self.cum_regret.pop(to_evict, None)
            self.cum_strategy.pop(to_evict, None)
            self._last_seen.pop(to_evict, None)

    def _mccfr(self, st: NState, target_seat: int,
               reach_my: float = 1.0,
               reach_others: float = 1.0,
               sample_reach: float = 1.0,
               depth: int = 0) -> float:
        """Recursive MCCFR traversal optimizing `target_seat` with importance weighting."""
        self._stat_add("mccfr_calls", 1.0)
        self._stat_add("mccfr_depth_total", float(depth))
        self._stat_record_max("mccfr_max_depth", float(depth))
        if np_is_terminal(st):
            self._stat_add("mccfr_terminal", 1.0)
            payoff = self._evaluate_utility(st, target_seat)
            if sample_reach <= 0.0:
                return payoff
            return float(reach_others / sample_reach) * payoff

        if self.rollout_depth is not None and depth >= self.rollout_depth:
            self._stat_add("mccfr_rollout_calls", 1.0)
            return self._estimate_rollout_value(st, target_seat, self.rng)

        cur_seat = int(st.cur_player)
        infoset_key, legal_mask = self._infoset(st, cur_seat)
        sigma = self._get_strategy(infoset_key, legal_mask)
        visit_count = self._increment_visit(infoset_key)

        legal_actions = np.nonzero(legal_mask)[0]
        subset_masks = self._subset_sum_masks_cached(st.table, legal_actions)
        if cur_seat == target_seat:
            action_values = np.zeros(NUM_CARDS, dtype=np.float32)
            util_acc = 0.0
            prob_acc = 0.0
            branch_actions = self._select_branch_actions(infoset_key, legal_actions, sigma, visit_count)
            if branch_actions.size == 0:
                branch_actions = legal_actions
            visited_mask = np.zeros(NUM_CARDS, dtype=bool)
            for a in branch_actions:
                a_int = int(a)
                prob = float(max(sigma[a_int], 0.0))
                diff = self._apply_action_inplace(st, a_int, subset_mask=subset_masks.get(a_int))
                v = self._mccfr(
                    st,
                    target_seat,
                    reach_my=reach_my * prob,
                    reach_others=reach_others,
                    sample_reach=sample_reach,
                    depth=depth + 1,
                )
                self._undo_action(st, diff)
                action_values[a_int] = v
                util_acc += prob * float(v)
                prob_acc += prob
                visited_mask[a_int] = True

            util = util_acc
            if legal_actions.size > 0:
                unseen_mask = visited_mask[legal_actions] == False
                if unseen_mask.any():
                    unseen_actions = legal_actions[unseen_mask]
                    if prob_acc > 0.0:
                        baseline = util_acc / max(prob_acc, 1e-12)
                    else:
                        baseline = 0.0
                    tail_probs = np.clip(sigma[unseen_actions], 0.0, None)
                    action_values[unseen_actions] = baseline
                    util += float(tail_probs.sum()) * baseline

            weight = reach_others / max(sample_reach, 1e-12)
            regrets = np.zeros(NUM_CARDS, dtype=np.float32)
            if legal_actions.size > 0:
                regrets[legal_actions] = (action_values[legal_actions] - util) * float(weight)
            self.cum_regret[infoset_key] += regrets.astype(self.accum_dtype)
            if self.rm_plus:
                self.cum_regret[infoset_key] = np.maximum(
                    np.asarray(self.cum_regret[infoset_key], dtype=np.float32),
                    0.0
                ).astype(self.accum_dtype)

            # Average strategy accumulates opponents' reach probability (pi_-i)
            strategy_weight = reach_others / max(sample_reach, 1e-12)
            self.cum_strategy[infoset_key] += (sigma * strategy_weight).astype(self.accum_dtype)
            return util

        action, sample_prob = self._safe_sample(sigma, legal_mask, self.rng, return_prob=True)
        sample_prob = float(sample_prob)
        if sample_prob <= 0.0:
            sample_prob = 1.0 / max(legal_actions.size, 1)

        diff = self._apply_action_inplace(st, int(action), subset_mask=subset_masks.get(int(action)))
        actual_prob = float(max(sigma[int(action)], 0.0))
        value = self._mccfr(
            st,
            target_seat,
            reach_my=reach_my,
            reach_others=reach_others * actual_prob,
            sample_reach=sample_reach * sample_prob,
            depth=depth + 1,
        )
        self._undo_action(st, diff)
        return value

    def dump_top_infosets(self, iteration: int, top_k: int, path: str, mode: str = "max_regret") -> None:
        """Persist the most problematic infosets for offline inspection."""
        if top_k is None or top_k <= 0:
            return
        entries = []
        for (seat, key), regrets in self.cum_regret.items():
            arr = np.asarray(regrets, dtype=np.float64)
            abs_arr = np.abs(arr)
            max_reg = float(abs_arr.max()) if abs_arr.size > 0 else 0.0
            mean_reg = float(abs_arr.mean()) if abs_arr.size > 0 else 0.0
            visits = int(self._infoset_visits.get((seat, key), 0))
            strat_sum = self.cum_strategy.get((seat, key))
            entropy = None
            top_actions = []
            if strat_sum is not None:
                strat = np.asarray(strat_sum, dtype=np.float64)
                total = float(strat.sum())
                probs = strat / total if total > 0 else np.zeros_like(strat)
                entropy = float(-np.sum(np.where(probs > 1e-12, probs * np.log2(np.clip(probs, 1e-12, 1.0)), 0.0)))
                top_idx = np.argsort(probs)[-5:][::-1]
                top_actions = [
                    {
                        "action": int(idx),
                        "prob": float(probs[idx]),
                        "regret": float(arr[idx]),
                    }
                    for idx in top_idx
                    if probs[idx] > 1e-6
                ]
            entries.append(
                {
                    "seat": int(seat),
                    "infoset_key": key.hex(),
                    "max_regret": max_reg,
                    "mean_abs_regret": mean_reg,
                    "visits": visits,
                    "strategy_entropy": entropy,
                    "top_actions": top_actions,
                }
            )
        if not entries:
            return
        sort_key = "mean_abs_regret" if mode == "mean_abs_regret" else "max_regret"
        entries.sort(key=lambda item: item[sort_key], reverse=True)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "iteration": int(iteration),
            "mode": sort_key,
            "top_k": int(top_k),
            "entries": entries[:top_k],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _metrics_snapshot(self) -> Dict[str, float]:
        n_infosets = float(len(self.cum_regret))
        if n_infosets == 0:
            return {"num_infosets": 0.0, "avg_regret_abs": 0.0}
        total_abs = 0.0
        total_sq = 0.0
        count = 0
        max_abs = 0.0
        for arr in self.cum_regret.values():
            a = np.asarray(arr)
            abs_arr = np.abs(a)
            total_abs += abs_arr.sum()
            total_sq += np.square(a).sum()
            count += a.size
            local_max = float(abs_arr.max()) if abs_arr.size > 0 else 0.0
            if local_max > max_abs:
                max_abs = local_max
        avg_abs = float(total_abs / max(count, 1))
        rms = float(np.sqrt(total_sq / max(count, 1))) if count > 0 else 0.0
        return {"num_infosets": n_infosets, "avg_regret_abs": avg_abs, "regret_rms": rms, "max_regret_abs": max_abs}

    def _log_metrics(self, it: int, util0: Optional[float] = None):
        if self.tlogger is None:
            return
        snap = self._metrics_snapshot()
        self.tlogger.writer.add_scalar("CFR/num_infosets", snap["num_infosets"], it)
        self.tlogger.writer.add_scalar("CFR/avg_regret_abs", snap["avg_regret_abs"], it)
        self.tlogger.writer.add_scalar("CFR/regret_rms", snap["regret_rms"], it)
        self.tlogger.writer.add_scalar("CFR/regret_max_abs", snap["max_regret_abs"], it)
        if util0 is not None:
            self.tlogger.writer.add_scalar("CFR/util_seat0", float(util0), it)

    # --------------------------
    # Evaluation utilities
    # --------------------------
    def _category_breakdown(self, st: NState) -> Dict[str, float]:
        """Compute Scopa category wins, points, and collaboration metrics."""
        captures = np_final_captures(st)
        team0_caps = (captures.T @ TEAM0_MASK).astype(np.int32)
        team1_caps = (captures.T @ TEAM1_MASK).astype(np.int32)

        scopa0 = int(st.scopas @ TEAM0_MASK)
        scopa1 = int(st.scopas @ TEAM1_MASK)

        cards_per_player = captures.sum(axis=1).astype(np.int32)
        cards_team0 = int(cards_per_player[0] + cards_per_player[2])
        cards_team1 = int(cards_per_player[1] + cards_per_player[3])
        mc0 = 1 if cards_team0 > cards_team1 else 0
        mc1 = 1 if cards_team1 > cards_team0 else 0

        is_bello = (CARD_SUITS == 3).astype(np.int32)
        coins_per_player = (captures * is_bello).sum(axis=1).astype(np.int32)
        coins_team0 = int(coins_per_player[0] + coins_per_player[2])
        coins_team1 = int(coins_per_player[1] + coins_per_player[3])
        mb0 = 1 if coins_team0 > coins_team1 else 0
        mb1 = 1 if coins_team1 > coins_team0 else 0

        sette_bello_idx = 3 * NUM_RANKS + (7 - 1)
        sb0 = 1 if team0_caps[sette_bello_idx] > 0 else 0
        sb1 = 1 if team1_caps[sette_bello_idx] > 0 else 0

        pri = PRIMIERA_PRIORITY[1:]
        primiera_per_player = np.zeros(4, dtype=np.float32)
        for seat in range(4):
            cards = captures[seat].reshape(NUM_SUITS, NUM_RANKS)
            primiera_per_player[seat] = float((cards * pri).max(axis=1).sum())
        prim0_val = float(primiera_per_player[0] + primiera_per_player[2])
        prim1_val = float(primiera_per_player[1] + primiera_per_player[3])
        pr0 = 1 if prim0_val > prim1_val else 0
        pr1 = 1 if prim1_val > prim0_val else 0

        points0 = scopa0 + mc0 + mb0 + sb0 + pr0
        points1 = scopa1 + mc1 + mb1 + sb1 + pr1

        team_points, player_points = np_round_scores(st)
        inter_delta = float(team_points[0] - team_points[1])
        intra_delta = 0.5 * (abs(player_points[0] - player_points[2]) + abs(player_points[1] - player_points[3]))

        return {
            "points0": float(points0),
            "points1": float(points1),
            "scopas0": float(scopa0),
            "scopas1": float(scopa1),
            "most_cards0": float(mc0),
            "most_cards1": float(mc1),
            "most_coins0": float(mb0),
            "most_coins1": float(mb1),
            "sette_bello0": float(sb0),
            "sette_bello1": float(sb1),
            "primiera0": float(pr0),
            "primiera1": float(pr1),
            "team_points0": float(team_points[0]),
            "team_points1": float(team_points[1]),
            "player_points0": float(player_points[0]),
            "player_points1": float(player_points[1]),
            "player_points2": float(player_points[2]),
            "player_points3": float(player_points[3]),
            "inter_team_delta": inter_delta,
            "intra_team_delta": float(intra_delta),
        }

    def _select_action_eval(self, st: NState, seat: int, mode: str,
                             policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]] = None,
                             rng: Optional[Generator] = None) -> int:
        """Select an action for evaluation.

        mode: 'current' -> regret-matching policy; 'avg' -> average policy; 'random' -> uniform legal.
        """
        if rng is None:
            rng = self.rng
        obs = np_build_obs(st, seat)
        legal = (obs[0] > 0).astype(np.float32)
        legal_actions = np.nonzero(legal)[0]

        def _safe_sample(p_arr: np.ndarray, legal_mask: np.ndarray) -> int:
            lm = legal_mask.astype(np.float64)
            if lm.sum() <= 0:
                return 0
            p = np.asarray(p_arr, dtype=np.float64) * lm
            s = float(p.sum())
            if s <= 0.0 or not np.isfinite(s):
                legal_idx = np.nonzero(lm)[0]
                return int(rng.choice(legal_idx)) if legal_idx.size > 0 else 0
            p /= s
            legal_idx = np.nonzero(lm)[0]
            if legal_idx.size > 0:
                adjust = 1.0 - float(p.sum())
                if abs(adjust) > 1e-15:
                    p[legal_idx[-1]] += adjust
                p = np.clip(p, 0.0, 1.0)
                s2 = float(p.sum())
                if s2 > 0.0:
                    p /= s2
                else:
                    return int(rng.choice(legal_idx))
            return int(rng.choice(NUM_CARDS, p=p))

        if mode == "random":
            return int(rng.choice(legal_actions)) if legal_actions.size > 0 else 0

        if mode == "avg":
            key = (seat, self._obs_key(obs))
            if policy_map is not None and key in policy_map:
                probs = policy_map[key]
                return self._safe_sample(probs, legal, rng)
            strat_sum = self.cum_strategy.get(key)
            if strat_sum is not None:
                strat = np.asarray(strat_sum, dtype=np.float32)
                total = float(strat.sum())
                if total > 0.0 and np.isfinite(total):
                    probs = strat / total
                    return self._safe_sample(probs, legal, rng)
            # Fallback to current if unseen

        # Current regret-matching policy
        infoset_key = (seat, self._obs_key(obs))
        sigma = self._get_strategy(infoset_key, legal.astype(np.int32))
        return self._safe_sample(sigma, legal, rng)

    def evaluate(self, episodes: int = 50, seed: int = 12345,
                 selfplay: bool = True, use_avg_policy: bool = True):
        """Run evaluation episodes and return aggregated metrics.

        If selfplay=True, all seats use the same policy. Otherwise, seats 0/2 use policy and 1/3 use random.
        """
        rng = Generator(PCG64(seed))
        # Compute average policy on-demand during evaluation to avoid duplicating memory
        policy_map = None

        wins0 = 0
        wins1 = 0
        draws = 0
        # Category sums (auto-extend to new metrics)
        sums = defaultdict(float)

        for ep in trange(episodes, desc="eval"):
            st = np_init_state(Generator(PCG64(seed + 1000 + ep)))
            while not np_is_terminal(st):
                seat = int(st.cur_player)
                # Determine which mode each seat uses
                if selfplay:
                    mode = "avg" if use_avg_policy else "current"
                else:
                    # Team 0 uses policy, team 1 uses random
                    mode = ("avg" if use_avg_policy else "current") if (seat % 2 == 0) else "random"
                a = self._select_action_eval(st, seat, mode, policy_map, rng)
                st, _ = np_step(st, a)

            cat = self._category_breakdown(st)
            for k, v in cat.items():
                sums[k] += float(v)
            if cat["points0"] > cat["points1"]:
                wins0 += 1
            elif cat["points1"] > cat["points0"]:
                wins1 += 1
            else:
                draws += 1

        inv_eps = 1.0 / max(episodes, 1)
        out = {
            "win_rate_team0": wins0 * inv_eps,
            "win_rate_team1": wins1 * inv_eps,
            "draw_rate": draws * inv_eps,
        }
        # Average each category
        for k, v in sums.items():
            out[f"avg_{k}"] = v * inv_eps
        return out

    
    # --------------------------
    # Exploitability utilities
    # --------------------------
    def _policy_action_probs(self, seat: int, obs: np.ndarray,
                             policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
                             use_avg_policy: bool) -> np.ndarray:
        """Return a probability distribution over legal actions for an infoset."""
        legal = (obs[0] > 0).astype(np.float32)
        key = (seat, self._obs_key(obs))

        probs = None
        if policy_map is not None:
            probs = policy_map.get(key)

        if probs is None and use_avg_policy:
            strat_sum = self.cum_strategy.get(key)
            if strat_sum is not None:
                strat = np.asarray(strat_sum, dtype=np.float32)
                total = float(strat.sum())
                if total > 0.0 and np.isfinite(total):
                    probs = strat / total

        if probs is None:
            probs = self._peek_strategy(key, legal.astype(np.int32))

        probs = np.asarray(probs, dtype=np.float32) * legal
        total = float(probs.sum())
        if total > 0.0 and np.isfinite(total):
            probs /= total
        else:
            count = max(int(legal.sum()), 1)
            probs = legal / float(count)
        return probs

    def _policy_value_recursive(
        self,
        st: NState,
        team_override: Optional[int],
        policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
        use_avg_policy: bool,
    ) -> float:
        """Evaluate a policy assuming optional best-response override for one team."""

        if np_is_terminal(st):
            return float(self._evaluate_utility(st, seat=0))

        seat = int(st.cur_player)
        team = int(TEAM_INDEX[seat])
        obs = np_build_obs(st, seat)
        legal_mask = (obs[0] > 0).astype(np.int32)
        legal_actions = np.nonzero(legal_mask)[0]
        if legal_actions.size == 0:
            return float(self._evaluate_utility(st, seat=0))

        subset_masks = self._subset_sum_masks_cached(st.table, legal_actions)

        if team_override is not None and team == int(team_override):
            best_val = float("-inf") if team_override == 0 else float("inf")
            for action in legal_actions:
                diff = self._apply_action_inplace(st, int(action), subset_mask=subset_masks.get(int(action)))
                val = self._policy_value_recursive(st, team_override, policy_map, use_avg_policy)
                self._undo_action(st, diff)
                if team_override == 0:
                    if val > best_val:
                        best_val = val
                else:
                    if val < best_val:
                        best_val = val
            if not np.isfinite(best_val):
                best_val = 0.0
            return float(best_val)

        probs = self._policy_action_probs(seat, obs, policy_map, use_avg_policy)
        total = 0.0
        for action in legal_actions:
            prob = float(probs[action])
            if prob <= 0.0:
                continue
            diff = self._apply_action_inplace(st, int(action), subset_mask=subset_masks.get(int(action)))
            val = self._policy_value_recursive(st, team_override, policy_map, use_avg_policy)
            self._undo_action(st, diff)
            total += prob * float(val)
        return float(total)

    def _policy_value_triplet(
        self,
        st: NState,
        policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
        use_avg_policy: bool,
    ) -> Tuple[float, float, float]:
        """Return (policy value, team0 best-response, team1 best-response)."""

        base_state = np_clone_state(st)
        value = self._policy_value_recursive(base_state, None, policy_map, use_avg_policy)

        br0_state = np_clone_state(st)
        br0 = self._policy_value_recursive(br0_state, 0, policy_map, use_avg_policy)

        br1_state = np_clone_state(st)
        br1 = self._policy_value_recursive(br1_state, 1, policy_map, use_avg_policy)

        return float(value), float(br0), float(br1)

    def _simulate_policy_episode(self, st: NState,
                                  policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
                                  use_avg_policy: bool,
                                  rng: Generator) -> float:
        trace: list[ActionDiff] = []
        while not np_is_terminal(st):
            seat = int(st.cur_player)
            obs = np_build_obs(st, seat)
            legal_mask = (obs[0] > 0).astype(np.int32)
            legal_actions = np.nonzero(legal_mask)[0]
            if legal_actions.size == 0:
                break
            subset_masks = self._subset_sum_masks_cached(st.table, legal_actions)
            probs = self._policy_action_probs(seat, obs, policy_map, use_avg_policy)
            action = int(self._safe_sample(probs, legal_mask, rng))
            diff = self._apply_action_inplace(st, action, subset_mask=subset_masks.get(action))
            trace.append(diff)
        payoff = float(self._evaluate_utility(st, seat=0))
        while trace:
            self._undo_action(st, trace.pop())
        return payoff

    def _estimate_policy_value_monte_carlo(self, st: NState,
                                           policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
                                           use_avg_policy: bool,
                                           rng: Generator,
                                           rollouts: int) -> float:
        total = 0.0
        runs = max(int(rollouts), 1)
        for _ in range(runs):
            total += self._simulate_policy_episode(st, policy_map, use_avg_policy, rng)
        return total / float(runs)

    def _sampled_best_response(self, st: NState,
                               policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
                               use_avg_policy: bool,
                               team: int,
                               rng: Generator,
                               action_samples: int,
                               depth: int = 0) -> float:
        if np_is_terminal(st):
            return float(self._evaluate_utility(st, seat=0))

        if self.rollout_depth is not None and depth >= self.rollout_depth:
            return float(self._estimate_rollout_value(st, target_seat=0, rng=rng))

        obs = np_build_obs(st, int(st.cur_player))
        legal_mask = (obs[0] > 0).astype(np.int32)
        legal_actions = np.nonzero(legal_mask)[0]
        if legal_actions.size == 0:
            return float(self._evaluate_utility(st, seat=0))

        subset_masks = self._subset_sum_masks_cached(st.table, legal_actions)
        seat = int(st.cur_player)
        team_idx = int(TEAM_INDEX[seat])

        if team_idx == team:
            samples = max(int(action_samples), 1)
            best_val = float('-inf') if team == 0 else float('inf')
            for action in legal_actions:
                action_total = 0.0
                for _ in range(samples):
                    diff = self._apply_action_inplace(st, int(action), subset_mask=subset_masks.get(int(action)))
                    val = self._sampled_best_response(st, policy_map, use_avg_policy, team, rng, action_samples, depth + 1)
                    action_total += float(val)
                    self._undo_action(st, diff)
                avg_val = action_total / float(samples)
                if team == 0:
                    if avg_val > best_val:
                        best_val = avg_val
                else:
                    if avg_val < best_val:
                        best_val = avg_val
            if not np.isfinite(best_val):
                best_val = 0.0
            return float(best_val)

        probs = self._policy_action_probs(seat, obs, policy_map, use_avg_policy)
        choice = int(self._safe_sample(probs, legal_mask, rng))
        diff = self._apply_action_inplace(st, choice, subset_mask=subset_masks.get(choice))
        val = self._sampled_best_response(st, policy_map, use_avg_policy, team, rng, action_samples, depth + 1)
        self._undo_action(st, diff)
        return float(val)

    def _estimate_best_response_value(self, st: NState,
                                      policy_map: Optional[Dict[Tuple[int, bytes], np.ndarray]],
                                      use_avg_policy: bool,
                                      team: int,
                                      rng: Generator,
                                      rollouts: int,
                                      action_samples: int) -> float:
        runs = max(int(rollouts), 1)
        total = 0.0
        for _ in range(runs):
            total += self._sampled_best_response(st, policy_map, use_avg_policy, team, rng, action_samples)
        return total / float(runs)

    def compute_exploitability(self, episodes: int = 8, seed: int = 12345,
                               use_avg_policy: bool = True,
                               config: Optional[ExploitabilityConfig] = None) -> Dict[str, float]:
        """Estimate exploitability via Monte Carlo best responses."""
        if episodes <= 0:
            raise ValueError("episodes must be positive for exploitability computation")

        cfg = (config.sanitized() if config is not None else self.exploit_config).sanitized()
        policy_map = self.get_average_policy() if use_avg_policy else None
        total_value = 0.0
        total_br0 = 0.0
        total_br1 = 0.0
        rng = Generator(PCG64(seed + 424242))

        for ep in trange(episodes, desc='Exploitability'):
            deal_rng = Generator(PCG64(seed + 1000 + ep))
            base_state = np_init_state(deal_rng)

            policy_state = np_clone_state(base_state)
            total_value += self._estimate_policy_value_monte_carlo(
                policy_state,
                policy_map,
                use_avg_policy,
                rng,
                cfg.policy_rollouts,
            )

            br0_state = np_clone_state(base_state)
            total_br0 += self._estimate_best_response_value(
                br0_state,
                policy_map,
                use_avg_policy,
                team=0,
                rng=rng,
                rollouts=cfg.best_response_rollouts,
                action_samples=cfg.opponent_samples,
            )

            br1_state = np_clone_state(base_state)
            total_br1 += self._estimate_best_response_value(
                br1_state,
                policy_map,
                use_avg_policy,
                team=1,
                rng=rng,
                rollouts=cfg.best_response_rollouts,
                action_samples=cfg.opponent_samples,
            )

        inv_eps = 1.0 / float(episodes)
        avg_value = total_value * inv_eps
        avg_br0 = total_br0 * inv_eps
        avg_br1 = total_br1 * inv_eps
        gain0 = max(0.0, avg_br0 - avg_value)
        gain1 = max(0.0, avg_value - avg_br1)
        exploit = 0.5 * (gain0 + gain1)
        return {
            'exploitability': float(exploit),
            'policy_value': float(avg_value),
            'best_response_team0': float(avg_br0),
            'best_response_team1': float(avg_br1),
            'gain_team0': float(gain0),
            'gain_team1': float(gain1),
        }

    def _perform_evaluations(
        self,
        iteration: int,
        schedule: Optional[EvaluationSchedule],
        base_seed: int,
        best_save_path: Optional[str],
        best_save_kind: Optional[str],
        verbose: bool,
    ) -> None:
        if schedule is None or not schedule.should_run(iteration):
            return
        if self.tlogger is None and best_save_path is None:
            return

        episodes = schedule.resolved_episodes()
        use_avg = schedule.use_avg_policy

        if self.tlogger is not None:
            selfplay_metrics = self.evaluate(
                episodes=episodes,
                seed=base_seed + 7777 + iteration,
                selfplay=True,
                use_avg_policy=use_avg,
            )
            for k, v in selfplay_metrics.items():
                self.tlogger.writer.add_scalar(f"EvalSelf/{k}", float(v), iteration)
        else:
            selfplay_metrics = {}

        vsrnd_metrics = self.evaluate(
            episodes=episodes,
            seed=base_seed + 8888 + iteration,
            selfplay=False,
            use_avg_policy=use_avg,
        )

        if self.tlogger is not None:
            for k, v in vsrnd_metrics.items():
                self.tlogger.writer.add_scalar(f"EvalVsRandom/{k}", float(v), iteration)

        if best_save_path is not None and vsrnd_metrics:
            win0 = float(vsrnd_metrics.get("win_rate_team0", 0.0))
            if win0 > self.best_vs_random_win_rate:
                self.best_vs_random_win_rate = win0
                kind = best_save_kind if best_save_kind else "avg"
                try:
                    self.save(best_save_path, kind=kind)
                    if verbose:
                        print(
                            f"[CFR] Saved new best checkpoint to {best_save_path} "
                            f"(win_rate_team0={win0:.3f})"
                        )
                except Exception as exc:
                    if verbose:
                        print(f"[CFR] WARNING: failed to save best checkpoint to {best_save_path}: {exc}")

    def _maybe_compute_exploitability(
        self,
        iteration: int,
        schedule: Optional[ExploitabilitySchedule],
        default_episodes: int,
        seed: int,
        verbose: bool,
    ) -> Optional[Dict[str, float]]:
        if schedule is None or not schedule.should_run(iteration):
            return None

        exploit_eps = schedule.resolved_episodes(default_episodes)
        try:
            stats = self.compute_exploitability(
                episodes=exploit_eps,
                seed=seed + 9999 + iteration,
                use_avg_policy=schedule.use_avg_policy,
                config=self.exploit_config,
            )
        except Exception as exc:
            if verbose:
                print(f"[CFR] WARNING: exploitability computation failed at iter {iteration}: {exc}")
            return None

        if self.tlogger is not None:
            for key, val in stats.items():
                self.tlogger.writer.add_scalar(f"Exploitability/{key}", float(val), iteration)
        if verbose:
            print(f"[CFR] Iter {iteration}: exploitability {stats['exploitability']:.4f}")
        return stats

    def train(self, iterations: int = 1000, seed: int = 42, verbose: bool = False, log_every: int = 100,
            eval_every: Optional[int] = None, eval_episodes: int = 32, eval_use_avg_policy: bool = True,
            batch_size: Optional[int] = None, traversals_per_deal: Optional[int] = 1,
            best_save_path: Optional[str] = None, best_save_kind: Optional[str] = None,
            branch_topk_decay: float = 1.0, branch_topk_min: Optional[int] = None,
            exploit_every: Optional[int] = None, exploit_episodes: Optional[int] = None,
            exploit_use_avg_policy: bool = True, debug_topk: Optional[int] = None, debug_dir: Optional[str] = None):
        """Run MCCFR for a given number of iterations with mini-batching.

        Each iteration samples B independent deals and accumulates regrets/strategy.
        """
        eval_schedule = (
            EvaluationSchedule(
                every=eval_every,
                episodes=eval_episodes,
                use_avg_policy=eval_use_avg_policy,
            )
            if (eval_every is not None and eval_every > 0)
            else None
        )
        exploit_schedule = (
            ExploitabilitySchedule(
                every=exploit_every,
                episodes=exploit_episodes,
                use_avg_policy=exploit_use_avg_policy,
            )
            if (exploit_every is not None and exploit_every > 0)
            else None
        )

        debug_topk = int(debug_topk) if (debug_topk is not None and int(debug_topk) > 0) else None
        debug_dump_dir = None
        if debug_topk:
            if debug_dir:
                debug_dump_dir = debug_dir
            elif self.tlogger is not None:
                debug_dump_dir = os.path.join(self.tlogger.get_log_dir(), "debug")
            else:
                debug_dump_dir = os.path.join(os.getcwd(), "cfr_debug")
            os.makedirs(debug_dump_dir, exist_ok=True)

        for it in trange(1, iterations + 1, desc="Iter"):
            self._begin_iteration_stats()
            t0 = time.time()
            if self.branch_topk is not None:
                min_cap = branch_topk_min if branch_topk_min is not None else 1
                if min_cap <= 0:
                    min_cap = 1
                if branch_topk_decay < 1.0:
                    new_val = max(int(self.branch_topk * branch_topk_decay), min_cap)
                    if new_val != self.branch_topk:
                        self.branch_topk = new_val
                elif branch_topk_decay > 1.0:
                    self.branch_topk = max(int(self.branch_topk * branch_topk_decay), self.branch_topk)
            B = int(batch_size) if (batch_size is not None) else int(getattr(self, "batch_size", 1))
            if B <= 0:
                B = 1

            util0_acc = 0.0
            seat0_visits = 0

            for b in range(B):
                rng = Generator(PCG64(seed + it * 100003 + b))
                st = np_init_state(rng)
                trav_per_deal = 1 if traversals_per_deal is None else int(traversals_per_deal)
                if trav_per_deal <= 0:
                    targets = range(4)
                else:
                    targets = [int(self.rng.integers(0, 4)) for _ in range(trav_per_deal)]
                for target in targets:
                    util = self._mccfr(st, target)
                    self._stat_add("utility_count", 1.0)
                    self._stat_add("utility_sum", float(util))
                    self._stat_add("utility_sumsq", float(util * util))
                    if target == 0:
                        self._stat_add("utility_count_seat0", 1.0)
                        self._stat_add("utility_sum_seat0", float(util))
                        self._stat_add("utility_sumsq_seat0", float(util * util))
                        util0_acc += util
                    seat0_visits += 1

            util0_avg = util0_acc / float(seat0_visits) if seat0_visits > 0 else None
            if verbose and it % max(log_every, 1) == 0:
                if seat0_visits > 0:
                    print(f"Iter {it}: util seat0 (avg over {seat0_visits}) ~ {util0_avg:.3f}")
                else:
                    print(f"Iter {it}: util seat0 not sampled this iteration")

            should_emit = self.tlogger is not None and it % max(log_every, 1) == 0
            if should_emit:
                self._log_metrics(it, util0=util0_acc / float(B))
                self._log_metrics(it, util0=util0_avg)
                if seat0_visits > 0:
                    self.tlogger.writer.add_scalar("CFR/seat0_traversals", float(seat0_visits), it)
                if debug_topk:
                    dump_path = os.path.join(debug_dump_dir, f"infosets_{it:06d}.json") if debug_dump_dir else None
                    if dump_path is not None:
                        self.dump_top_infosets(it, debug_topk, dump_path)

            self._finalize_iteration_stats(it, emit=should_emit)

            self._perform_evaluations(
                iteration=it,
                schedule=eval_schedule,
                base_seed=seed,
                best_save_path=best_save_path,
                best_save_kind=best_save_kind,
                verbose=verbose,
            )
            self._maybe_compute_exploitability(
                iteration=it,
                schedule=exploit_schedule,
                default_episodes=eval_episodes,
                seed=seed,
                verbose=verbose,
            )
            t1 = time.time()
            if self.tlogger is not None:
                self.tlogger.writer.add_scalar("dt", t1-t0, it)
    def get_average_policy(self) -> Dict[Tuple[int, bytes], np.ndarray]:
        policy = {}
        for k, strat_sum in self.cum_strategy.items():
            strat = np.asarray(strat_sum, dtype=np.float32)
            total = float(strat.sum())
            if total <= 0.0 or not np.isfinite(total):
                # Fallback: uniform over support if any, else uniform over all actions
                support = (strat > 0).astype(np.float32)
                s = float(support.sum())
                if s > 0:
                    norm = support / s
                else:
                    norm = np.full_like(strat, 1.0 / max(strat.size, 1))
            else:
                norm = strat / total
            policy[k] = norm
        return policy

    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:
        """Select an action via regret-matching at the given observation."""
        infoset_key = (seat, self._obs_key(obs))
        legal = (obs[0] > 0).astype(np.int32)
        if infoset_key not in self.cum_regret:
            self.cum_regret[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.accum_dtype)
        probs = regret_matching(self.cum_regret[infoset_key], legal.astype(np.float32))
        return self._safe_sample(probs, legal, self.rng)

    # --------------------------
    # Save / Load utilities
    # --------------------------
    def save(self, path: str, kind: str = "avg") -> None:
        """Save the trained model.

        kind:
          - 'avg': save normalized average policy per infoset.
          - 'full': save full trainer state (cum_regret and cum_strategy).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if kind == "avg":
            policy = self.get_average_policy()
            payload = {"type": "avg_policy", "policy": policy, "obs_key_mode": getattr(self, "obs_key_mode", "full")}
        elif kind == "full":
            payload = {
                "type": "full",
                "cum_regret": self.cum_regret,
                "cum_strategy": self.cum_strategy,
                "branch_topk": self.branch_topk,
                "max_infosets": self.max_infosets,
                "obs_key_mode": self.obs_key_mode,
                "dtype": str(self.dtype.name),
                "progressive_widening": self.progressive_widening,
                "pw_alpha": self.pw_alpha,
                "pw_tail": self.pw_tail,
                "regret_prune": self.regret_prune,
                "prune_threshold": self.prune_threshold,
                "prune_warmup": self.prune_warmup,
                "prune_reactivation": self.prune_reactivation,
                "subset_cache_size": self._subset_cache_cap,
                "max_branch_actions": self.max_branch_actions,
                "rollout_depth": self.rollout_depth,
                "rollout_samples": self.rollout_samples,
            }
        else:
            raise ValueError(f"Unknown save kind: {kind}")
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    
    def load_avg_policy(path: str, seed: int = 0) -> "SavedPolicy":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("type") == "avg_policy":
            policy = payload["policy"]
            mode = payload.get("obs_key_mode", "full")
        elif payload.get("type") == "full":
            tmp = CFRTrainer(seed=seed)
            tmp.cum_strategy = payload["cum_strategy"]
            policy = tmp.get_average_policy()
            mode = getattr(tmp, "obs_key_mode", "full")
        else:
            raise ValueError("Unsupported payload for avg policy load")
        return SavedPolicy(policy_map=policy, seed=seed, obs_key_mode=mode)


    @staticmethod
    def load_trainer(path: str, tlogger: Optional[object] = None, seed: int = 0) -> "CFRTrainer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("type") != "full":
            raise ValueError("Not a full trainer checkpoint")
        tr = CFRTrainer(seed=seed, tlogger=tlogger,
                        branch_topk=payload.get("branch_topk"),
                        max_infosets=payload.get("max_infosets"),
                        obs_key_mode=payload.get("obs_key_mode", "full"),
                        dtype=np.dtype(payload.get("dtype", "float16")),
                        progressive_widening=payload.get("progressive_widening", True),
                        pw_alpha=float(payload.get("pw_alpha", 0.5)),
                        pw_tail=int(payload.get("pw_tail", 1)),
                        regret_prune=payload.get("regret_prune", True),
                        prune_threshold=float(payload.get("prune_threshold", -1.0)),
                        prune_warmup=int(payload.get("prune_warmup", 64)),
                        prune_reactivation=int(payload.get("prune_reactivation", 16)),
                        subset_cache_size=int(payload.get("subset_cache_size", 4096)),
                        max_branch_actions=int(payload.get("max_branch_actions", 0)),
                        rollout_depth=int(payload.get("rollout_depth", 0)),
                        rollout_samples=int(payload.get("rollout_samples", 0)))
        tr.cum_regret = payload["cum_regret"]
        tr.cum_strategy = payload["cum_strategy"]
        return tr



class SavedPolicy:
    """Lightweight actor for a saved average policy map."""
    def __init__(self, policy_map: Dict[Tuple[int, bytes], np.ndarray], seed: int = 0, obs_key_mode: str = "full"):
        self.policy_map = policy_map
        self.rng = Generator(PCG64(seed))
        self.obs_key_mode = str(obs_key_mode)

    def _obs_key(self, seat: int, obs: np.ndarray) -> Tuple[int, bytes]:
        mode = self.obs_key_mode
        if mode == "hand_table":
            obs_slice = obs[:2]
        elif mode == "compact":
            obs_slice = obs[:3]
        else:
            obs_slice = obs
        planes = (np.asarray(obs_slice, dtype=np.float32) > 0.5).astype(np.uint8)
        packed = []
        for plane in planes:
            b = np.packbits(plane, bitorder="little")  # 40->5 bytes
            packed.append(b[:5].tobytes())
        return (seat, b"".join(packed))

    def act(self, seat: int, obs: np.ndarray) -> int:
        """Sample an action from the saved average policy, restricted to legal."""
        legal = (obs[0] > 0).astype(np.int32)
        legal_idx = np.nonzero(legal)[0]
        if legal_idx.size == 0:
            return 0

        # Try the configured keying first, then fallbacks
        keys = [self._obs_key(seat, obs)]
        for alt in ("compact", "hand_table", "full"):
            if alt != self.obs_key_mode:
                old = self.obs_key_mode
                self.obs_key_mode = alt
                keys.append(self._obs_key(seat, obs))
                self.obs_key_mode = old

        probs = None
        for k in keys:
            probs = self.policy_map.get(k)
            if probs is not None:
                break
        if probs is None:
            return int(self.rng.choice(legal_idx))

        p = np.asarray(probs, dtype=np.float64)[legal_idx]
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            return int(self.rng.choice(legal_idx))
        p /= s
        adjust = 1.0 - float(p.sum())
        if abs(adjust) > 1e-15:
            p[-1] += adjust
        p = np.clip(p, 0.0, 1.0)
        s2 = float(p.sum())
        if s2 <= 0.0 or not np.isfinite(s2):
            return int(self.rng.choice(legal_idx))
        p /= s2
        return int(self.rng.choice(legal_idx, p=p))

    def act_with_mask(self, seat: int, obs: np.ndarray, mask: np.ndarray) -> int:
        return self.act_from_obs(seat, obs)

    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:
        return self.act(seat, obs)
