from typing import Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
import os
import pickle
import hashlib

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import trange


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
        self.last_capture_player = np.int32(-1)  # awarding leftover table at end


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


# ---------- helper capture masks (single-over-subset priority) ----------

def _single_match_mask(table: np.ndarray, target_rank: int) -> np.ndarray:
    """Mask with True at indices where a single equal-rank capture is possible."""
    if target_rank <= 0:
        return np.zeros((NUM_CARDS,), dtype=bool)
    return (table.astype(bool) & (CARD_RANKS == target_rank))


def _subset_sum_mask(table: np.ndarray, target_rank: int) -> np.ndarray:
    """Subset-sum DP over table ranks to reach target_rank (fallback if no single)."""
    dp_has = np.zeros(NUM_RANKS + 1, dtype=bool)
    dp_has[0] = True
    dp_mask = np.zeros((NUM_RANKS + 1, NUM_CARDS), dtype=bool)
    for i in range(NUM_CARDS):
        if table[i] != 1:
            continue
        r = int(CARD_RANKS[i])
        for s in range(NUM_RANKS, r - 1, -1):
            if not dp_has[s] and dp_has[s - r]:
                dp_has[s] = True
                dp_mask[s] = dp_mask[s - r].copy()
                dp_mask[s][i] = True
    if target_rank >= 0 and target_rank <= NUM_RANKS and dp_has[target_rank]:
        return dp_mask[target_rank]
    return np.zeros((NUM_CARDS,), dtype=bool)


def _best_capture_mask(table: np.ndarray, target_rank: int) -> np.ndarray:
    """Priority rule: prefer single equal-rank, else subset-sum combination."""
    sm = _single_match_mask(table, target_rank)
    if sm.any():
        return sm
    return _subset_sum_mask(table, target_rank)


# ------------------- rollout/np_* env with corrected rules -------------------

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


def np_step(st: NState, action_idx: int) -> Tuple[NState, bool]:
    """Apply action to state with:
       - single-over-subset priority
       - Ace sweep counts scopa (unless last move)
       - track last_capture_player
    """
    seat = int(st.cur_player)
    # Remove from hand, mark history
    st.hands[seat, action_idx] = 0
    st.history[seat, action_idx] = 1
    rank = int(CARD_RANKS[action_idx])

    scopa = False

    if rank == 1:
        # Ace captures entire table + played card
        table_nonempty = (st.table.sum() > 0)
        captured = st.table.astype(np.int8).copy()
        captured[action_idx] = 1
        st.captures[seat] += captured
        st.table[:] = 0
        st.last_capture_player = np.int32(seat)
        # Ace sweep is a scopa iff table had cards before
        scopa = bool(table_nonempty)
        if scopa:
            st.scopas[seat] += 1
    else:
        # Priority: single equal rank > any combination
        subset_mask = _best_capture_mask(st.table, rank)
        if subset_mask.any():
            st.table[subset_mask] = 0
            captured = subset_mask.astype(np.int8)
            captured[action_idx] = 1
            st.captures[seat] += captured
            scopa = (st.table.sum() == 0)
            if scopa:
                st.scopas[seat] += 1
            st.last_capture_player = np.int32(seat)
        else:
            st.table[action_idx] = 1
            scopa = False

    # If this move ends the hand, do NOT count a scopa made on the last move
    st.cur_player = np.int32((seat + 1) % 4)
    if (st.hands.sum() == 0) and scopa:
        st.scopas[seat] = max(0, int(st.scopas[seat]) - 1)
        scopa = False

    return st, scopa


def np_is_terminal(st: NState) -> bool:
    return int(st.hands.sum()) == 0


def _award_leftover_to_last_capturer(st: NState) -> None:
    """At terminal, award remaining table cards to the last capturer (if any)."""
    if st.last_capture_player >= 0 and st.table.sum() > 0:
        idx = np.nonzero(st.table)[0]
        if idx.size > 0:
            st.captures[int(st.last_capture_player), idx] += 1
            st.table[idx] = 0


def np_evaluate_round(st: NState) -> Tuple[int, int]:
    # Award leftover table first
    _award_leftover_to_last_capturer(st)

    # Team captures
    team0_mask = np.array([1, 0, 1, 0], dtype=np.int32)
    team1_mask = np.array([0, 1, 0, 1], dtype=np.int32)
    team0_caps = (st.captures.T @ team0_mask).astype(np.int32)
    team1_caps = (st.captures.T @ team1_mask).astype(np.int32)

    t0 = 0
    t1 = 0

    # Scopas
    t0 += int(st.scopas @ team0_mask)
    t1 += int(st.scopas @ team1_mask)

    # Most cards
    c0 = int(team0_caps.sum())
    c1 = int(team1_caps.sum())
    t0 += 1 if c0 > c1 else 0
    t1 += 1 if c1 > c0 else 0

    # Most coins (bello suit == 3)
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

    # Primiera priorities (same as env)
    RANK_PRIORITY = np.array([0, 2, 0, 0, 0, 1, 3, 4, 0, 0, 0], dtype=np.int32)
    pri = RANK_PRIORITY[1:]
    by_suit_rank0 = team0_caps.reshape(NUM_SUITS, NUM_RANKS)
    by_suit_rank1 = team1_caps.reshape(NUM_SUITS, NUM_RANKS)
    prim0 = int((by_suit_rank0 * pri).max(axis=1).sum())
    prim1 = int((by_suit_rank1 * pri).max(axis=1).sum())
    t0 += 1 if prim0 > prim1 else 0
    t1 += 1 if prim1 > prim0 else 0

    if t0 > t1:
        return 1, -1
    if team_points[1] > team_points[0]:
        return -1, 1
    return 0, 0


# -------------------------- CFR utilities --------------------------

def regret_matching(regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0) * legal_mask
    s = float(pos.sum())
    if s > 0.0:
        return pos / s
    lm = legal_mask.astype(np.float32)
    denom = float(max(int(lm.sum()), 1))
    return lm / denom


# -------------------------- CFRTrainer --------------------------

class CFRTrainer:
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

        bt = branch_topk if (branch_topk is not None and int(branch_topk) > 0) else None
        self.branch_topk = bt
        self.progressive_widening = bool(progressive_widening) and (self.branch_topk is not None)
        self.pw_alpha = max(float(pw_alpha), 0.0)
        self.pw_tail = max(int(pw_tail), 0)

        self.regret_prune = bool(regret_prune)
        self.prune_threshold = float(prune_threshold)
        self.prune_warmup = max(int(prune_warmup), 0)
        self.prune_reactivation = max(int(prune_reactivation), 1)

        # caches, visits, memory limits
        self._infoset_visits: Dict[Tuple[int, bytes], int] = {}
        self._subset_cache: OrderedDict[bytes, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._subset_cache_cap = max(int(subset_cache_size), 0)
        self.max_branch_actions = int(max_branch_actions) if (max_branch_actions and max_branch_actions > 0) else None
        if self.max_branch_actions is not None and self.branch_topk is not None:
            self.branch_topk = min(self.branch_topk, self.max_branch_actions)

        self.rollout_depth = int(rollout_depth) if (rollout_depth and rollout_depth > 0) else None
        self.rollout_samples = max(int(rollout_samples), 1) if self.rollout_depth is not None else 0
        self._max_rollout_steps = NUM_CARDS * 2

        self.max_infosets = int(max_infosets) if (max_infosets is not None and max_infosets > 0) else None
        self._last_seen: Dict[Tuple[int, bytes], int] = {}
        self._clock: int = 0
        self.obs_key_mode = str(obs_key_mode)

        table_dtype = np.dtype(dtype)
        self.dtype = table_dtype
        # accumulators at least float32
        accum_dtype = np.dtype(np.float32)
        if table_dtype.itemsize >= accum_dtype.itemsize:
            accum_dtype = table_dtype
        self.accum_dtype = accum_dtype

        # RNG
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

        # infoset tables
        self.cum_regret: Dict[Tuple[int, bytes], np.ndarray] = {}
        self.cum_strategy: Dict[Tuple[int, bytes], np.ndarray] = {}

    # --- strategy helpers ---

    def _get_strategy(self, infoset_key, legal_mask: np.ndarray) -> np.ndarray:
        if infoset_key not in self.cum_regret:
            self.cum_regret[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.accum_dtype)
            self.cum_strategy[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.accum_dtype)
            self._touch_key(infoset_key)
            self._evict_if_needed(avoid_key=infoset_key)
        else:
            self._touch_key(infoset_key)
        return regret_matching(np.asarray(self.cum_regret[infoset_key], dtype=np.float32),
                               legal_mask.astype(np.float32))

    def _peek_strategy(self, infoset_key: Tuple[int, bytes], legal_mask: np.ndarray) -> np.ndarray:
        if infoset_key in self.cum_regret:
            regrets = np.asarray(self.cum_regret[infoset_key], dtype=np.float32)
            return regret_matching(regrets, legal_mask.astype(np.float32))
        legal = legal_mask.astype(np.float32)
        total = float(legal.sum())
        return legal / total if total > 0 else legal

    # --- caches / visits ---

    def _subset_sum_mask_cached(self, table: np.ndarray, target_rank: int) -> np.ndarray:
        """Cache only the subset-sum; caller handles single-over-subset priority."""
        if target_rank <= 0:
            return np.zeros((NUM_CARDS,), dtype=bool)
        if self._subset_cache_cap <= 0 or table.sum() <= 0:
            return _subset_sum_dp_numpy(table)
        key_bytes = np.packbits(table.astype(np.uint8), bitorder="little").tobytes()
        cached = self._subset_cache.get(key_bytes)
        if cached is not None:
            self._subset_cache.move_to_end(key_bytes)
            return cached
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
        cnt = self._infoset_visits.get(infoset_key, 0) + 1
        self._infoset_visits[infoset_key] = cnt
        return cnt

    # --- regret pruning / branching ---

    def _apply_regret_pruning(self, infoset_key: Tuple[int, bytes], actions: np.ndarray, visit_count: int) -> np.ndarray:
        if not self.regret_prune or actions.size == 0 or visit_count <= self.prune_warmup:
            return actions
        if visit_count % self.prune_reactivation == 0:
            return actions
        regrets = np.asarray(self.cum_regret[infoset_key], dtype=np.float32)[actions]
        keep = regrets > self.prune_threshold
        return actions[keep] if keep.any() else actions

    def _sample_weighted_subset(self, choices: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
        if choices.size == 0 or k <= 0:
            return np.empty((0,), dtype=choices.dtype)
        probs = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        tot = float(probs.sum())
        if tot <= 0.0 or not np.isfinite(tot):
            probs = np.full(choices.size, 1.0 / float(choices.size), dtype=np.float64)
        else:
            probs /= tot
            adj = 1.0 - float(probs.sum())
            if abs(adj) > 1e-15:
                probs[-1] += adj
            probs = np.clip(probs, 0.0, 1.0)
            s2 = float(probs.sum())
            probs = (probs / s2) if s2 > 0.0 and np.isfinite(s2) else np.full(choices.size, 1.0 / float(choices.size))
        idx = self.rng.choice(choices.size, size=min(k, choices.size), replace=False, p=probs)
        return choices[np.atleast_1d(idx)]

    def _select_branch_actions(self, infoset_key: Tuple[int, bytes], legal_actions: np.ndarray,
                               sigma: np.ndarray, visit_count: int) -> np.ndarray:
        actions = legal_actions
        if actions.size == 0:
            return actions
        pruned = self._apply_regret_pruning(infoset_key, actions, visit_count)
        actions = pruned if pruned.size > 0 else actions

        eff_limit = self.branch_topk
        if eff_limit is None and self.max_branch_actions is not None:
            eff_limit = self.max_branch_actions
        width_cap = actions.size

        if eff_limit is None or actions.size <= eff_limit:
            selected = actions
        else:
            probs_masked = sigma[actions]
            width_cap = min(actions.size, eff_limit)
            if self.progressive_widening:
                extra = int(np.power(max(visit_count - 1, 0), self.pw_alpha))
                width_cap = min(actions.size, eff_limit + extra)
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
                        selected = np.concatenate([selected, sampled.astype(selected.dtype)])

        selected = np.unique(selected.astype(np.int32))
        if self.max_branch_actions is not None and selected.size > self.max_branch_actions:
            down = self._sample_weighted_subset(selected, sigma[selected], self.max_branch_actions)
            selected = down.astype(np.int32) if down.size > 0 else selected[:self.max_branch_actions]
        return np.sort(selected.astype(np.int32))

    # --- apply / undo actions (mirror env rules) ---

    def _apply_action_inplace(self, st: NState, action_idx: int) -> ActionDiff:
        seat = int(st.cur_player)
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

        # play card
        st.hands[seat, action_idx] = 0
        st.history[seat, action_idx] = 1
        rank = int(CARD_RANKS[action_idx])

        scopa_now = False

        if rank == 1:
            # Ace sweep
            table_idx = np.nonzero(st.table)[0]
            if table_idx.size > 0:
                diff.table_indices = table_idx
                diff.table_prev = st.table[table_idx].copy()
                st.table[table_idx] = 0
            captured = np.zeros((NUM_CARDS,), dtype=np.int8)
            if table_idx.size > 0:
                captured[table_idx] = 1
            captured[action_idx] = 1
            cap_idx = np.nonzero(captured)[0]
            if cap_idx.size > 0:
                diff.captures_indices = cap_idx
                st.captures[seat, cap_idx] += 1
            st.last_capture_player = np.int32(seat)
            scopa_now = (table_idx.size > 0)
            if scopa_now:
                st.scopas[seat] += 1
                diff.scopa_delta = 1
        else:
            # Priority: single over subset
            single_mask = _single_match_mask(st.table, rank)
            if single_mask.any():
                subset_mask = single_mask
            else:
                subset_mask = self._subset_sum_mask_cached(st.table, rank)

            if subset_mask.any():
                table_idx = np.nonzero(subset_mask)[0]
                diff.table_indices = table_idx
                diff.table_prev = st.table[table_idx].copy()
                st.table[table_idx] = 0
                captured = subset_mask.astype(np.int8)
                captured[action_idx] = 1
                cap_idx = np.nonzero(captured)[0]
                if cap_idx.size > 0:
                    diff.captures_indices = cap_idx
                    st.captures[seat, cap_idx] += 1
                scopa_now = (st.table.sum() == 0)
                if scopa_now:
                    st.scopas[seat] += 1
                    diff.scopa_delta = 1
                st.last_capture_player = np.int32(seat)
            else:
                # place on table
                diff.table_indices = np.array([action_idx], dtype=np.int32)
                diff.table_prev = np.array([st.table[action_idx]], dtype=np.int8)
                st.table[action_idx] = 1
                scopa_now = False

        # next player
        st.cur_player = np.int32((seat + 1) % 4)

        # If last action of the hand, undo scopa counted on this move
        if (st.hands.sum() == 0) and scopa_now and diff.scopa_delta == 1:
            st.scopas[seat] -= 1
            diff.scopa_delta = 0  # reflect that we undid it

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

    # --- rollout / evaluation helpers ---

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
        """Sample robustly over the legal set; fallback to uniform if degenerate."""
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
        t0, t1 = np_evaluate_round(st)
        return float(t0 if (seat % 2 == 0) else t1)

    # --- infoset keying / memory management ---

    def _obs_key(self, obs: np.ndarray) -> bytes:
        """Bit-pack observation planes to bytes.
        Full: 6×40 -> 30 bytes; compact: 3×40 -> 15B; hand_table: 2×40 -> 10B.
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

    # --- core MCCFR traversal ---

    def _mccfr(self, st: NState, target_seat: int,
               reach_my: float = 1.0,
               reach_others: float = 1.0,
               sample_reach: float = 1.0,
               depth: int = 0) -> float:
        if np_is_terminal(st):
            payoff = self._evaluate_utility(st, target_seat)
            if sample_reach <= 0.0:
                return payoff
            # Leaf returns weighted by (reach_others / sample_reach)
            return float(reach_others / sample_reach) * payoff

        if self.rollout_depth is not None and depth >= self.rollout_depth:
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
                    baseline = (util_acc / max(prob_acc, 1e-12)) if prob_acc > 0.0 else 0.0
                    tail_probs = np.clip(sigma[unseen_actions], 0.0, None)
                    action_values[unseen_actions] = baseline
                    util += float(tail_probs.sum()) * baseline

            # Update regrets (no extra weights)
            regrets = np.zeros(NUM_CARDS, dtype=np.float32)
            if legal_actions.size > 0:
                regrets[legal_actions] = (action_values[legal_actions] - util)
            self.cum_regret[infoset_key] += regrets.astype(self.accum_dtype)
            if self.rm_plus:
                self.cum_regret[infoset_key] = np.maximum(
                    np.asarray(self.cum_regret[infoset_key], dtype=np.float32),
                    0.0
                ).astype(self.accum_dtype)

            # Average strategy update
            strategy_weight = reach_my / max(sample_reach, 1e-12)
            self.cum_strategy[infoset_key] += (sigma * strategy_weight).astype(self.accum_dtype)
            return util

        # Opponent/teammate: sample one action
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

    # --------------------------
    # Evaluation & training
    # --------------------------

    def _category_breakdown(self, st: NState) -> Dict[str, int]:
        # Team masks
        team0_mask = np.array([1, 0, 1, 0], dtype=np.int32)
        team1_mask = np.array([0, 1, 0, 1], dtype=np.int32)

        team0_caps = (st.captures.T @ team0_mask).astype(np.int32)
        team1_caps = (st.captures.T @ team1_mask).astype(np.int32)

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

        # Primiera
        RANK_PRIORITY = np.array([0, 2, 0, 0, 0, 1, 3, 4, 0, 0, 0], dtype=np.int32)
        pri = RANK_PRIORITY[1:]
        by_suit_rank0 = team0_caps.reshape(NUM_SUITS, NUM_RANKS)
        by_suit_rank1 = team1_caps.reshape(NUM_SUITS, NUM_RANKS)
        prim0 = int((by_suit_rank0 * pri).max(axis=1).sum())
        prim1 = int((by_suit_rank1 * pri).max(axis=1).sum())
        pr0 = 1 if prim0 > prim1 else 0
        pr1 = 1 if prim1 > prim0 else 0

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
        if rng is None:
            rng = self.rng
        obs = np_build_obs(st, seat)
        legal = (obs[0] > 0).astype(np.float32)
        legal_actions = np.nonzero(legal)[0]

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
            # Fallback to current

        infoset_key = (seat, self._obs_key(obs))
        sigma = self._get_strategy(infoset_key, legal.astype(np.int32))
        return self._safe_sample(sigma, legal, rng)

    def evaluate(self, episodes: int = 50, seed: int = 12345,
                 selfplay: bool = True, use_avg_policy: bool = True):
        rng = Generator(PCG64(seed))

        wins0 = 0
        wins1 = 0
        draws = 0
        sums = {
            "points0": 0.0, "points1": 0.0,
            "scopas0": 0.0, "scopas1": 0.0,
            "most_cards0": 0.0, "most_cards1": 0.0,
            "most_coins0": 0.0, "most_coins1": 0.0,
            "sette_bello0": 0.0, "sette_bello1": 0.0,
            "primiera0": 0.0, "primiera1": 0.0,
        }

        for ep in range(episodes):
            st = np_init_state(Generator(PCG64(seed + 1000 + ep)))
            while not np_is_terminal(st):
                seat = int(st.cur_player)
                if selfplay:
                    mode = "avg" if use_avg_policy else "current"
                else:
                    mode = ("avg" if use_avg_policy else "current") if (seat % 2 == 0) else "random"
                a = self._select_action_eval(st, seat, mode, None, rng)
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
        for k, v in sums.items():
            out[f"avg_{k}"] = v * inv_eps
        return out

    def train(self, iterations: int = 1000, seed: int = 42, verbose: bool = False, log_every: int = 100,
              eval_every: Optional[int] = None, eval_episodes: int = 32, eval_use_avg_policy: bool = True,
              batch_size: Optional[int] = None):
        for it in trange(1, iterations + 1, desc="Iter"):
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
                    if target == 0:
                        util0_acc += util
                    seat0_visits += 1

            util0_avg = util0_acc / float(seat0_visits) if seat0_visits > 0 else None
            if verbose and it % max(log_every, 1) == 0:
                if seat0_visits > 0:
                    print(f"Iter {it}: util seat0 (avg over {seat0_visits}) ~ {util0_avg:.3f}")
                else:
                    print(f"Iter {it}: util seat0 not sampled this iteration")

            if self.tlogger is not None and it % max(log_every, 1) == 0:
                self._log_metrics(it, util0=util0_acc / float(B))

            do_eval = (eval_every is not None and eval_every > 0 and (it % eval_every == 0))
            if do_eval and self.tlogger is not None:
                selfplay_metrics = self.evaluate(episodes=eval_episodes, seed=seed + 7777 + it,
                                                 selfplay=True, use_avg_policy=eval_use_avg_policy)
                for k, v in selfplay_metrics.items():
                    self.tlogger.writer.add_scalar(f"EvalSelf/{k}", float(v), it)

                vsrnd_metrics = self.evaluate(episodes=eval_episodes, seed=seed + 8888 + it,
                                              selfplay=False, use_avg_policy=eval_use_avg_policy)
                for k, v in vsrnd_metrics.items():
                    self.tlogger.writer.add_scalar(f"EvalVsRandom/{k}", float(v), it)

    def _metrics_snapshot(self) -> Dict[str, float]:
        n_infosets = float(len(self.cum_regret))
        if n_infosets == 0:
            return {"num_infosets": 0.0, "avg_regret_abs": 0.0}
        total_abs = 0.0
        count = 0
        for arr in self.cum_regret.values():
            a = np.asarray(arr)
            total_abs += np.abs(a).sum()
            count += a.size
        avg_abs = float(total_abs / max(count, 1))
        return {"num_infosets": n_infosets, "avg_regret_abs": avg_abs}

    def _log_metrics(self, it: int, util0: Optional[float] = None):
        if self.tlogger is None:
            return
        snap = self._metrics_snapshot()
        self.tlogger.writer.add_scalar("CFR/num_infosets", snap["num_infosets"], it)
        self.tlogger.writer.add_scalar("CFR/avg_regret_abs", snap["avg_regret_abs"], it)
        if util0 is not None:
            self.tlogger.writer.add_scalar("CFR/util_seat0", float(util0), it)

    # --------------------------
    # Policy I/O
    # --------------------------

    def get_average_policy(self) -> Dict[Tuple[int, bytes], np.ndarray]:
        policy = {}
        for k, strat_sum in self.cum_strategy.items():
            strat = np.asarray(strat_sum, dtype=np.float32)
            total = float(strat.sum())
            if total <= 0.0 or not np.isfinite(total):
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
                        max_branch_actions=int(payload.get("max_branch_actions", 0)) if payload.get("max_branch_actions", 0) else 0,
                        rollout_depth=int(payload.get("rollout_depth", 0)),
                        rollout_samples=int(payload.get("rollout_samples", 0)))
        tr.cum_regret = payload["cum_regret"]
        tr.cum_strategy = payload["cum_strategy"]
        return tr


# -------------------------- SavedPolicy --------------------------

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

        # Try configured keying first, then fallbacks
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
