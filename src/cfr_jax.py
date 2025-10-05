from typing import Dict, Tuple, Optional
import os
import pickle

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


class NState:
    def __init__(self):
        self.hands = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.table = np.zeros((NUM_CARDS,), dtype=np.int8)
        self.captures = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.history = np.zeros((4, NUM_CARDS), dtype=np.int8)
        self.scopas = np.zeros((4,), dtype=np.int32)
        self.cur_player = np.int32(0)
        self.last_capture_player = np.int32(-1)


def np_init_state(rng: Generator) -> NState:
    st = NState()
    perm = rng.permutation(NUM_CARDS)
    for seat in range(4):
        idx = perm[seat * 10:(seat + 1) * 10]
        st.hands[seat, idx] = 1
    st.cur_player = np.int32(rng.integers(0, 4))
    return st


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


def _subset_sum_mask(table: np.ndarray, target_rank: int) -> np.ndarray:
    dp_has = np.zeros(NUM_RANKS + 1, dtype=bool)
    dp_has[0] = True
    dp_mask = np.zeros((NUM_RANKS + 1, NUM_CARDS), dtype=bool)
    # Iterate cards present on table
    for i in range(NUM_CARDS):
        if table[i] != 1:
            continue
        r = int(CARD_RANKS[i])
        for s in range(NUM_RANKS, r - 1, -1):
            if not dp_has[s] and dp_has[s - r]:
                dp_has[s] = True
                dp_mask[s] = dp_mask[s - r].copy()
                dp_mask[s][i] = True
    if dp_has[target_rank]:
        return dp_mask[target_rank]
    return np.zeros((NUM_CARDS,), dtype=bool)


def np_step(st: NState, action_idx: int) -> tuple[NState, bool]:
    seat = int(st.cur_player)
    # Remove from hand, mark history
    st.hands[seat, action_idx] = 0
    st.history[seat, action_idx] = 1
    rank = int(CARD_RANKS[action_idx])

    if rank == 1:
        # Ace captures entire table + played card (no Scopa in env)
        subset_mask = st.table.astype(bool)
        captured = subset_mask.astype(np.int8)
        captured[action_idx] = 1
        st.captures[seat] += captured
        st.table[:] = 0
        st.last_capture_player = np.int32(seat)
        scopa = False
    else:
        subset_mask = _subset_sum_mask(st.table, rank)
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

    st.cur_player = np.int32((seat + 1) % 4)
    return st, scopa


def np_is_terminal(st: NState) -> bool:
    return int(st.hands.sum()) == 0


def np_evaluate_round(st: NState) -> tuple[int, int]:
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

    # Most coins (suit == bello -> suit_id 3)
    is_bello = (CARD_SUITS == 3).astype(np.int32)
    b0 = int((team0_caps * is_bello).sum())
    b1 = int((team1_caps * is_bello).sum())
    t0 += 1 if b0 > b1 else 0
    t1 += 1 if b1 > b0 else 0

    # Sette bello
    sette_bello_idx = 3 * NUM_RANKS + (7 - 1)
    t0 += 1 if team0_caps[sette_bello_idx] > 0 else 0
    t1 += 1 if team1_caps[sette_bello_idx] > 0 else 0

    # Primiera priorities
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
    if t1 > t0:
        return -1, 1
    return 0, 0


# --------------------------
# CFR Trainer (NumPy)
# --------------------------

def regret_matching(regrets: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0) * legal_mask
    s = pos.sum()
    if s > 0:
        return pos / s
    # Uniform over legal
    denom = max(int(legal_mask.sum()), 1)
    return legal_mask.astype(np.float32) / denom


class CFRTrainer:
    def __init__(self, seed: int = 42, tlogger: Optional[object] = None, branch_topk: Optional[int] = None,
                 max_infosets: Optional[int] = None, obs_key_mode: str = "full", dtype: np.dtype = np.float16):
        self.rng = Generator(PCG64(seed))
        self.tlogger = tlogger
        self.branch_topk = branch_topk  # limit branching at traverser nodes for speed

        # Memory controls
        self.max_infosets = int(max_infosets) if (max_infosets is not None and max_infosets > 0) else None
        self._last_seen: Dict[Tuple[int, bytes], int] = {}
        self._clock: int = 0
        self.obs_key_mode = str(obs_key_mode)
        self.dtype = np.dtype(dtype)

        # infoset -> arrays of size 40 (global card index space)
        self.cum_regret: Dict[Tuple[int, bytes], np.ndarray] = {}
        self.cum_strategy: Dict[Tuple[int, bytes], np.ndarray] = {}

    def _get_strategy(self, infoset_key, legal_mask: np.ndarray) -> np.ndarray:
        if infoset_key not in self.cum_regret:
            self.cum_regret[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.dtype)
            self.cum_strategy[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.dtype)
            self._touch_key(infoset_key)
            self._evict_if_needed(avoid_key=infoset_key)
        else:
            self._touch_key(infoset_key)
        # Use float32 math for stability
        return regret_matching(np.asarray(self.cum_regret[infoset_key], dtype=np.float32), legal_mask.astype(np.float32))

    def _safe_sample(self, p_arr: np.ndarray, legal_mask: np.ndarray, rng: Optional[Generator] = None) -> int:
        """Sample an action robustly over the legal set.

        - Works on the restricted legal index set to avoid tiny normalization errors.
        - Falls back to uniform legal if probabilities are degenerate.
        """
        if rng is None:
            rng = self.rng
        lm = (legal_mask > 0).astype(np.int32)
        legal_idx = np.nonzero(lm)[0]
        if legal_idx.size == 0:
            return 0
        p = np.asarray(p_arr, dtype=np.float64)[legal_idx]
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            return int(rng.choice(legal_idx))
        p /= s
        # Correct any tiny drift
        adjust = 1.0 - float(p.sum())
        if abs(adjust) > 1e-15:
            p[-1] += adjust
        # Clip and renormalize for safety
        p = np.clip(p, 0.0, 1.0)
        s2 = float(p.sum())
        if s2 <= 0.0 or not np.isfinite(s2):
            return int(rng.choice(legal_idx))
        p /= s2
        return int(rng.choice(legal_idx, p=p))

    def _evaluate_utility(self, st: NState, seat: int) -> float:
        t0, t1 = np_evaluate_round(st)
        return float(t0 if (seat % 2 == 0) else t1)

    def _obs_key(self, obs: np.ndarray) -> bytes:
        # Compact, deterministic key for an observation (16-byte BLAKE2b digest)
        # Optionally coarsen the key to reduce unique infosets and memory.
        mode = self.obs_key_mode
        if mode == "hand_table":
            obs_slice = obs[:2]
        elif mode == "compact":
            obs_slice = obs[:3]
        else:
            obs_slice = obs
        view = np.ascontiguousarray(obs_slice).view(np.uint8)
        return hashlib.blake2b(view, digest_size=16).digest()

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

    def _mccfr(self, st: NState, target_seat: int) -> float:
        if np_is_terminal(st):
            return self._evaluate_utility(st, target_seat)

        cur_seat = int(st.cur_player)
        infoset_key, legal_mask = self._infoset(st, cur_seat)
        sigma = self._get_strategy(infoset_key, legal_mask)

        legal_actions = np.nonzero(legal_mask)[0]
        if cur_seat == target_seat:
            # Deterministic branching over legal actions for traverser
            action_values = np.zeros(NUM_CARDS, dtype=np.float32)
            util = 0.0
            # Optionally limit branching to top-K by current policy
            branch_actions = legal_actions
            if self.branch_topk is not None and self.branch_topk > 0 and branch_actions.size > self.branch_topk:
                probs_masked = sigma[branch_actions]
                top_idx = np.argsort(probs_masked)[-self.branch_topk:]
                branch_actions = branch_actions[top_idx]
            for a in branch_actions:
                # Copy-on-write lightweight state
                st_next = NState()
                st_next.hands = st.hands.copy()
                st_next.table = st.table.copy()
                st_next.captures = st.captures.copy()
                st_next.history = st.history.copy()
                st_next.scopas = st.scopas.copy()
                st_next.cur_player = st.cur_player
                st_next.last_capture_player = st.last_capture_player

                st_next, _ = np_step(st_next, int(a))
                v = self._mccfr(st_next, target_seat)
                action_values[int(a)] = v
                util += float(sigma[int(a)]) * float(v)

            regrets = np.zeros(NUM_CARDS, dtype=np.float32)
            # Only actions we evaluated contribute; others stay at 0 for this visit
            regrets[branch_actions] = action_values[branch_actions] - util
            self.cum_regret[infoset_key] += regrets.astype(self.dtype)
            self.cum_strategy[infoset_key] += sigma.astype(self.dtype)
            return util
        else:
            # External sampling for others
            # Robust sampling over legal actions using current policy
            a = self._safe_sample(sigma, legal_mask, self.rng)

            st_next = NState()
            st_next.hands = st.hands.copy()
            st_next.table = st.table.copy()
            st_next.captures = st.captures.copy()
            st_next.history = st.history.copy()
            st_next.scopas = st.scopas.copy()
            st_next.cur_player = st.cur_player
            st_next.last_capture_player = st.last_capture_player

            st_next, _ = np_step(st_next, a)
            # Track average strategy
            if infoset_key not in self.cum_strategy:
                self.cum_strategy[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.dtype)
                self._touch_key(infoset_key)
                self._evict_if_needed(avoid_key=infoset_key)
            self.cum_strategy[infoset_key] += sigma.astype(self.dtype)
            return self._mccfr(st_next, target_seat)

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
    # Evaluation utilities
    # --------------------------
    def _category_breakdown(self, st: NState) -> Dict[str, int]:
        """Compute Scopa category wins and counts for each team from terminal state."""
        # Team masks (team 0: seats 0 and 2; team 1: seats 1 and 3)
        team0_mask = np.array([1, 0, 1, 0], dtype=np.int32)
        team1_mask = np.array([0, 1, 0, 1], dtype=np.int32)

        team0_caps = (st.captures.T @ team0_mask).astype(np.int32)
        team1_caps = (st.captures.T @ team1_mask).astype(np.int32)

        # Scopas (counts)
        scopa0 = int(st.scopas @ team0_mask)
        scopa1 = int(st.scopas @ team1_mask)

        # Most cards
        c0 = int(team0_caps.sum())
        c1 = int(team1_caps.sum())
        mc0 = 1 if c0 > c1 else 0
        mc1 = 1 if c1 > c0 else 0

        # Most coins (bello suit == 3)
        is_bello = (CARD_SUITS == 3).astype(np.int32)
        b0 = int((team0_caps * is_bello).sum())
        b1 = int((team1_caps * is_bello).sum())
        mb0 = 1 if b0 > b1 else 0
        mb1 = 1 if b1 > b0 else 0

        # Sette bello
        sette_bello_idx = 3 * NUM_RANKS + (7 - 1)
        sb0 = 1 if team0_caps[sette_bello_idx] > 0 else 0
        sb1 = 1 if team1_caps[sette_bello_idx] > 0 else 0

        # Primiera (priority values consistent with env scoring)
        RANK_PRIORITY = np.array([0, 2, 0, 0, 0, 1, 3, 4, 0, 0, 0], dtype=np.int32)
        pri = RANK_PRIORITY[1:]
        by_suit_rank0 = team0_caps.reshape(NUM_SUITS, NUM_RANKS)
        by_suit_rank1 = team1_caps.reshape(NUM_SUITS, NUM_RANKS)
        prim0 = int((by_suit_rank0 * pri).max(axis=1).sum())
        prim1 = int((by_suit_rank1 * pri).max(axis=1).sum())
        pr0 = 1 if prim0 > prim1 else 0
        pr1 = 1 if prim1 > prim0 else 0

        # Aggregate points (category points + scopas)
        points0 = scopa0 + mc0 + mb0 + sb0 + pr0
        points1 = scopa1 + mc1 + mb1 + sb1 + pr1

        return {
            "points0": points0,
            "points1": points1,
            "scopas0": scopa0,
            "scopas1": scopa1,
            "most_cards0": mc0,
            "most_cards1": mc1,
            "most_coins0": mb0,
            "most_coins1": mb1,
            "sette_bello0": sb0,
            "sette_bello1": sb1,
            "primiera0": pr0,
            "primiera1": pr1,
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
        # Category sums
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
                # Determine which mode each seat uses
                if selfplay:
                    mode = "avg" if use_avg_policy else "current"
                else:
                    # Team 0 uses policy, team 1 uses random
                    mode = ("avg" if use_avg_policy else "current") if (seat % 2 == 0) else "random"
                a = self._select_action_eval(st, seat, mode, policy_map, rng)
                st, _ = np_step(st, a)

            cat = self._category_breakdown(st)
            sums = {k: sums[k] + float(cat[k]) for k in sums}
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

    def train(self, iterations: int = 1000, seed: int = 42, verbose: bool = False, log_every: int = 100,
              eval_every: Optional[int] = None, eval_episodes: int = 32, eval_use_avg_policy: bool = True):
        # Independent RNG per iteration for deck
        for it in trange(1, iterations + 1, desc="Iter"):
            rng = Generator(PCG64(seed + it))
            st = np_init_state(rng)

            util0 = None
            for target in range(4):
                util = self._mccfr(st, target)
                if target == 0:
                    util0 = util
                if verbose and it % max(log_every, 1) == 0 and target == 0:
                    print(f"Iter {it}: util seat0 ~ {util:.3f}")

            if it % max(log_every, 1) == 0:
                self._log_metrics(it, util0)
                # Optional evaluation logging
                do_eval = (eval_every is not None and eval_every > 0 and (it % eval_every == 0))
                if do_eval and self.tlogger is not None:
                    # Self-play evaluation
                    selfplay_metrics = self.evaluate(episodes=eval_episodes, seed=seed + 7777 + it,
                                                     selfplay=True, use_avg_policy=eval_use_avg_policy)
                    for k, v in selfplay_metrics.items():
                        self.tlogger.writer.add_scalar(f"EvalSelf/{k}", float(v), it)
                    # Versus random evaluation (policy team on seats 0/2)
                    vsrnd_metrics = self.evaluate(episodes=eval_episodes, seed=seed + 8888 + it,
                                                  selfplay=False, use_avg_policy=eval_use_avg_policy)
                    for k, v in vsrnd_metrics.items():
                        self.tlogger.writer.add_scalar(f"EvalVsRandom/{k}", float(v), it)

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
        infoset_key = (seat, self._obs_key(obs))
        legal = (obs[0] > 0).astype(np.int32)
        if infoset_key not in self.cum_regret:
            self.cum_regret[infoset_key] = np.zeros((NUM_CARDS,), dtype=self.dtype)
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
            payload = {"type": "avg_policy", "policy": policy}
        elif kind == "full":
            payload = {
                "type": "full",
                "cum_regret": self.cum_regret,
                "cum_strategy": self.cum_strategy,
                "branch_topk": self.branch_topk,
                "max_infosets": self.max_infosets,
                "obs_key_mode": self.obs_key_mode,
                "dtype": str(self.dtype.name),
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
        elif payload.get("type") == "full":
            tmp = CFRTrainer(seed=seed)
            tmp.cum_strategy = payload["cum_strategy"]
            policy = tmp.get_average_policy()
        else:
            raise ValueError("Unsupported payload for avg policy load")
        return SavedPolicy(policy_map=policy, seed=seed)

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
                        dtype=np.dtype(payload.get("dtype", "float16")))
        tr.cum_regret = payload["cum_regret"]
        tr.cum_strategy = payload["cum_strategy"]
        return tr


class SavedPolicy:
    """Lightweight actor for a saved average policy map."""
    def __init__(self, policy_map: Dict[Tuple[int, bytes], np.ndarray], seed: int = 0):
        self.policy_map = policy_map
        self.rng = Generator(PCG64(seed))

    def act(self, seat: int, obs: np.ndarray) -> int:
        legal = (obs[0] > 0).astype(np.int32)
        legal_idx = np.nonzero(legal)[0]
        if legal_idx.size == 0:
            return 0
        # Use same compact keying as trainer
        key = (seat, hashlib.blake2b(obs.view(np.uint8), digest_size=16).digest())
        probs = self.policy_map.get(key)
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

    def act_from_obs(self, seat: int, obs: np.ndarray) -> int:
        return self.act(seat, obs)
