from typing import Callable, NamedTuple, Tuple

import jax
from jax import lax
import jax.numpy as jnp

from jax_cards import (
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    CARD_RANKS,
    CARD_SUITS,
    RANK_PRIORITY,
    card_index,
    split_teams,
)

class State(NamedTuple):
    hands: jnp.ndarray         # int8 [4, 40]
    table: jnp.ndarray         # int8 [40]
    captures: jnp.ndarray      # int8 [4, 40]
    history: jnp.ndarray       # int8 [4, 40]
    scopas: jnp.ndarray        # int32 [4]
    cur_player: jnp.ndarray    # int32 scalar
    last_capture_player: jnp.ndarray  # int32 scalar (-1 if none)

def _one_hot(indices: jnp.ndarray, size: int, dtype=jnp.int8) -> jnp.ndarray:
    return (indices[..., None] == jnp.arange(size)[None, ...]).astype(dtype)

def init_state(key: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
    key, kperm, kstart = jax.random.split(key, 3)
    perm = jax.random.permutation(kperm, NUM_CARDS)
    hands = jnp.zeros((4, NUM_CARDS), dtype=jnp.int8)
    for seat in range(4):
        start = seat * 10
        end = start + 10
        idx = perm[start:end]
        hands = hands.at[seat].set(jnp.zeros((NUM_CARDS,), dtype=jnp.int8).at[idx].set(1))

    table = jnp.zeros((NUM_CARDS,), dtype=jnp.int8)
    captures = jnp.zeros((4, NUM_CARDS), dtype=jnp.int8)
    history = jnp.zeros((4, NUM_CARDS), dtype=jnp.int8)
    scopas = jnp.zeros((4,), dtype=jnp.int32)
    cur_player = jax.random.randint(kstart, (), 0, 4, dtype=jnp.int32)
    last_capture_player = jnp.array(-1, dtype=jnp.int32)
    return key, State(hands, table, captures, history, scopas, cur_player, last_capture_player)

@jax.jit
def legal_action_mask(state: State) -> jnp.ndarray:
    return state.hands[state.cur_player]

def _subset_sum_capture_mask(table: jnp.ndarray, target_rank: jnp.ndarray) -> jnp.ndarray:
    dp_has = jnp.zeros((NUM_RANKS + 1,), dtype=jnp.bool_).at[0].set(True)
    dp_mask = jnp.zeros((NUM_RANKS + 1, NUM_CARDS), dtype=jnp.bool_)

    def body_card(carry, i):
        dp_has, dp_mask = carry
        present = table[i] == 1
        r = CARD_RANKS[i]

        def update_for_card(carry):
            dp_has, dp_mask = carry

            def body_sum(carry2, s_rev):
                dp_has, dp_mask = carry2
                s = NUM_RANKS - s_rev
                cond_upd = (s >= r) & (~dp_has[s]) & dp_has[s - r]
                base_row = dp_mask[s]
                new_row = jnp.where(cond_upd, dp_mask[s - r].at[i].set(True), base_row)
                dp_mask = dp_mask.at[s].set(new_row)
                dp_has = dp_has.at[s].set(jnp.where(cond_upd, True, dp_has[s]))
                return (dp_has, dp_mask), None

            (dp_has, dp_mask), _ = lax.scan(body_sum, (dp_has, dp_mask), jnp.arange(NUM_RANKS + 1))
            return dp_has, dp_mask

        dp_has, dp_mask = lax.cond(present, update_for_card, lambda c: c, (dp_has, dp_mask))
        return (dp_has, dp_mask), None

    (dp_has, dp_mask), _ = lax.scan(body_card, (dp_has, dp_mask), jnp.arange(NUM_CARDS))
    has = dp_has[target_rank]
    mask = dp_mask[target_rank]
    return jnp.where(has, mask, jnp.zeros_like(mask))

def _single_match_mask(table: jnp.ndarray, target_rank: jnp.ndarray) -> jnp.ndarray:
    """Mask with True at indices where a single equal-rank capture is possible."""
    return (table.astype(jnp.bool_) & (CARD_RANKS == target_rank))

def _best_capture_mask(table: jnp.ndarray, target_rank: jnp.ndarray) -> jnp.ndarray:
    """Priority rule: prefer single equal-rank, else subset-sum."""
    sm = _single_match_mask(table, target_rank)
    has_single = jnp.any(sm)
    subset = _subset_sum_capture_mask(table, target_rank)
    return jnp.where(has_single, sm, subset)

def _apply_capture(state: State, seat: jnp.ndarray, subset_mask: jnp.ndarray, include_action: bool, action_idx: jnp.ndarray) -> Tuple[State, jnp.ndarray]:
    captured_mask = subset_mask.astype(jnp.int8)
    if include_action:
        captured_mask = captured_mask.at[action_idx].set(1)
    table_after = state.table * (1 - subset_mask.astype(jnp.int8))
    captured_any = jnp.any(subset_mask)
    captures_after = state.captures.at[seat].set(state.captures[seat] + captured_mask)
    scopa_happened = (jnp.sum(table_after) == 0) & captured_any
    scopas_after = state.scopas.at[seat].set(state.scopas[seat] + scopa_happened.astype(jnp.int32))
    state2 = State(
        state.hands,
        table_after,
        captures_after,
        state.history,
        scopas_after,
        state.cur_player,
        seat,  # last_capture_player
    )
    return state2, scopa_happened

@jax.jit
def step(state: State, action_idx: jnp.ndarray) -> Tuple[State, jnp.ndarray]:
    seat = state.cur_player
    hands = state.hands.at[seat, action_idx].set(0)
    history = state.history.at[seat, action_idx].set(1)
    state = State(hands, state.table, state.captures, history, state.scopas, state.cur_player, state.last_capture_player)

    rank = CARD_RANKS[action_idx]
    is_ace = rank == 1

    def do_ace(st: State) -> Tuple[State, jnp.ndarray]:
        subset_mask = st.table.astype(jnp.bool_)
        return _apply_capture(st, seat, subset_mask, include_action=True, action_idx=action_idx)

    def do_non_ace(st: State) -> Tuple[State, jnp.ndarray]:
        subset_mask = _best_capture_mask(st.table, rank)  # single-over-subset priority
        has_capture = jnp.any(subset_mask)

        def do_capture(xst: State):
            return _apply_capture(xst, seat, subset_mask, include_action=True, action_idx=action_idx)

        def do_place(xst: State):
            table_after = xst.table.at[action_idx].set(1)
            xst2 = State(xst.hands, table_after, xst.captures, xst.history, xst.scopas, xst.cur_player, xst.last_capture_player)
            return xst2, jnp.array(False)

        return lax.cond(has_capture, do_capture, do_place, st)

    state, scopa = lax.cond(is_ace, do_ace, do_non_ace, state)

    # If this move ends the hand, DO NOT count a scopa made on the last move.
    # (Undo the increment if any.)
    remaining_cards = jnp.sum(state.hands)
    def undo_last_scopa(st):
        # decrement scopas for current seat if we just counted one
        new_scopas = st.scopas.at[seat].add(jnp.where(scopa, -1, 0))
        return State(st.hands, st.table, st.captures, st.history, new_scopas, st.cur_player, st.last_capture_player)
    state = lax.cond((remaining_cards == 0) & scopa, undo_last_scopa, lambda x: x, state)

    cur_player = (seat + 1) % 4
    state = State(state.hands, state.table, state.captures, state.history, state.scopas, jnp.array(cur_player, dtype=jnp.int32), state.last_capture_player)
    return state, scopa

@jax.jit
def is_terminal(state: State) -> jnp.ndarray:
    return jnp.sum(state.hands) == 0

@jax.jit
def build_observation(state: State, seat: jnp.ndarray) -> jnp.ndarray:
    friend = (seat + 2) % 4
    enemy1 = (seat + 1) % 4
    enemy2 = (seat + 3) % 4

    out = jnp.zeros((6, NUM_CARDS), dtype=jnp.float32)
    out = out.at[0].set(state.hands[seat].astype(jnp.float32))
    out = out.at[1].set(state.table.astype(jnp.float32))
    out = out.at[2].set((state.captures[seat] + state.captures[friend]).clip(max=1).astype(jnp.float32))
    out = out.at[3].set(state.history[enemy1].astype(jnp.float32))
    out = out.at[4].set(state.history[enemy2].astype(jnp.float32))
    out = out.at[5].set(state.history[friend].astype(jnp.float32))
    return out

@jax.jit
def evaluate_round(state: State) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Award remaining table cards to last capturer (if any)
    def _award_table(captures):
        extra = state.table.astype(jnp.int8)
        updated = jnp.clip(captures[state.last_capture_player] + extra, 0, 1)
        return captures.at[state.last_capture_player].set(updated)

    captures = lax.cond(
        state.last_capture_player >= 0,
        _award_table,
        lambda caps: caps,
        state.captures,
    )

    team0_mask, team1_mask = split_teams()
    team0_caps = (captures.T @ team0_mask).astype(jnp.int32)
    team1_caps = (captures.T @ team1_mask).astype(jnp.int32)

    team0_points = jnp.array(0, dtype=jnp.int32)
    team1_points = jnp.array(0, dtype=jnp.int32)

    # Scopas (already adjusted to exclude last-move scopa in step)
    team0_points += (state.scopas @ team0_mask)
    team1_points += (state.scopas @ team1_mask)

    # Most cards
    t0_cards = jnp.sum(team0_caps)
    t1_cards = jnp.sum(team1_caps)
    team0_points += (t0_cards > t1_cards).astype(jnp.int32)
    team1_points += (t1_cards > t0_cards).astype(jnp.int32)

    # Most coins (denari/bello -> suit_id 3)
    is_bello = (CARD_SUITS == 3).astype(jnp.int32)
    t0_coins = jnp.sum(team0_caps * is_bello)
    t1_coins = jnp.sum(team1_caps * is_bello)
    team0_points += (t0_coins > t1_coins).astype(jnp.int32)
    team1_points += (t1_coins > t0_coins).astype(jnp.int32)

    # Settebello
    sette_bello_idx = card_index(7, 3)
    team0_points += (team0_caps[sette_bello_idx] > 0).astype(jnp.int32)
    team1_points += (team1_caps[sette_bello_idx] > 0).astype(jnp.int32)

    # Primiera
    by_suit_rank0 = jnp.reshape(team0_caps, (NUM_SUITS, NUM_RANKS))
    by_suit_rank1 = jnp.reshape(team1_caps, (NUM_SUITS, NUM_RANKS))
    pri = RANK_PRIORITY[1:]  # [10]
    prim0 = jnp.sum(jnp.max(by_suit_rank0 * pri, axis=1))
    prim1 = jnp.sum(jnp.max(by_suit_rank1 * pri, axis=1))
    team0_points += (prim0 > prim1).astype(jnp.int32)
    team1_points += (prim1 > prim0).astype(jnp.int32)

    t0_reward = jnp.where(team0_points > team1_points, 1, jnp.where(team1_points > team0_points, -1, 0))
    t1_reward = -t0_reward
    return t0_reward, t1_reward

# play_round_scan, play_rounds_batched unchanged
