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
    # All arrays are JAX arrays; shapes are fixed for JIT.
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
    """Initialize a new game state with a shuffled deal (10 cards per player).

    Returns (key', state).
    """
    key, kperm, kstart = jax.random.split(key, 3)
    perm = jax.random.permutation(kperm, NUM_CARDS)
    # Deal 10 cards per player from permutation
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
    """Compute a boolean mask over table cards that sum to target_rank.

    Uses DP in O(40 * 10). Returns mask[40] with True for selected subset;
    returns all-False if none exists.
    """
    # dp_has[s] -> reachable sum s
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
                s = NUM_RANKS - s_rev  # 10..0 descending
                cond_upd = (s >= r) & (~dp_has[s]) & dp_has[s - r]
                base_row = dp_mask[s]
                new_row = jnp.where(
                    cond_upd,
                    dp_mask[s - r].at[i].set(True),
                    base_row,
                )
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


def _apply_capture(state: State, seat: jnp.ndarray, subset_mask: jnp.ndarray, include_action: bool, action_idx: jnp.ndarray) -> Tuple[State, jnp.ndarray]:
    # Move subset from table to captures; optionally include the played card itself.
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
        seat,
    )
    return state2, scopa_happened


@jax.jit
def step(state: State, action_idx: jnp.ndarray) -> Tuple[State, jnp.ndarray]:
    seat = state.cur_player
    # Remove from hand, mark history
    hands = state.hands.at[seat, action_idx].set(0)
    history = state.history.at[seat, action_idx].set(1)
    state = State(hands, state.table, state.captures, history, state.scopas, state.cur_player, state.last_capture_player)

    rank = CARD_RANKS[action_idx]
    is_ace = rank == 1

    def do_ace(st: State) -> Tuple[State, jnp.ndarray]:
        # Capture entire table + played card
        subset_mask = st.table.astype(jnp.bool_)
        return _apply_capture(st, seat, subset_mask, include_action=True, action_idx=action_idx)

    def do_non_ace(st: State) -> Tuple[State, jnp.ndarray]:
        subset_mask = _subset_sum_capture_mask(st.table, rank)
        has_capture = jnp.any(subset_mask)

        def do_capture(xst: State):
            return _apply_capture(xst, seat, subset_mask, include_action=True, action_idx=action_idx)

        def do_place(xst: State):
            table_after = xst.table.at[action_idx].set(1)
            xst2 = State(xst.hands, table_after, xst.captures, xst.history, xst.scopas, xst.cur_player, xst.last_capture_player)
            return xst2, jnp.array(False)

        return lax.cond(has_capture, do_capture, do_place, st)

    state, scopa = lax.cond(is_ace, do_ace, do_non_ace, state)

    # Next seat
    cur_player = (seat + 1) % 4
    state = State(state.hands, state.table, state.captures, state.history, state.scopas, jnp.array(cur_player, dtype=jnp.int32), state.last_capture_player)
    return state, scopa


@jax.jit
def is_terminal(state: State) -> jnp.ndarray:
    return jnp.sum(state.hands) == 0


@jax.jit
def build_observation(state: State, seat: jnp.ndarray) -> jnp.ndarray:
    # 4x40 planes: hand, table, captures(me+friend), history(friend)
    friend = (seat + 2) % 4

    out = jnp.zeros((4, NUM_CARDS), dtype=jnp.float32)
    out = out.at[0].set(state.hands[seat].astype(jnp.float32))
    out = out.at[1].set(state.table.astype(jnp.float32))
    captures = (state.captures[seat] + state.captures[friend]).clip(max=1)
    out = out.at[2].set(captures.astype(jnp.float32))
    out = out.at[3].set(state.history[friend].astype(jnp.float32))
    return out


@jax.jit
def evaluate_round(state: State) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Remaining table cards go to the last player who captured (if any)
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

    # Team captures
    team0_mask, team1_mask = split_teams()
    team0_caps = (captures.T @ team0_mask).astype(jnp.int32)  # [40]
    team1_caps = (captures.T @ team1_mask).astype(jnp.int32)

    team0_points = jnp.array(0, dtype=jnp.int32)
    team1_points = jnp.array(0, dtype=jnp.int32)

    # Scopas
    team0_points += (state.scopas @ team0_mask)
    team1_points += (state.scopas @ team1_mask)

    # Most cards
    t0_cards = jnp.sum(team0_caps)
    t1_cards = jnp.sum(team1_caps)
    team0_points += (t0_cards > t1_cards).astype(jnp.int32)
    team1_points += (t1_cards > t0_cards).astype(jnp.int32)

    # Most coins (suit == bello -> suit_id 3)
    is_bello = (CARD_SUITS == 3).astype(jnp.int32)
    t0_coins = jnp.sum(team0_caps * is_bello)
    t1_coins = jnp.sum(team1_caps * is_bello)
    team0_points += (t0_coins > t1_coins).astype(jnp.int32)
    team1_points += (t1_coins > t0_coins).astype(jnp.int32)

    # Sette bello (rank==7 and suit==bello)
    sette_bello_idx = card_index(7, 3)
    team0_points += (team0_caps[sette_bello_idx] > 0).astype(jnp.int32)
    team1_points += (team1_caps[sette_bello_idx] > 0).astype(jnp.int32)

    # Primiera
    # For each suit, compute max priority among captured ranks; sum over suits
    # Build [4,10] presence by suit/rank for each team
    # team_caps_by_suit_rank[suit, rank-1] = presence
    by_suit_rank0 = jnp.reshape(team0_caps, (NUM_SUITS, NUM_RANKS))
    by_suit_rank1 = jnp.reshape(team1_caps, (NUM_SUITS, NUM_RANKS))
    # Map rank presence to priorities and take max per suit
    pri = RANK_PRIORITY[1:]  # [10]
    prim0 = jnp.sum(jnp.max(by_suit_rank0 * pri, axis=1))
    prim1 = jnp.sum(jnp.max(by_suit_rank1 * pri, axis=1))
    team0_points += (prim0 > prim1).astype(jnp.int32)
    team1_points += (prim1 > prim0).astype(jnp.int32)

    # Final reward: 1 or -1 or 0 like env
    t0_reward = jnp.where(team0_points > team1_points, 1, jnp.where(team1_points > team0_points, -1, 0))
    t1_reward = -t0_reward
    return t0_reward, t1_reward


def _mask_and_normalize(probs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    p = probs * mask
    s = jnp.sum(p)
    # If no mass, fallback to uniform over legal
    uni = mask / jnp.maximum(jnp.sum(mask), 1)
    return jnp.where(s > 0, p / s, uni)


def uniform_policy(_params, _key, _state: State, seat: jnp.ndarray) -> jnp.ndarray:
    # Uniform over legal actions
    mask = legal_action_mask(_state).astype(jnp.float32)
    return _mask_and_normalize(jnp.ones((NUM_CARDS,), dtype=jnp.float32), mask)


def play_round_scan(key: jnp.ndarray,
                    params,
                    policy_apply: Callable[[object, jnp.ndarray, State, jnp.ndarray], jnp.ndarray]
                    ) -> Tuple[jnp.ndarray, State, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Play a full round (40 steps) with policy_apply.

    Returns (key', final_state, (team0_reward, team1_reward)).
    """
    key, state_key = init_state(key)

    def body(carry, _):
        key, st = carry
        seat = st.cur_player
        key, ks = jax.random.split(key)
        probs = policy_apply(params, ks, st, seat)
        mask = legal_action_mask(st).astype(jnp.float32)
        probs = _mask_and_normalize(probs, mask)
        action = jax.random.choice(ks, NUM_CARDS, p=probs)
        st, _ = step(st, action)
        return (key, st), None

    (key, state), _ = lax.scan(body, (key, state_key), xs=None, length=NUM_CARDS)
    r0, r1 = evaluate_round(state)
    return key, state, (r0, r1)


play_round_scan = jax.jit(play_round_scan, static_argnums=(2,))


def play_rounds_batched(key: jnp.ndarray, params, policy_apply, batch_size: int) -> Tuple[jnp.ndarray, State, Tuple[jnp.ndarray, jnp.ndarray]]:
    keys = jax.random.split(key, batch_size)
    v_play = jax.vmap(play_round_scan, in_axes=(0, None, None))
    keys_out, states, rewards = v_play(keys, params, policy_apply)
    # Return aggregated key and the last state just for reference
    key_out = jax.random.fold_in(keys_out[0], jnp.sum(jnp.arange(batch_size)))
    r0, r1 = rewards
    return key_out, states, (r0, r1)

