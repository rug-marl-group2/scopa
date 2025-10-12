from pathlib import Path
import sys

import numpy as np
import pytest

try:
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if exc.name == "jax":
        pytest.skip("jax is required for JAX environment tests", allow_module_level=True)
    raise

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from jax_env import State, step, evaluate_round
from jax_cards import card_index, CARD_SUITS
from cfr_jax import NState, np_evaluate_round


def make_empty_state():
    return State(
        hands=jnp.zeros((4, 40), dtype=jnp.int8),
        table=jnp.zeros((40,), dtype=jnp.int8),
        captures=jnp.zeros((4, 40), dtype=jnp.int8),
        history=jnp.zeros((4, 40), dtype=jnp.int8),
        scopas=jnp.zeros((4,), dtype=jnp.int32),
        cur_player=jnp.array(0, dtype=jnp.int32),
        last_capture_player=jnp.array(-1, dtype=jnp.int32),
    )


def test_ace_on_empty_table_is_not_scopa():
    ace_idx = card_index(1, 0)
    state = make_empty_state()
    state = state._replace(hands=state.hands.at[0, ace_idx].set(1))
    next_state, scopa = step(state, jnp.array(ace_idx))

    assert int(scopa) == 0
    assert int(next_state.scopas[0]) == 0


def test_evaluate_round_awards_remaining_table_to_last_captor():
    sette_bello = card_index(7, 3)
    state = make_empty_state()
    # Table still has sette bello and player 1 was the last to capture
    state = state._replace(
        table=state.table.at[sette_bello].set(1),
        last_capture_player=jnp.array(1, dtype=jnp.int32),
    )

    team0_reward, team1_reward = evaluate_round(state)

    assert int(team0_reward) == -1
    assert int(team1_reward) == 1

def test_last_captor_gets_remaining_cards_and_coins():
    state = make_empty_state()

    team0_indices = jnp.array([
        card_index(7, 0),
        card_index(6, 0),
        card_index(5, 0),
        card_index(4, 0),
        card_index(3, 0),
    ])
    team1_indices = jnp.array([
        card_index(2, 1),
        card_index(3, 1),
        card_index(4, 1),
        card_index(5, 1),
    ])
    table_indices = jnp.array([
        card_index(2, 3),
        card_index(3, 3),
        card_index(4, 3),
    ])

    captures = state.captures
    captures = captures.at[0, team0_indices].set(1)
    captures = captures.at[1, team1_indices].set(1)
    table = state.table.at[table_indices].set(1)

    state = state._replace(
        captures=captures,
        table=table,
        last_capture_player=jnp.array(1, dtype=jnp.int32),
    )

    team0_reward, team1_reward = evaluate_round(state)
    assert int(team0_reward) == -1
    assert int(team1_reward) == 1

    captures_awarded = captures
    captures_awarded = captures_awarded.at[1].set(
        jnp.clip(captures_awarded[1] + state.table, 0, 1)
    )

    team0_mask = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
    team1_mask = jnp.array([0, 1, 0, 1], dtype=jnp.int32)

    team0_caps = (captures_awarded.T @ team0_mask).astype(jnp.int32)
    team1_caps = (captures_awarded.T @ team1_mask).astype(jnp.int32)

    team0_cards = int(jnp.sum(team0_caps))
    team1_cards = int(jnp.sum(team1_caps))

    is_bello = (CARD_SUITS == 3).astype(jnp.int32)
    team0_coins = int(jnp.sum(team0_caps * is_bello))
    team1_coins = int(jnp.sum(team1_caps * is_bello))

    assert team1_cards > team0_cards
    assert team1_coins > team0_coins

    np_state = NState()
    np_state.captures[0, team0_indices.tolist()] = 1
    np_state.captures[1, team1_indices.tolist()] = 1
    np_state.table[table_indices.tolist()] = 1
    np_state.last_capture_player = np.int32(1)

    np_team0, np_team1 = np_evaluate_round(np_state)
    assert np_team0 == -1
    assert np_team1 == 1