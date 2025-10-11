from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from jax_env import State, step, evaluate_round
from jax_cards import card_index


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
