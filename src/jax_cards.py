from typing import Tuple

import jax.numpy as jnp


# Suits order matches observation/index mapping used in env/cfr_jax:
# index = (rank-1) + suit_offset, where offsets are
# {'cuori': 0, 'picche': 10, 'fiori': 20, 'bello': 30}

# Suit IDs consistent with the above ordering
SUIT_ID = {
    "cuori": 0,
    "picche": 1,
    "fiori": 2,
    "bello": 3,
}

NUM_SUITS = 4
NUM_RANKS = 10  # 1..10
NUM_CARDS = NUM_SUITS * NUM_RANKS  # 40


def card_index(rank: int, suit_id: int) -> int:
    """Map (rank in 1..10, suit_id in 0..3) -> 0..39 index."""
    return suit_id * NUM_RANKS + (rank - 1)


# Precompute metadata arrays for index -> (rank, suit)
CARD_RANKS = jnp.concatenate(
    [jnp.arange(1, NUM_RANKS + 1, dtype=jnp.int32) for _ in range(NUM_SUITS)]
)
CARD_SUITS = jnp.concatenate(
    [jnp.full((NUM_RANKS,), si, dtype=jnp.int32) for si in range(NUM_SUITS)]
)


def split_teams() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return boolean masks for team0 (seats 0,2) and team1 (seats 1,3)."""
    team0 = jnp.array([1, 0, 1, 0], dtype=jnp.int32)
    team1 = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    return team0, team1


# Rank priority for Primiera as per env:
# {7: 4, 6: 3, 1: 2, 5: 1, 4: 0, 3: 0, 2: 0, 8: 0, 9: 0, 10: 0}
RANK_PRIORITY = jnp.array([0, 2, 0, 0, 0, 1, 3, 4, 0, 0, 0], dtype=jnp.int32)

