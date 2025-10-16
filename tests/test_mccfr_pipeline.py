from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cfr_jax import (  # noqa: E402  # pylint: disable=wrong-import-position
    CFRTrainer,
    EvaluationSchedule,
    ExploitabilitySchedule,
    NState,
)


def make_trainer() -> CFRTrainer:
    return CFRTrainer(
        seed=123,
        branch_topk=None,
        progressive_widening=False,
        regret_prune=False,
        subset_cache_size=0,
        rollout_depth=1,
        rollout_samples=1,
    )


def make_single_card_state(action_idx: int = 0) -> NState:
    state = NState()
    state.hands.fill(0)
    state.hands[0, action_idx] = 1
    state.cur_player = np.int32(0)
    state.last_capture_player = np.int32(-1)
    state.table.fill(0)
    state.captures.fill(0)
    state.history.fill(0)
    return state


def test_policy_value_triplet_matches_recursive():
    trainer = make_trainer()
    state = make_single_card_state()

    triplet = trainer._policy_value_triplet(state, policy_map=None, use_avg_policy=True)
    base = trainer._policy_value_recursive(state, None, policy_map=None, use_avg_policy=True)
    br0 = trainer._policy_value_recursive(state, 0, policy_map=None, use_avg_policy=True)
    br1 = trainer._policy_value_recursive(state, 1, policy_map=None, use_avg_policy=True)

    assert triplet[0] == pytest.approx(base)
    assert triplet[1] == pytest.approx(br0)
    assert triplet[2] == pytest.approx(br1)


def test_schedule_utilities():
    eval_schedule = EvaluationSchedule(every=5, episodes=32, use_avg_policy=True)
    assert eval_schedule.is_enabled()
    assert not eval_schedule.should_run(3)
    assert eval_schedule.should_run(10)
    assert eval_schedule.resolved_episodes() == 32

    exploit_schedule = ExploitabilitySchedule(every=3, episodes=None, use_avg_policy=False)
    assert exploit_schedule.is_enabled()
    assert exploit_schedule.should_run(6)
    assert exploit_schedule.resolved_episodes(7) == 7 
