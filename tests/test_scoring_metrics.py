from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cfr_jax import CFRTrainer, NState, np_round_scores


def test_round_scores_scopa_distribution():
    state = NState()
    state.scopas[0] = 1
    state.scopas[2] = 1

    team_points, player_points = np_round_scores(state)

    np.testing.assert_allclose(team_points, np.array([2.0, 0.0]))
    np.testing.assert_allclose(player_points, np.array([1.0, 0.0, 1.0, 0.0]))


def test_reward_modes_change_utilities():
    state = NState()
    state.scopas[0] = 2

    team_trainer = CFRTrainer(seed=0, reward_mode="team")
    selfish_trainer = CFRTrainer(seed=0, reward_mode="selfish")

    team_utils = [team_trainer._evaluate_utility(state, seat) for seat in range(4)]
    selfish_utils = [selfish_trainer._evaluate_utility(state, seat) for seat in range(4)]

    assert team_utils == [1.0, -1.0, 1.0, -1.0]
    assert selfish_utils == [2.0, -1.0, 0.0, -1.0]


def test_evaluate_reports_collaboration_metrics():
    trainer = CFRTrainer(seed=3)
    metrics = trainer.evaluate(episodes=1, seed=7)

    for key in [
        "avg_player_points0",
        "avg_player_points1",
        "avg_player_points2",
        "avg_player_points3",
        "avg_team_points0",
        "avg_team_points1",
        "avg_inter_team_delta",
        "avg_intra_team_delta",
    ]:
        assert key in metrics, f"missing metric: {key}"
