"""
Scopa Game Environments

This package contains implementations of Scopa card game variants
for reinforcement learning research.
"""

from .mini_scopa_game import MiniScopaGame, MiniScopaEnv
from .full_scopa_game import FullScopaGame, FullScopaEnv
from .team_mini_scopa_game import TeamMiniScopaGame, TeamMiniScopaEnv

__all__ = [
    'MiniScopaGame',
    'MiniScopaEnv',
    'FullScopaGame',
    'FullScopaEnv',
    'TeamMiniScopaGame',
    'TeamMiniScopaEnv',
]

