"""
CFR Algorithms

This package contains implementations of Counterfactual Regret Minimization
algorithms for Scopa.
"""

from .vanilla_cfr import CFRTrainer, InfoNode, LearnedCFRPolicy, RandomPolicy
from .mc_cfr import MCCFRTrainer, ScopaLearnedPolicy

__all__ = [
    'CFRTrainer',
    'InfoNode',
    'LearnedCFRPolicy',
    'RandomPolicy',
    'MCCFRTrainer',
    'ScopaLearnedPolicy',
]

