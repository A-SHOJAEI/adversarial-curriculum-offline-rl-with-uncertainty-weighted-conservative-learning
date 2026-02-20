"""Adversarial Curriculum Offline RL with Uncertainty-Weighted Conservative Learning.

This package implements a novel approach to offline reinforcement learning that
combines Conservative Q-Learning (CQL) with an adversarial curriculum weighted by
epistemic uncertainty estimates from bootstrapped ensembles.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"
__license__ = "MIT"

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import (
    CQLAgent,
    EnsembleCritic,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.training.trainer import (
    AdversarialCurriculumTrainer,
)

__all__ = [
    "CQLAgent",
    "EnsembleCritic",
    "AdversarialCurriculumTrainer",
]
