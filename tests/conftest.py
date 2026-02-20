"""Pytest fixtures for testing."""

import numpy as np
import pytest
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.loader import D4RLDataset
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def state_dim():
    """State dimension."""
    return 17


@pytest.fixture
def action_dim():
    """Action dimension."""
    return 6


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 32


@pytest.fixture
def dummy_dataset(state_dim, action_dim):
    """Create dummy dataset for testing."""
    n_samples = 1000

    observations = np.random.randn(n_samples, state_dim).astype(np.float32)
    actions = np.random.randn(n_samples, action_dim).astype(np.float32)
    rewards = np.random.randn(n_samples).astype(np.float32)
    next_observations = np.random.randn(n_samples, state_dim).astype(np.float32)
    terminals = np.random.randint(0, 2, n_samples).astype(np.float32)

    return D4RLDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )


@pytest.fixture
def cql_agent(state_dim, action_dim, device):
    """Create CQL agent for testing."""
    agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        ensemble_size=3,
        activation="relu",
        max_action=1.0,
        dropout=0.0,
    )
    agent.to(device)
    return agent


@pytest.fixture
def sample_batch(batch_size, state_dim, action_dim):
    """Create sample batch for testing."""
    return {
        "observations": torch.randn(batch_size, state_dim),
        "actions": torch.randn(batch_size, action_dim),
        "rewards": torch.randn(batch_size, 1),
        "next_observations": torch.randn(batch_size, state_dim),
        "terminals": torch.randint(0, 2, (batch_size, 1)).float(),
    }


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "experiment": {
            "name": "test_experiment",
            "seed": 42,
            "device": "cpu",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "cql_alpha": 5.0,
            "cql_temperature": 1.0,
            "curriculum_enabled": True,
            "initial_conservatism": 10.0,
            "min_conservatism": 1.0,
            "uncertainty_threshold": 0.5,
            "grad_clip": 1.0,
            "early_stopping": False,
        },
    }
