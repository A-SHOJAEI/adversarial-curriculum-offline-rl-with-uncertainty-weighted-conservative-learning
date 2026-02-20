"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.loader import (
    D4RLDataset,
    create_ood_states,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.preprocessing import (
    compute_dataset_statistics,
    normalize_observations,
    normalize_rewards,
    split_dataset,
)


def test_d4rl_dataset_creation(dummy_dataset):
    """Test D4RL dataset creation."""
    assert len(dummy_dataset) == 1000

    sample = dummy_dataset[0]
    assert "observations" in sample
    assert "actions" in sample
    assert "rewards" in sample
    assert "next_observations" in sample
    assert "terminals" in sample


def test_d4rl_dataset_indexing(dummy_dataset, state_dim, action_dim):
    """Test dataset indexing."""
    sample = dummy_dataset[0]

    assert sample["observations"].shape == (state_dim,)
    assert sample["actions"].shape == (action_dim,)
    assert sample["rewards"].shape == (1,)
    assert sample["next_observations"].shape == (state_dim,)
    assert sample["terminals"].shape == (1,)


def test_normalize_observations(state_dim):
    """Test observation normalization."""
    observations = np.random.randn(100, state_dim)
    mean = observations.mean(axis=0)
    std = observations.std(axis=0) + 1e-8

    normalized = normalize_observations(observations, mean, std)

    assert normalized.shape == observations.shape
    assert np.abs(normalized.mean(axis=0)).max() < 0.1
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.1


def test_normalize_rewards():
    """Test reward normalization."""
    rewards = np.random.randn(100)
    mean = rewards.mean()
    std = rewards.std() + 1e-8

    normalized = normalize_rewards(rewards, mean, std)

    assert normalized.shape == rewards.shape
    assert abs(normalized.mean()) < 0.1
    assert abs(normalized.std() - 1.0) < 0.1


def test_compute_dataset_statistics(state_dim, action_dim):
    """Test dataset statistics computation."""
    observations = np.random.randn(100, state_dim)
    actions = np.random.randn(100, action_dim)
    rewards = np.random.randn(100)

    stats = compute_dataset_statistics(observations, actions, rewards)

    assert "obs_mean" in stats
    assert "obs_std" in stats
    assert "action_mean" in stats
    assert "action_std" in stats
    assert "reward_mean" in stats
    assert "reward_std" in stats

    assert stats["obs_mean"].shape == (state_dim,)
    assert stats["obs_std"].shape == (state_dim,)


def test_split_dataset():
    """Test dataset splitting."""
    dataset_size = 1000
    train_indices, val_indices, test_indices = split_dataset(
        dataset_size, train_ratio=0.8, val_ratio=0.1, seed=42
    )

    assert len(train_indices) == 800
    assert len(val_indices) == 100
    assert len(test_indices) == 100

    # Check no overlap
    all_indices = set(train_indices) | set(val_indices) | set(test_indices)
    assert len(all_indices) == dataset_size


def test_create_ood_states(dummy_dataset):
    """Test OOD state creation."""
    num_samples = 50
    ood_states = create_ood_states(dummy_dataset, num_samples, noise_scale=0.5)

    assert ood_states.shape[0] == num_samples
    assert ood_states.shape[1] == dummy_dataset.observations.shape[1]
    assert isinstance(ood_states, torch.Tensor)
