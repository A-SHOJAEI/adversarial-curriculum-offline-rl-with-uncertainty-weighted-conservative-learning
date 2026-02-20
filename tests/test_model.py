"""Tests for model components."""

import pytest
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.components import (
    CQLLoss,
    UncertaintyWeightedCurriculumScheduler,
    compute_ensemble_uncertainty,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import (
    Actor,
    Critic,
    EnsembleCritic,
    CQLAgent,
)


def test_actor_forward(state_dim, action_dim, device):
    """Test actor forward pass."""
    actor = Actor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
    )
    actor.to(device)

    batch_size = 32
    states = torch.randn(batch_size, state_dim).to(device)

    actions = actor(states)

    assert actions.shape == (batch_size, action_dim)
    assert torch.all(torch.abs(actions) <= 1.0)


def test_critic_forward(state_dim, action_dim, device):
    """Test critic forward pass."""
    critic = Critic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
    )
    critic.to(device)

    batch_size = 32
    states = torch.randn(batch_size, state_dim).to(device)
    actions = torch.randn(batch_size, action_dim).to(device)

    q_values = critic(states, actions)

    assert q_values.shape == (batch_size, 1)


def test_ensemble_critic_forward(state_dim, action_dim, device):
    """Test ensemble critic forward pass."""
    ensemble_size = 5
    ensemble_critic = EnsembleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        ensemble_size=ensemble_size,
    )
    ensemble_critic.to(device)

    batch_size = 32
    states = torch.randn(batch_size, state_dim).to(device)
    actions = torch.randn(batch_size, action_dim).to(device)

    # Test mean output
    q_mean = ensemble_critic(states, actions, return_all=False)
    assert q_mean.shape == (batch_size, 1)

    # Test all outputs
    q_all = ensemble_critic(states, actions, return_all=True)
    assert q_all.shape == (ensemble_size, batch_size, 1)


def test_ensemble_uncertainty(state_dim, action_dim, device):
    """Test ensemble uncertainty computation."""
    ensemble_critic = EnsembleCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64, 64],
        ensemble_size=5,
    )
    ensemble_critic.to(device)

    batch_size = 32
    states = torch.randn(batch_size, state_dim).to(device)
    actions = torch.randn(batch_size, action_dim).to(device)

    mean_q, uncertainty = ensemble_critic.get_uncertainty(states, actions)

    assert mean_q.shape == (batch_size, 1)
    assert uncertainty.shape == (batch_size, 1)
    assert torch.all(uncertainty >= 0)


def test_cql_agent_initialization(cql_agent, state_dim, action_dim):
    """Test CQL agent initialization."""
    assert cql_agent.state_dim == state_dim
    assert cql_agent.action_dim == action_dim
    assert hasattr(cql_agent, "actor")
    assert hasattr(cql_agent, "critic")
    assert hasattr(cql_agent, "target_actor")
    assert hasattr(cql_agent, "target_critic")


def test_cql_agent_select_action(cql_agent, state_dim, action_dim, device):
    """Test action selection."""
    batch_size = 32
    states = torch.randn(batch_size, state_dim).to(device)

    # Deterministic action
    action_det = cql_agent.select_action(states, deterministic=True)
    assert action_det.shape == (batch_size, action_dim)

    # Stochastic action
    action_stoch = cql_agent.select_action(states, deterministic=False)
    assert action_stoch.shape == (batch_size, action_dim)


def test_cql_agent_update_targets(cql_agent):
    """Test target network update."""
    # Get initial target parameters
    initial_target_params = [
        p.clone() for p in cql_agent.target_critic.parameters()
    ]

    # Update main network
    for p in cql_agent.critic.parameters():
        p.data += 1.0

    # Update targets
    cql_agent.update_targets(tau=0.1)

    # Check targets changed
    for initial, current in zip(
        initial_target_params,
        cql_agent.target_critic.parameters(),
    ):
        assert not torch.allclose(initial, current)


def test_cql_loss():
    """Test CQL loss computation."""
    batch_size = 32
    cql_loss = CQLLoss(initial_alpha=5.0, temperature=1.0, learnable_alpha=False)

    q_values = torch.randn(batch_size, 1)
    target_q = torch.randn(batch_size, 1)
    random_q = torch.randn(batch_size, 5)  # Multiple random actions
    policy_q = torch.randn(batch_size, 5)  # Policy actions

    total_loss, td_loss, conservative_loss, alpha_loss = cql_loss(
        q_values, target_q, random_q, policy_q, curriculum_weight=1.0
    )

    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(td_loss, torch.Tensor)
    assert isinstance(conservative_loss, torch.Tensor)
    assert isinstance(alpha_loss, torch.Tensor)
    assert total_loss.ndim == 0  # Scalar


def test_curriculum_scheduler():
    """Test curriculum scheduler."""
    scheduler = UncertaintyWeightedCurriculumScheduler(
        initial_conservatism=10.0,
        min_conservatism=1.0,
        warmup_steps=100,
        uncertainty_threshold=0.5,
    )

    # Test warmup
    for _ in range(50):
        weight = scheduler.step(mean_uncertainty=0.3)
        assert 1.0 <= weight <= 10.0

    # Test after warmup with low uncertainty
    for _ in range(20):
        weight = scheduler.step(mean_uncertainty=0.2)

    assert weight < scheduler.initial_conservatism


def test_compute_ensemble_uncertainty():
    """Test ensemble uncertainty computation."""
    batch_size = 32
    ensemble_size = 5

    q_values_list = [
        torch.randn(batch_size, 1) for _ in range(ensemble_size)
    ]

    uncertainty = compute_ensemble_uncertainty(q_values_list)

    assert uncertainty.shape == (batch_size, 1)
    assert torch.all(uncertainty >= 0)


def test_compute_ensemble_uncertainty_empty():
    """Test ensemble uncertainty with empty list."""
    with pytest.raises(ValueError, match="Empty Q-values list"):
        compute_ensemble_uncertainty([])
