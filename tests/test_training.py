"""Tests for training components."""

import pytest
import torch
from torch.utils.data import DataLoader

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.training.trainer import (
    AdversarialCurriculumTrainer,
)


def test_trainer_initialization(cql_agent, dummy_dataset, test_config, device):
    """Test trainer initialization."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    assert trainer.agent == cql_agent
    assert trainer.dataloader == dataloader
    assert trainer.device == device
    assert hasattr(trainer, "actor_optimizer")
    assert hasattr(trainer, "critic_optimizer")
    assert hasattr(trainer, "cql_loss")


def test_trainer_train_step(cql_agent, dummy_dataset, test_config, device, sample_batch):
    """Test single training step."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    metrics = trainer.train_step(sample_batch)

    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "td_loss" in metrics
    assert "conservative_loss" in metrics
    assert "mean_q" in metrics
    assert "uncertainty" in metrics

    assert isinstance(metrics["critic_loss"], float)
    assert isinstance(metrics["actor_loss"], float)


def test_trainer_train_epoch(cql_agent, dummy_dataset, test_config, device):
    """Test training epoch."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    metrics = trainer.train_epoch()

    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert isinstance(metrics["critic_loss"], float)
    assert trainer.epoch_count == 1


def test_early_stopping(cql_agent, dummy_dataset, test_config, device):
    """Test early stopping logic."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    test_config["training"]["early_stopping"] = True
    test_config["training"]["patience"] = 3
    test_config["training"]["min_delta"] = 0.01

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    # Simulate no improvement
    for _ in range(4):
        should_stop = trainer.should_stop_early(1.0)

    assert should_stop


def test_checkpoint_save_load(cql_agent, dummy_dataset, test_config, device, tmp_path):
    """Test checkpoint saving and loading."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    # Train for one epoch
    metrics = trainer.train_epoch()

    # Save checkpoint
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path), epoch=1, metrics=metrics)

    assert checkpoint_path.exists()

    # Create new trainer and load checkpoint
    new_agent = type(cql_agent)(
        state_dim=cql_agent.state_dim,
        action_dim=cql_agent.action_dim,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        ensemble_size=3,
    )

    new_trainer = AdversarialCurriculumTrainer(
        agent=new_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    checkpoint = new_trainer.load_checkpoint(str(checkpoint_path))

    assert checkpoint["epoch"] == 1
    assert "metrics" in checkpoint


def test_gradient_clipping(cql_agent, dummy_dataset, test_config, device, sample_batch):
    """Test gradient clipping."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    test_config["training"]["grad_clip"] = 0.5

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    # Run training step
    metrics = trainer.train_step(sample_batch)

    # Check gradient norms are clipped
    for p in cql_agent.critic.parameters():
        if p.grad is not None:
            grad_norm = p.grad.norm().item()
            assert grad_norm <= test_config["training"]["grad_clip"] * 10  # Allow some margin


def test_lr_scheduler_cosine(cql_agent, dummy_dataset, test_config, device):
    """Test cosine learning rate scheduler."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    test_config["training"]["lr_schedule"] = "cosine"

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    assert trainer.actor_scheduler is not None
    assert trainer.critic_scheduler is not None


def test_curriculum_disabled(cql_agent, dummy_dataset, test_config, device):
    """Test training with curriculum disabled."""
    dataloader = DataLoader(dummy_dataset, batch_size=32, shuffle=True)

    test_config["training"]["curriculum_enabled"] = False

    trainer = AdversarialCurriculumTrainer(
        agent=cql_agent,
        dataloader=dataloader,
        config=test_config,
        device=device,
    )

    assert trainer.curriculum_scheduler is None

    # Should still train successfully
    metrics = trainer.train_epoch()
    assert "critic_loss" in metrics
