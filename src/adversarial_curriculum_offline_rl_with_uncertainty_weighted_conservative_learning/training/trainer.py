"""Training loop with curriculum learning and early stopping."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.components import (
    CQLLoss,
    UncertaintyWeightedCurriculumScheduler,
    compute_ensemble_uncertainty,
    AdversarialStateGenerator,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent

logger = logging.getLogger(__name__)


class AdversarialCurriculumTrainer:
    """Trainer for adversarial curriculum offline RL.

    Implements the training loop with:
    - Conservative Q-Learning loss
    - Uncertainty-weighted curriculum scheduling
    - Early stopping with patience
    - Learning rate scheduling
    - Gradient clipping

    Args:
        agent: CQL agent to train.
        dataloader: Training data loader.
        config: Training configuration.
        device: Training device.
    """

    def __init__(
        self,
        agent: CQLAgent,
        dataloader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer."""
        self.agent = agent
        self.dataloader = dataloader
        self.config = config
        self.device = device

        # Extract training config
        train_config = config.get("training", {})

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            agent.actor.parameters(),
            lr=train_config.get("learning_rate", 0.0003),
            weight_decay=train_config.get("weight_decay", 0.0001),
        )

        self.critic_optimizer = torch.optim.Adam(
            agent.critic.parameters(),
            lr=train_config.get("learning_rate", 0.0003),
            weight_decay=train_config.get("weight_decay", 0.0001),
        )

        # Learning rate schedulers
        self.actor_scheduler = self._create_scheduler(
            self.actor_optimizer,
            train_config,
        )
        self.critic_scheduler = self._create_scheduler(
            self.critic_optimizer,
            train_config,
        )

        # Loss function with learnable alpha
        self.cql_loss = CQLLoss(
            initial_alpha=train_config.get("cql_alpha", 5.0),
            temperature=train_config.get("cql_temperature", 1.0),
            target_action_gap=train_config.get("target_action_gap", 0.0),
            learnable_alpha=train_config.get("learnable_alpha", True),
        )

        # Optimizer for CQL alpha if learnable
        if train_config.get("learnable_alpha", True):
            self.alpha_optimizer = torch.optim.Adam(
                [self.cql_loss.log_alpha],
                lr=train_config.get("alpha_lr", 0.0001),
            )
        else:
            self.alpha_optimizer = None

        # Adversarial state generator for OOD curriculum
        if train_config.get("use_adversarial_generator", True):
            self.adversarial_generator = AdversarialStateGenerator(
                state_dim=agent.state_dim,
                hidden_dim=train_config.get("generator_hidden_dim", 256),
                noise_dim=train_config.get("generator_noise_dim", 32),
            ).to(device)

            self.generator_optimizer = torch.optim.Adam(
                self.adversarial_generator.parameters(),
                lr=train_config.get("generator_lr", 0.0001),
            )
        else:
            self.adversarial_generator = None
            self.generator_optimizer = None

        # Curriculum scheduler
        if train_config.get("curriculum_enabled", True):
            self.curriculum_scheduler = UncertaintyWeightedCurriculumScheduler(
                initial_conservatism=train_config.get("initial_conservatism", 10.0),
                min_conservatism=train_config.get("min_conservatism", 1.0),
                warmup_steps=train_config.get("curriculum_warmup", 100),
                uncertainty_threshold=train_config.get("uncertainty_threshold", 0.5),
            )
        else:
            self.curriculum_scheduler = None

        # Training parameters
        self.gamma = train_config.get("gamma", 0.99)
        self.tau = train_config.get("tau", 0.005)
        self.grad_clip = train_config.get("grad_clip", 1.0)
        self.target_update_freq = train_config.get("target_update_freq", 2)

        # Early stopping
        self.early_stopping = train_config.get("early_stopping", True)
        self.patience = train_config.get("patience", 50)
        self.min_delta = train_config.get("min_delta", 0.01)
        self.best_loss = float("inf")
        self.patience_counter = 0

        # Tracking
        self.step_count = 0
        self.epoch_count = 0

        logger.info("Initialized AdversarialCurriculumTrainer")

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        train_config: Dict[str, Any],
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule.
            train_config: Training configuration.

        Returns:
            LR scheduler or None.
        """
        schedule_type = train_config.get("lr_schedule", "cosine")

        if schedule_type == "cosine":
            # Compute T_max as number of epochs.  Use explicit num_epochs if
            # available, otherwise derive from total_timesteps.  Ensure >= 1
            # to avoid division by zero in the scheduler.
            num_epochs = train_config.get("num_epochs", None)
            if num_epochs is None:
                dl_len = max(1, len(self.dataloader))
                num_epochs = max(1, train_config.get("total_timesteps", 1000000) // dl_len)
            t_max = max(1, int(num_epochs))
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=train_config.get("min_lr", 0.00001),
            )
        elif schedule_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
                gamma=0.5,
            )
        elif schedule_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
        else:
            return None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with bootstrap masking and adversarial OOD generation.

        Args:
            batch: Batch of transitions.

        Returns:
            Dictionary of losses and metrics.
        """
        # Move batch to device
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        terminals = batch["terminals"].to(self.device)

        batch_size = observations.shape[0]

        # Generate bootstrap mask for ensemble
        bootstrap_mask = self.agent.critic.get_bootstrap_mask(batch_size, self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.agent.target_actor(next_observations)
            target_q = self.agent.target_critic(next_observations, next_actions, mask=bootstrap_mask)
            target_q = rewards + (1 - terminals) * self.gamma * target_q

        # Current Q-values with bootstrap mask
        current_q = self.agent.critic(observations, actions, mask=bootstrap_mask)

        # Random actions for CQL
        random_actions = torch.rand_like(actions) * 2 - 1  # [-1, 1]
        random_q_all = []
        for i in range(self.agent.critic.ensemble_size):
            q = self.agent.critic.critics[i](observations, random_actions)
            random_q_all.append(q)
        random_q = torch.cat(random_q_all, dim=1)

        # Policy actions for CQL
        with torch.no_grad():
            policy_actions = self.agent.actor(observations)
        policy_q_all = []
        for i in range(self.agent.critic.ensemble_size):
            q = self.agent.critic.critics[i](observations, policy_actions)
            policy_q_all.append(q)
        policy_q = torch.cat(policy_q_all, dim=1)

        # Generate adversarial OOD states if enabled
        if self.adversarial_generator is not None and self.step_count > 1000:
            noise = torch.randn(batch_size, self.adversarial_generator.noise_dim, device=self.device)
            ood_states = self.adversarial_generator(observations, noise)

            # Get uncertainty on OOD states to train generator
            with torch.no_grad():
                ood_actions = self.agent.actor(ood_states)
                _, ood_uncertainty = self.agent.critic.get_uncertainty(ood_states, ood_actions)

            # Train generator to maximize uncertainty (adversarial objective)
            generator_loss = -ood_uncertainty.mean()

            if self.generator_optimizer is not None:
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                self.generator_optimizer.step()
        else:
            generator_loss = torch.tensor(0.0)

        # Get curriculum weight
        if self.curriculum_scheduler is not None:
            _, uncertainty = self.agent.critic.get_uncertainty(observations, actions)
            mean_uncertainty = uncertainty.mean().item()
            curriculum_weight = self.curriculum_scheduler.step(mean_uncertainty)
        else:
            curriculum_weight = 1.0
            mean_uncertainty = 0.0

        # Compute CQL loss with learnable alpha
        critic_loss, td_loss, conservative_loss, alpha_loss = self.cql_loss(
            current_q,
            target_q,
            random_q,
            policy_q,
            curriculum_weight=curriculum_weight,
        )

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.critic.parameters(),
                self.grad_clip,
            )

        self.critic_optimizer.step()

        # Update alpha if learnable
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Update actor (policy improvement)
        policy_actions = self.agent.actor(observations)
        actor_q = self.agent.critic(observations, policy_actions)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.actor.parameters(),
                self.grad_clip,
            )

        self.actor_optimizer.step()

        # Update target networks
        if self.step_count % self.target_update_freq == 0:
            self.agent.update_targets(self.tau)

        self.step_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "td_loss": td_loss.item(),
            "conservative_loss": conservative_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_q": current_q.mean().item(),
            "uncertainty": mean_uncertainty,
            "curriculum_weight": curriculum_weight,
            "alpha": self.cql_loss.alpha.item(),
            "alpha_loss": alpha_loss.item(),
            "generator_loss": generator_loss.item() if isinstance(generator_loss, torch.Tensor) else 0.0,
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of epoch metrics.
        """
        self.agent.train()

        epoch_metrics = {
            "critic_loss": [],
            "td_loss": [],
            "conservative_loss": [],
            "actor_loss": [],
            "mean_q": [],
            "uncertainty": [],
            "curriculum_weight": [],
            "alpha": [],
            "alpha_loss": [],
            "generator_loss": [],
        }

        for batch in tqdm(self.dataloader, desc=f"Epoch {self.epoch_count}"):
            metrics = self.train_step(batch)

            for key, value in metrics.items():
                epoch_metrics[key].append(value)

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # Update learning rate
        if self.actor_scheduler is not None:
            if isinstance(self.actor_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.actor_scheduler.step(avg_metrics["actor_loss"])
                self.critic_scheduler.step(avg_metrics["critic_loss"])
            else:
                self.actor_scheduler.step()
                self.critic_scheduler.step()

        self.epoch_count += 1

        return avg_metrics

    def should_stop_early(self, current_loss: float) -> bool:
        """Check if training should stop early.

        Args:
            current_loss: Current epoch loss.

        Returns:
            True if should stop.
        """
        if not self.early_stopping:
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement"
                )
                return True

        return False

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "step": self.step_count,
            "agent_state_dict": self.agent.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self.best_loss,
        }

        if self.actor_scheduler is not None:
            checkpoint["actor_scheduler"] = self.actor_scheduler.state_dict()
            checkpoint["critic_scheduler"] = self.critic_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint.

        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if "actor_scheduler" in checkpoint and self.actor_scheduler is not None:
            self.actor_scheduler.load_state_dict(checkpoint["actor_scheduler"])
            self.critic_scheduler.load_state_dict(checkpoint["critic_scheduler"])

        self.step_count = checkpoint["step"]
        self.epoch_count = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]

        logger.info(f"Loaded checkpoint from {path}")

        return checkpoint
