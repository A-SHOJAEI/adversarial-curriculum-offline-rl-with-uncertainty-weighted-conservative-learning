"""Evaluation metrics for offline RL agents."""

import logging
from typing import Dict, List, Tuple

try:
    import d4rl
    D4RL_AVAILABLE = True
except ImportError:
    D4RL_AVAILABLE = False

import gymnasium as gym
import numpy as np
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent

logger = logging.getLogger(__name__)


def evaluate_agent(
    agent: CQLAgent,
    env: gym.Env,
    num_episodes: int = 10,
    deterministic: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluate agent on environment.

    Args:
        agent: Agent to evaluate.
        env: Environment to evaluate on.
        num_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic policy.
        device: Device to run evaluation on.

    Returns:
        Dictionary of evaluation metrics.
    """
    agent.eval()

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (done or truncated):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            # Select action
            with torch.no_grad():
                action = agent.select_action(obs_tensor, deterministic=deterministic)
                action = action.cpu().numpy().flatten()

            # Step environment
            obs, reward, done, truncated, _ = env.step(action)

            episode_return += reward
            episode_length += 1

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        logger.debug(
            f"Episode {episode + 1}/{num_episodes}: "
            f"return={episode_return:.2f}, length={episode_length}"
        )

    metrics = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
        "mean_length": np.mean(episode_lengths),
    }

    logger.info(
        f"Evaluation complete: mean_return={metrics['mean_return']:.2f} ± "
        f"{metrics['std_return']:.2f}"
    )

    return metrics


def compute_normalized_score(
    raw_score: float,
    env_name: str,
) -> float:
    """Compute D4RL normalized score.

    Args:
        raw_score: Raw episode return.
        env_name: D4RL environment name.

    Returns:
        Normalized score (0-100 scale).
    """
    if not D4RL_AVAILABLE:
        # Without d4rl, just return raw score as normalized score
        logger.warning("d4rl not available, returning raw score as normalized score")
        return raw_score

    # Get reference scores from D4RL
    try:
        env = gym.make(env_name)
    except Exception as e:
        logger.warning(f"Could not create environment {env_name}: {e}")
        return raw_score

    # D4RL normalized score formula
    if hasattr(env, "get_normalized_score"):
        normalized = env.get_normalized_score(raw_score) * 100
    else:
        # Fallback calculation
        ref_min_score = env.ref_min_score if hasattr(env, "ref_min_score") else 0.0
        ref_max_score = env.ref_max_score if hasattr(env, "ref_max_score") else 1.0

        if ref_max_score - ref_min_score > 0:
            normalized = (raw_score - ref_min_score) / (ref_max_score - ref_min_score) * 100
        else:
            normalized = raw_score

    try:
        env.close()
    except Exception:
        pass

    return normalized


def compute_ood_success_rate(
    agent: CQLAgent,
    ood_states: torch.Tensor,
    in_distribution_states: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute success rate on OOD states using uncertainty.

    Args:
        agent: Agent to evaluate.
        ood_states: Out-of-distribution states.
        in_distribution_states: In-distribution states.
        device: Device to run on.
        threshold: Uncertainty threshold for success.

    Returns:
        Dictionary of OOD metrics.
    """
    agent.eval()

    with torch.no_grad():
        # Get actions for OOD states
        ood_actions = agent.actor(ood_states.to(device))

        # Compute uncertainty for OOD states
        _, ood_uncertainty = agent.critic.get_uncertainty(
            ood_states.to(device),
            ood_actions,
        )

        # Get actions for in-distribution states
        id_actions = agent.actor(in_distribution_states.to(device))

        # Compute uncertainty for in-distribution states
        _, id_uncertainty = agent.critic.get_uncertainty(
            in_distribution_states.to(device),
            id_actions,
        )

    # Success: low uncertainty on ID states, high uncertainty on OOD states
    id_success = (id_uncertainty < threshold).float().mean().item()
    ood_detection = (ood_uncertainty > threshold).float().mean().item()

    # Overall success rate
    success_rate = (id_success + ood_detection) / 2.0

    metrics = {
        "ood_success_rate": success_rate,
        "id_low_uncertainty_rate": id_success,
        "ood_high_uncertainty_rate": ood_detection,
        "mean_id_uncertainty": id_uncertainty.mean().item(),
        "mean_ood_uncertainty": ood_uncertainty.mean().item(),
    }

    logger.info(
        f"OOD evaluation: success_rate={success_rate:.3f}, "
        f"ID_uncertainty={metrics['mean_id_uncertainty']:.3f}, "
        f"OOD_uncertainty={metrics['mean_ood_uncertainty']:.3f}"
    )

    return metrics


def compute_ensemble_disagreement_correlation(
    agent: CQLAgent,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute correlation between ensemble disagreement and uncertainty.

    Args:
        agent: Agent with ensemble critic.
        states: States to evaluate.
        actions: Actions to evaluate.
        device: Device to run on.

    Returns:
        Correlation coefficient.
    """
    agent.eval()

    with torch.no_grad():
        # Get all ensemble Q-values
        all_q_values = agent.critic(
            states.to(device),
            actions.to(device),
            return_all=True,
        )

        # Compute disagreement (std)
        disagreement = all_q_values.std(dim=0).cpu().numpy().flatten()

        # Compute uncertainty
        _, uncertainty = agent.critic.get_uncertainty(
            states.to(device),
            actions.to(device),
        )
        uncertainty = uncertainty.cpu().numpy().flatten()

    # Compute correlation
    correlation = np.corrcoef(disagreement, uncertainty)[0, 1]

    logger.info(f"Ensemble disagreement correlation: {correlation:.3f}")

    return correlation


def evaluate_multiple_environments(
    agent: CQLAgent,
    env_names: List[str],
    num_episodes: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, float]]:
    """Evaluate agent on multiple environments.

    Args:
        agent: Agent to evaluate.
        env_names: List of environment names.
        num_episodes: Number of episodes per environment.
        device: Device to run on.

    Returns:
        Dictionary mapping env names to metrics.
    """
    all_results = {}

    for env_name in env_names:
        logger.info(f"Evaluating on {env_name}")

        env = gym.make(env_name)
        metrics = evaluate_agent(agent, env, num_episodes, device=device)

        # Compute normalized score
        normalized_score = compute_normalized_score(metrics["mean_return"], env_name)
        metrics["normalized_score"] = normalized_score

        all_results[env_name] = metrics
        env.close()

    return all_results
