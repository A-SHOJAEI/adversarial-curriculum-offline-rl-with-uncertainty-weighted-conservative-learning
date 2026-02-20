#!/usr/bin/env python
"""Evaluation script for trained agents."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gymnasium as gym
import numpy as np
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.loader import (
    load_d4rl_dataset,
    create_ood_states,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.metrics import (
    evaluate_agent,
    compute_normalized_score,
    compute_ood_success_rate,
    compute_ensemble_disagreement_correlation,
    evaluate_multiple_environments,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.analysis import (
    generate_evaluation_report,
    plot_uncertainty_analysis,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained offline RL agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )
    parser.add_argument(
        "--multi-env",
        action="store_true",
        help="Evaluate on multiple environments",
    )
    parser.add_argument(
        "--ood-eval",
        action="store_true",
        help="Perform OOD state evaluation",
    )
    return parser.parse_args()


def load_agent_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    config: dict,
    device: torch.device,
) -> CQLAgent:
    """Load agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        state_dim: State dimension.
        action_dim: Action dimension.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded CQL agent.
    """
    model_config = config.get("model", {})

    # Create agent
    agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_dims=model_config.get("actor_hidden_dims", [256, 256, 256]),
        critic_hidden_dims=model_config.get("critic_hidden_dims", [256, 256, 256]),
        ensemble_size=model_config.get("ensemble_size", 5),
        activation=model_config.get("activation", "relu"),
        max_action=1.0,
        dropout=model_config.get("dropout", 0.1),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)

    agent.to(device)
    agent.eval()

    return agent


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(
        log_dir=str(output_dir),
        log_level=logging.INFO,
        log_file="evaluation.log",
    )

    logger.info(f"Starting evaluation")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        sys.exit(1)

    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    # Get device
    device_name = config.get("experiment", {}).get("device", "cuda")
    device = get_device(device_name)
    logger.info(f"Using device: {device}")

    # Load dataset to get dimensions
    env_config = config.get("env", {})
    env_name = env_config.get("name", "halfcheetah-medium-v2")

    logger.info(f"Loading dataset: {env_name}")
    dataset, env, stats = load_d4rl_dataset(
        env_name=env_name,
        normalize_obs=env_config.get("normalize_obs", True),
        normalize_reward=env_config.get("normalize_reward", True),
    )

    # Get dimensions
    sample = dataset[0]
    state_dim = sample["observations"].shape[0]
    action_dim = sample["actions"].shape[0]

    # Load agent
    logger.info(f"Loading agent from {args.checkpoint}")
    agent = load_agent_from_checkpoint(
        args.checkpoint,
        state_dim,
        action_dim,
        config,
        device,
    )

    results = {}

    # Standard evaluation
    logger.info(f"Evaluating on {env_name}")
    eval_metrics = evaluate_agent(
        agent,
        env,
        num_episodes=args.num_episodes,
        deterministic=True,
        device=device,
    )

    normalized_score = compute_normalized_score(eval_metrics["mean_return"], env_name)
    eval_metrics["normalized_score"] = normalized_score

    logger.info("Evaluation results:")
    logger.info(f"  Mean return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
    logger.info(f"  Normalized score: {normalized_score:.2f}")
    logger.info(f"  Min return: {eval_metrics['min_return']:.2f}")
    logger.info(f"  Max return: {eval_metrics['max_return']:.2f}")
    logger.info(f"  Mean episode length: {eval_metrics['mean_length']:.1f}")

    results["primary_environment"] = {
        "name": env_name,
        "metrics": eval_metrics,
    }

    # Multi-environment evaluation
    if args.multi_env:
        logger.info("Running multi-environment evaluation")

        env_names = [
            "halfcheetah-medium-v2",
            "hopper-medium-replay-v2",
            "walker2d-medium-expert-v2",
        ]

        multi_env_results = evaluate_multiple_environments(
            agent,
            env_names,
            num_episodes=args.num_episodes,
            device=device,
        )

        results["environments"] = multi_env_results

        # Print summary
        logger.info("\nMulti-environment results:")
        for env_name, metrics in multi_env_results.items():
            logger.info(
                f"  {env_name}: {metrics['mean_return']:.2f} "
                f"(normalized: {metrics['normalized_score']:.2f})"
            )

    # OOD evaluation
    if args.ood_eval:
        logger.info("Running OOD state evaluation")

        # Create OOD states
        num_ood_samples = 1000
        ood_states = create_ood_states(dataset, num_ood_samples, noise_scale=0.5)

        # Sample in-distribution states
        id_indices = torch.randint(0, len(dataset), (num_ood_samples,))
        id_states = dataset.observations[id_indices]

        # Evaluate OOD performance
        ood_metrics = compute_ood_success_rate(
            agent,
            ood_states,
            id_states,
            device,
            threshold=config.get("training", {}).get("uncertainty_threshold", 0.5),
        )

        results["ood_metrics"] = ood_metrics

        logger.info("OOD evaluation results:")
        logger.info(f"  Success rate: {ood_metrics['ood_success_rate']:.3f}")
        logger.info(f"  ID low uncertainty rate: {ood_metrics['id_low_uncertainty_rate']:.3f}")
        logger.info(f"  OOD high uncertainty rate: {ood_metrics['ood_high_uncertainty_rate']:.3f}")

        # Compute ensemble disagreement correlation
        sample_indices = torch.randint(0, len(dataset), (1000,))
        sample_states = dataset.observations[sample_indices].to(device)
        sample_actions = dataset.actions[sample_indices].to(device)

        correlation = compute_ensemble_disagreement_correlation(
            agent,
            sample_states,
            sample_actions,
            device,
        )

        results["ensemble_disagreement_correlation"] = correlation
        logger.info(f"  Ensemble disagreement correlation: {correlation:.3f}")

        # Plot uncertainty analysis
        with torch.no_grad():
            id_actions = agent.actor(id_states.to(device))
            _, id_uncertainty = agent.critic.get_uncertainty(id_states.to(device), id_actions)

            ood_actions = agent.actor(ood_states.to(device))
            _, ood_uncertainty = agent.critic.get_uncertainty(ood_states.to(device), ood_actions)

        plot_path = output_dir / "uncertainty_analysis.png"
        plot_uncertainty_analysis(
            id_uncertainty.cpu().numpy(),
            ood_uncertainty.cpu().numpy(),
            save_path=str(plot_path),
        )

    # Generate comprehensive report
    generate_evaluation_report(results, str(output_dir))

    # Save detailed results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nEvaluation complete. Results saved to {output_dir}")

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<40} {'Value':<20}")
    print("-" * 70)
    print(f"{'Mean Return':<40} {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
    print(f"{'Normalized Score':<40} {normalized_score:.2f}")
    print(f"{'Episode Length':<40} {eval_metrics['mean_length']:.1f}")

    if args.ood_eval:
        print(f"{'OOD Success Rate':<40} {ood_metrics['ood_success_rate']:.3f}")
        print(f"{'Ensemble Disagreement Correlation':<40} {correlation:.3f}")

    print("=" * 70)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
