#!/usr/bin/env python
"""Training script for adversarial curriculum offline RL."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.loader import (
    load_d4rl_dataset,
    create_ood_states,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.training.trainer import (
    AdversarialCurriculumTrainer,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.metrics import (
    evaluate_agent,
    compute_normalized_score,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.analysis import (
    plot_training_curves,
)
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.utils.config import (
    load_config,
    setup_logging,
    set_seed,
    get_device,
    create_directories,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train adversarial curriculum offline RL agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train",
    )
    return parser.parse_args()


def train_agent(
    trainer: AdversarialCurriculumTrainer,
    num_epochs: int,
    config: dict,
    logger: logging.Logger,
) -> list:
    """Train the agent.

    Args:
        trainer: Trainer instance.
        num_epochs: Number of epochs to train.
        config: Configuration dictionary.
        logger: Logger instance.

    Returns:
        List of metrics history.
    """
    metrics_history = []
    best_loss = float("inf")

    train_config = config.get("training", {})
    logging_config = config.get("logging", {})

    checkpoint_dir = Path(logging_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {num_epochs} epochs")

    try:
        for epoch in range(num_epochs):
            # Train epoch
            epoch_metrics = trainer.train_epoch()
            metrics_history.append(epoch_metrics)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Critic Loss: {epoch_metrics['critic_loss']:.4f}, "
                f"Actor Loss: {epoch_metrics['actor_loss']:.4f}, "
                f"Mean Q: {epoch_metrics['mean_q']:.4f}, "
                f"Uncertainty: {epoch_metrics['uncertainty']:.4f}"
            )

            # Log to MLflow
            try:
                import mlflow
                mlflow.log_metrics(
                    {
                        f"train/{k}": v
                        for k, v in epoch_metrics.items()
                    },
                    step=epoch,
                )
            except Exception as e:
                logger.debug(f"MLflow logging failed: {e}")

            # Save best checkpoint
            if epoch_metrics["critic_loss"] < best_loss:
                best_loss = epoch_metrics["critic_loss"]
                best_path = checkpoint_dir / "best_model.pt"
                trainer.save_checkpoint(
                    str(best_path),
                    epoch=epoch,
                    metrics=epoch_metrics,
                )
                logger.info(f"Saved best model with loss {best_loss:.4f}")

            # Save periodic checkpoint
            save_freq = train_config.get("save_freq", 10000)
            if (epoch + 1) % max(1, save_freq // len(trainer.dataloader)) == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                trainer.save_checkpoint(
                    str(checkpoint_path),
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            # Early stopping
            if trainer.should_stop_early(epoch_metrics["critic_loss"]):
                logger.info("Early stopping triggered")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    logger.info("Training complete")

    return metrics_history


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Create directories
    create_directories(config)

    # Setup logging
    logging_config = config.get("logging", {})
    logger = setup_logging(
        log_dir=logging_config.get("log_dir", "logs"),
        log_level=logging.INFO,
    )

    logger.info(f"Starting training with config: {args.config}")

    # Set random seed
    experiment_config = config.get("experiment", {})
    seed = experiment_config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Get device
    device_name = experiment_config.get("device", "cuda")
    device = get_device(device_name)
    logger.info(f"Using device: {device}")

    # Initialize MLflow
    if logging_config.get("use_mlflow", True):
        try:
            import mlflow
            mlflow.set_experiment(experiment_config.get("name", "adversarial_curriculum_cql"))
            mlflow.start_run()
            mlflow.log_params({
                "config": args.config,
                "seed": seed,
                "device": str(device),
            })
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")

    # Load dataset
    env_config = config.get("env", {})
    env_name = env_config.get("name", "halfcheetah-medium-v2")

    logger.info(f"Loading dataset: {env_name}")

    try:
        dataset, env, stats = load_d4rl_dataset(
            env_name=env_name,
            normalize_obs=env_config.get("normalize_obs", True),
            normalize_reward=env_config.get("normalize_reward", True),
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Create dataloader
    train_config = config.get("training", {})
    batch_size = train_config.get("batch_size", 256)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"Created dataloader with batch size {batch_size}")

    # Get dimensions
    sample = dataset[0]
    state_dim = sample["observations"].shape[0]
    action_dim = sample["actions"].shape[0]

    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    model_config = config.get("model", {})

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

    agent.to(device)
    logger.info(f"Created CQL agent with {sum(p.numel() for p in agent.parameters())} parameters")

    # Create trainer
    trainer = AdversarialCurriculumTrainer(
        agent=agent,
        dataloader=dataloader,
        config=config,
        device=device,
    )

    # Load checkpoint if provided
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Evaluation only mode
    if args.eval_only:
        logger.info("Evaluation mode")
        metrics = evaluate_agent(
            agent,
            env,
            num_episodes=config.get("evaluation", {}).get("num_episodes", 10),
            device=device,
        )

        normalized_score = compute_normalized_score(metrics["mean_return"], env_name)

        logger.info(f"Evaluation results:")
        logger.info(f"  Mean return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
        logger.info(f"  Normalized score: {normalized_score:.2f}")

        return

    # Train
    num_epochs = args.num_epochs
    if num_epochs is None:
        # Use explicit num_epochs from config if available, otherwise calculate from timesteps
        num_epochs = train_config.get("num_epochs", None)
        if num_epochs is None:
            total_timesteps = train_config.get("total_timesteps", 1000000)
            steps_per_epoch = max(1, len(dataset) // batch_size)
            num_epochs = max(1, total_timesteps // steps_per_epoch)

    logger.info(f"Training for {num_epochs} epochs")

    metrics_history = train_agent(trainer, num_epochs, config, logger)

    # Plot training curves
    results_dir = Path(logging_config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_path = results_dir / "training_curves.png"
    plot_training_curves(metrics_history, save_path=str(plot_path))
    logger.info(f"Saved training curves to {plot_path}")

    # Final evaluation
    logger.info("Running final evaluation")
    eval_metrics = evaluate_agent(
        agent,
        env,
        num_episodes=config.get("evaluation", {}).get("num_episodes", 10),
        device=device,
    )

    normalized_score = compute_normalized_score(eval_metrics["mean_return"], env_name)

    logger.info("Final evaluation results:")
    logger.info(f"  Mean return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
    logger.info(f"  Normalized score: {normalized_score:.2f}")

    # Save final results
    import json

    results = {
        "environment": env_name,
        "mean_return": eval_metrics["mean_return"],
        "std_return": eval_metrics["std_return"],
        "normalized_score": normalized_score,
        "config": args.config,
    }

    results_path = results_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved final results to {results_path}")

    # Close MLflow
    try:
        import mlflow
        mlflow.log_metrics({
            "final/mean_return": eval_metrics["mean_return"],
            "final/normalized_score": normalized_score,
        })
        mlflow.end_run()
    except Exception:
        pass

    # Cleanup
    env.close()

    logger.info("Training script complete")


if __name__ == "__main__":
    main()
