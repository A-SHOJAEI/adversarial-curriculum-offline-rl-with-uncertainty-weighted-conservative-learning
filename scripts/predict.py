#!/usr/bin/env python
"""Prediction script for inference on new states."""

import argparse
import json
import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.model import CQLAgent
from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.utils.config import (
    load_config,
    get_device,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict actions for given states")
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
        "--state",
        type=str,
        default=None,
        help="State as JSON array or file path",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="Path to file containing states (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions",
    )
    parser.add_argument(
        "--with-uncertainty",
        action="store_true",
        help="Include uncertainty estimates",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy",
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


def predict_action(
    agent: CQLAgent,
    state: np.ndarray,
    device: torch.device,
    deterministic: bool = True,
    with_uncertainty: bool = False,
) -> dict:
    """Predict action for given state.

    Args:
        agent: Trained agent.
        state: State array.
        device: Device to run on.
        deterministic: Use deterministic policy.
        with_uncertainty: Include uncertainty estimate.

    Returns:
        Dictionary with action and optional uncertainty.
    """
    # Convert to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        # Predict action
        action = agent.select_action(state_tensor, deterministic=deterministic)
        action = action.cpu().numpy().flatten()

        result = {
            "action": action.tolist(),
        }

        # Add uncertainty if requested
        if with_uncertainty:
            _, uncertainty = agent.critic.get_uncertainty(state_tensor, torch.FloatTensor(action).unsqueeze(0).to(device))
            result["uncertainty"] = uncertainty.cpu().numpy().item()

            # Add Q-value
            q_value = agent.critic(state_tensor, torch.FloatTensor(action).unsqueeze(0).to(device))
            result["q_value"] = q_value.cpu().numpy().item()

    return result


def main():
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Get device
    device_name = config.get("experiment", {}).get("device", "cuda")
    device = get_device(device_name)

    # Determine state dimension from config
    # For simplicity, we'll try to infer from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Get dimensions from checkpoint
    if "agent_state_dict" in checkpoint:
        state_dict = checkpoint["agent_state_dict"]
    else:
        state_dict = checkpoint

    # Infer dimensions from first layer
    first_layer_key = None
    for key in state_dict.keys():
        if "actor.net.0.weight" in key:
            first_layer_key = key
            break

    if first_layer_key is None:
        print("Error: Could not infer state dimension from checkpoint")
        sys.exit(1)

    state_dim = state_dict[first_layer_key].shape[1]
    action_dim = state_dict["actor.net." + str(len(state_dict) // 2 - 1) + ".weight"].shape[0]

    # For output layer, find the last linear layer
    for key in state_dict.keys():
        if "actor.net" in key and "weight" in key:
            action_dim = state_dict[key].shape[0]

    print(f"Inferred dimensions: state_dim={state_dim}, action_dim={action_dim}")

    # Load agent
    print(f"Loading agent from {args.checkpoint}")
    agent = load_agent_from_checkpoint(
        args.checkpoint,
        state_dim,
        action_dim,
        config,
        device,
    )

    # Load states
    states = []

    if args.state is not None:
        # Parse single state
        try:
            state = json.loads(args.state)
            states.append(np.array(state, dtype=np.float32))
        except json.JSONDecodeError:
            # Try loading as file
            try:
                state = np.load(args.state)
                states.append(state)
            except Exception:
                print(f"Error: Could not parse state: {args.state}")
                sys.exit(1)

    elif args.state_file is not None:
        # Load states from file
        try:
            with open(args.state_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        state = json.loads(line)
                        states.append(np.array(state, dtype=np.float32))
        except Exception as e:
            print(f"Error loading state file: {e}")
            sys.exit(1)

    else:
        # Read from stdin
        print("Enter state as JSON array (or Ctrl+D to finish):")
        for line in sys.stdin:
            line = line.strip()
            if line:
                try:
                    state = json.loads(line)
                    states.append(np.array(state, dtype=np.float32))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")

    if len(states) == 0:
        print("Error: No states provided")
        sys.exit(1)

    # Validate state dimensions
    for i, state in enumerate(states):
        if len(state) != state_dim:
            print(f"Error: State {i} has dimension {len(state)}, expected {state_dim}")
            sys.exit(1)

    # Make predictions
    predictions = []

    print(f"\nMaking predictions for {len(states)} state(s)...")

    for i, state in enumerate(states):
        result = predict_action(
            agent,
            state,
            device,
            deterministic=args.deterministic,
            with_uncertainty=args.with_uncertainty,
        )

        predictions.append(result)

        # Print to console
        print(f"\nState {i + 1}:")
        print(f"  Action: {result['action']}")

        if args.with_uncertainty:
            print(f"  Q-value: {result['q_value']:.4f}")
            print(f"  Uncertainty: {result['uncertainty']:.4f}")

    # Save to file if requested
    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"\nPredictions saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Number of predictions: {len(predictions)}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Deterministic: {args.deterministic}")

    if args.with_uncertainty:
        uncertainties = [p["uncertainty"] for p in predictions]
        print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
        print(f"Std uncertainty: {np.std(uncertainties):.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
