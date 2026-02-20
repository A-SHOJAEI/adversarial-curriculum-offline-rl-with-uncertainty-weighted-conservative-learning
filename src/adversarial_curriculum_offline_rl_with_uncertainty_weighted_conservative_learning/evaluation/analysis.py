"""Results analysis and visualization."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def plot_training_curves(
    metrics_history: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves.

    Args:
        metrics_history: List of metrics per epoch.
        save_path: Path to save figure.
    """
    df = pd.DataFrame(metrics_history)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot each metric
    metrics_to_plot = [
        ("critic_loss", "Critic Loss"),
        ("actor_loss", "Actor Loss"),
        ("mean_q", "Mean Q-Value"),
        ("uncertainty", "Mean Uncertainty"),
        ("curriculum_weight", "Curriculum Weight"),
        ("conservative_loss", "Conservative Loss"),
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        if metric in df.columns:
            ax.plot(df.index, df[metric], linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")

    plt.close()


def plot_uncertainty_analysis(
    id_uncertainties: np.ndarray,
    ood_uncertainties: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot uncertainty distributions for ID and OOD states.

    Args:
        id_uncertainties: Uncertainties for in-distribution states.
        ood_uncertainties: Uncertainties for out-of-distribution states.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(id_uncertainties, bins=50, alpha=0.6, label="In-Distribution", density=True)
    ax.hist(ood_uncertainties, bins=50, alpha=0.6, label="Out-of-Distribution", density=True)
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot
    ax = axes[1]
    data = [id_uncertainties, ood_uncertainties]
    ax.boxplot(data, labels=["In-Distribution", "Out-of-Distribution"])
    ax.set_ylabel("Uncertainty")
    ax.set_title("Uncertainty Comparison")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved uncertainty analysis to {save_path}")

    plt.close()


def plot_returns_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison of returns across environments.

    Args:
        results: Dictionary mapping env names to metrics.
        save_path: Path to save figure.
    """
    env_names = list(results.keys())
    mean_returns = [results[env]["mean_return"] for env in env_names]
    std_returns = [results[env]["std_return"] for env in env_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(env_names))
    ax.bar(x, mean_returns, yerr=std_returns, capsize=5, alpha=0.7)
    ax.set_xlabel("Environment")
    ax.set_ylabel("Mean Return")
    ax.set_title("Performance Across Environments")
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved returns comparison to {save_path}")

    plt.close()


def generate_evaluation_report(
    results: Dict[str, Any],
    save_dir: str,
) -> None:
    """Generate comprehensive evaluation report.

    Args:
        results: Dictionary of evaluation results.
        save_dir: Directory to save report files.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = save_path / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {json_path}")

    # Create summary table
    if "environments" in results:
        summary_data = []
        for env_name, metrics in results["environments"].items():
            summary_data.append({
                "Environment": env_name,
                "Mean Return": f"{metrics['mean_return']:.2f}",
                "Std Return": f"{metrics['std_return']:.2f}",
                "Normalized Score": f"{metrics.get('normalized_score', 0):.2f}",
                "Mean Length": f"{metrics['mean_length']:.1f}",
            })

        df = pd.DataFrame(summary_data)

        # Save CSV
        csv_path = save_path / "summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary CSV to {csv_path}")

        # Plot comparison
        plot_path = save_path / "returns_comparison.png"
        plot_returns_comparison(results["environments"], str(plot_path))

    # Save OOD results
    if "ood_metrics" in results:
        ood_path = save_path / "ood_metrics.json"
        with open(ood_path, "w") as f:
            json.dump(results["ood_metrics"], f, indent=2)
        logger.info(f"Saved OOD metrics to {ood_path}")

    logger.info(f"Evaluation report generated in {save_dir}")


def create_ablation_comparison(
    baseline_results: Dict[str, Any],
    full_results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create comparison table for ablation study.

    Args:
        baseline_results: Results from baseline model.
        full_results: Results from full model.
        save_path: Path to save comparison.

    Returns:
        Comparison DataFrame.
    """
    comparison_data = []

    # Compare environment performance
    if "environments" in baseline_results and "environments" in full_results:
        for env_name in baseline_results["environments"].keys():
            baseline_score = baseline_results["environments"][env_name].get(
                "normalized_score", 0
            )
            full_score = full_results["environments"][env_name].get(
                "normalized_score", 0
            )

            comparison_data.append({
                "Environment": env_name,
                "Baseline Score": f"{baseline_score:.2f}",
                "Full Model Score": f"{full_score:.2f}",
                "Improvement": f"{full_score - baseline_score:.2f}",
                "Relative Gain (%)": f"{(full_score - baseline_score) / (baseline_score + 1e-8) * 100:.1f}",
            })

    df = pd.DataFrame(comparison_data)

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved ablation comparison to {save_path}")

    return df


def plot_curriculum_progression(
    curriculum_history: List[float],
    uncertainty_history: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot curriculum weight progression over training.

    Args:
        curriculum_history: History of curriculum weights.
        uncertainty_history: History of mean uncertainties.
        save_path: Path to save figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Curriculum weight
    ax = axes[0]
    ax.plot(curriculum_history, linewidth=2, color="blue")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Curriculum Weight")
    ax.set_title("Curriculum Weight Progression")
    ax.grid(True, alpha=0.3)

    # Mean uncertainty
    ax = axes[1]
    ax.plot(uncertainty_history, linewidth=2, color="red")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Uncertainty")
    ax.set_title("Mean Uncertainty Over Training")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved curriculum progression to {save_path}")

    plt.close()
