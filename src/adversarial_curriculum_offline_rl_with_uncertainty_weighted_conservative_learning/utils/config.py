"""Configuration utilities for loading YAML configs and setting up logging."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_dir: Directory to store log files. If None, logs to console only.
        log_level: Logging level (default: INFO).
        log_file: Name of log file. If None, uses 'training.log'.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("adversarial_curriculum_rl")
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            log_file = "training.log"

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "cuda") -> torch.device:
    """Get torch device.

    Args:
        device_name: Device name ('cuda' or 'cpu').

    Returns:
        Torch device object.
    """
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories for logging and checkpoints.

    Args:
        config: Configuration dictionary.
    """
    logging_config = config.get("logging", {})

    dirs = [
        logging_config.get("log_dir", "logs"),
        logging_config.get("checkpoint_dir", "checkpoints"),
        logging_config.get("results_dir", "results"),
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
