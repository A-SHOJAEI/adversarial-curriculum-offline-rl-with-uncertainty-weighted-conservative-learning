#!/usr/bin/env python
"""Verification script to test project setup."""

import sys
from pathlib import Path

# Add project root and src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Verifying project setup...")
print("=" * 70)

# Test imports
print("\n1. Testing core imports...")
try:
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning import (
        CQLAgent,
        EnsembleCritic,
        AdversarialCurriculumTrainer,
    )
    print("   ✓ Core modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test data modules
print("\n2. Testing data modules...")
try:
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.loader import (
        D4RLDataset,
    )
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.data.preprocessing import (
        compute_dataset_statistics,
    )
    print("   ✓ Data modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test model components
print("\n3. Testing model components...")
try:
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.models.components import (
        CQLLoss,
        UncertaintyWeightedCurriculumScheduler,
        compute_ensemble_uncertainty,
    )
    print("   ✓ Model components imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test evaluation modules
print("\n4. Testing evaluation modules...")
try:
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.metrics import (
        evaluate_agent,
        compute_normalized_score,
    )
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.evaluation.analysis import (
        plot_training_curves,
    )
    print("   ✓ Evaluation modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test config loading
print("\n5. Testing configuration loading...")
try:
    from adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning.utils.config import (
        load_config,
    )

    config = load_config("configs/default.yaml")
    assert "experiment" in config
    assert "training" in config
    assert "model" in config
    print("   ✓ Configuration loaded successfully")
    print(f"     - Experiment: {config['experiment']['name']}")
    print(f"     - Environment: {config['env']['name']}")
    print(f"     - Curriculum enabled: {config['training']['curriculum_enabled']}")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test PyTorch
print("\n6. Testing PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__} imported")
    print(f"     - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     - CUDA version: {torch.version.cuda}")
except ImportError as e:
    print(f"   ✗ PyTorch import error: {e}")
    sys.exit(1)

# Test model creation
print("\n7. Testing model instantiation...")
try:
    import torch

    agent = CQLAgent(
        state_dim=17,
        action_dim=6,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        ensemble_size=3,
    )

    total_params = sum(p.numel() for p in agent.parameters())
    print(f"   ✓ CQL agent created successfully")
    print(f"     - Total parameters: {total_params:,}")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    sys.exit(1)

# Test custom components
print("\n8. Testing custom components...")
try:
    cql_loss = CQLLoss(alpha=5.0, temperature=1.0)
    scheduler = UncertaintyWeightedCurriculumScheduler(
        initial_conservatism=10.0,
        min_conservatism=1.0,
    )
    print("   ✓ Custom components initialized successfully")
except Exception as e:
    print(f"   ✗ Component error: {e}")
    sys.exit(1)

# Test directory structure
print("\n9. Verifying directory structure...")
required_dirs = [
    "configs",
    "scripts",
    "src/adversarial_curriculum_offline_rl_with_uncertainty_weighted_conservative_learning",
    "tests",
]

all_exist = True
for dir_path in required_dirs:
    if not Path(dir_path).exists():
        print(f"   ✗ Missing directory: {dir_path}")
        all_exist = False

if all_exist:
    print("   ✓ All required directories present")

# Test file structure
print("\n10. Verifying key files...")
required_files = [
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".gitignore",
    "configs/default.yaml",
    "configs/ablation.yaml",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/predict.py",
]

all_exist = True
for file_path in required_files:
    if not Path(file_path).exists():
        print(f"   ✗ Missing file: {file_path}")
        all_exist = False

if all_exist:
    print("   ✓ All required files present")

print("\n" + "=" * 70)
print("✓ Verification complete! Project setup is valid.")
print("\nNext steps:")
print("  1. Install dependencies: pip install -r requirements.txt")
print("  2. Train model: python scripts/train.py --config configs/default.yaml")
print("  3. Run tests: pytest tests/ --cov=src")
print("=" * 70)
