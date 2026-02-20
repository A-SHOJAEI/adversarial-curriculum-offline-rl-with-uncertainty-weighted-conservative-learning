# Installation and Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Git (for cloning)

## Installation

### 1. Clone or navigate to the project

```bash
cd adversarial-curriculum-offline-rl-with-uncertainty-weighted-conservative-learning
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install project in development mode (optional)

```bash
pip install -e .
```

## Verify Installation

Run the verification script:

```bash
python scripts/verify_setup.py
```

This will check:
- All modules can be imported
- Configuration files are valid
- PyTorch is properly installed
- Model instantiation works

## Quick Start

### Training

Train the full model with adversarial curriculum:

```bash
python scripts/train.py --config configs/default.yaml --num-epochs 100
```

This will:
- Download the D4RL HalfCheetah-medium-v2 dataset
- Train a CQL agent with curriculum learning
- Save checkpoints to `checkpoints/`
- Log training curves to `logs/`
- Save results to `results/`

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --num-episodes 10
```

For comprehensive evaluation with OOD states and multiple environments:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --ood-eval \
    --multi-env \
    --num-episodes 20 \
    --output-dir results/evaluation
```

### Prediction

Make predictions on new states:

```bash
# From stdin
echo '[0.1, 0.2, 0.3, ..., 0.5]' | python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --with-uncertainty

# From file
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --state-file my_states.json \
    --output predictions.json \
    --with-uncertainty
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On Mac
xdg-open htmlcov/index.html  # On Linux
```

## Ablation Study

Compare the full model with curriculum learning against the baseline:

```bash
# Train full model
python scripts/train.py --config configs/default.yaml

# Train baseline (no curriculum)
python scripts/train.py --config configs/ablation.yaml

# Evaluate both
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output-dir results/full
python scripts/evaluate.py --checkpoint checkpoints_ablation/best_model.pt --output-dir results/baseline
```

## Configuration

All hyperparameters are in `configs/default.yaml`:

```yaml
# Key parameters to tune
training:
  learning_rate: 0.0003
  cql_alpha: 5.0              # Conservative penalty strength
  initial_conservatism: 10.0  # Starting curriculum weight
  min_conservatism: 1.0       # Minimum curriculum weight
  uncertainty_threshold: 0.5  # Threshold for high uncertainty
  curriculum_enabled: true    # Enable/disable curriculum
```

## Troubleshooting

### CUDA out of memory

Reduce batch size in config:

```yaml
training:
  batch_size: 128  # Default is 256
```

### D4RL installation issues

Install from source:

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master
```

### Import errors

Make sure you're in the project root and have installed dependencies:

```bash
pip install -r requirements.txt
```

## Next Steps

1. Read `README.md` for project overview
2. Check `PROJECT_SUMMARY.md` for technical details
3. Explore `src/` to understand the implementation
4. Run tests to verify everything works
5. Start training your own models

## Support

For issues, please check:
1. Dependencies are correctly installed
2. Python version is 3.8+
3. CUDA is properly configured (if using GPU)
4. Config files are valid YAML
