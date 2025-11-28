# GNN Sparsification Research - Migration Guide

This document outlines the migration from the original notebook-heavy structure to the new modular architecture.

## What Changed

### Directory Structure

**Before:**
```
gnn-sparsification-research/
├── notebooks/
│   ├── 01_exp_jaccard_sparsification_cora.ipynb
│   ├── 02_exp_jaccard_sparsification_pubmed.ipynb
│   └── tutorials/
├── data/
├── docs/
└── README.md
```

**After:**
```
gnn-sparsification-research/
├── configs/              # Hydra configuration files
│   ├── experiment/
│   ├── model/
│   ├── dataset/
│   └── sparsifier/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── exploratory/      # Your experiment notebooks moved here
│   └── tutorials/        # Pedagogical notebooks
├── scripts/              # Entry points (train.py)
├── src/                  # Main source code
│   ├── data/
│   ├── models/
│   ├── sparsification/   # Core research module
│   ├── training/
│   └── utils/
└── tests/                # Unit tests
```

## File Migrations Completed

1. **Notebooks:**
   - `01_exp_jaccard_sparsification_cora.ipynb` → `notebooks/exploratory/`
   - `02_exp_jaccard_sparsification_pubmed.ipynb` → `notebooks/exploratory/`
   - Tutorial notebooks remain in `notebooks/tutorials/`

2. **Documentation:**
   - `Projet_tremplin_sparsification_GNN.pdf` already in `docs/`

3. **Ignore Patterns:**
   - Updated `.gitignore` to exclude `data/`, `outputs/`, `wandb/`, and model checkpoints

## How to Use the New Structure

### Running Experiments

Instead of running notebooks, use the command-line interface:

```bash
# Run with default configuration
python scripts/train.py

# Override specific components
python scripts/train.py model=gat dataset=pubmed sparsifier=jaccard

# Use experiment presets
python scripts/train.py experiment=baseline

# Override parameters
python scripts/train.py model.hidden_dim=128 training.learning_rate=0.001
```

### Creating New Experiments

1. **Add a new sparsifier:**
   - Create implementation in `src/sparsification/your_method.py`
   - Inherit from `BaseSparsifier`
   - Create config in `configs/sparsifier/your_method.yaml`

2. **Add a new model:**
   - Create implementation in `src/models/your_model.py`
   - Create config in `configs/model/your_model.yaml`

3. **Create experiment preset:**
   - Create config in `configs/experiment/your_experiment.yaml`
   - Define default components and parameters

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_sparsification.py
```

## Next Steps

1. **Complete Model Implementations:**
   - Implement `src/models/gcn.py`
   - Implement `src/models/gat.py`

2. **Complete Training Loop:**
   - Uncomment and complete functions in `scripts/train.py`
   - Test with a simple experiment

3. **Enable W&B Logging:**
   - Uncomment W&B initialization in `scripts/train.py`
   - Set your W&B entity in config

4. **Refine Sparsification Methods:**
   - Implement actual spectral method in `src/sparsification/spectral.py`
   - Add more sophisticated metrics in `src/sparsification/metric.py`

## Benefits of New Structure

✅ **Modularity:** Easy to swap components (models, datasets, sparsifiers)
✅ **Reproducibility:** Config files ensure exact experiment reproduction
✅ **Scalability:** Clean separation enables parallel development
✅ **Testability:** Unit tests prevent regressions
✅ **Collaboration:** Clear structure for team development
✅ **MLOps Ready:** Integration with W&B, Hydra for production workflows
