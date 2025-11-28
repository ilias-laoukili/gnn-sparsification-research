# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/gnn-sparsification-research.git
cd gnn-sparsification-research
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .  # Install package in editable mode
```

## Running Your First Experiment

### 1. Basic Training (TODO: Complete model implementations first)

```bash
# Run baseline experiment with default settings
python scripts/train.py

# This will:
# - Load Cora dataset
# - Apply random sparsification (50% edges)
# - Train GCN model
# - Log results to W&B (if enabled)
```

### 2. Comparing Sparsification Methods

```bash
# Random baseline
python scripts/train.py sparsifier=random

# Jaccard similarity
python scripts/train.py sparsifier=jaccard

# Spectral method
python scripts/train.py sparsifier=spectral
```

### 3. Testing Different Models

```bash
# GCN
python scripts/train.py model=gcn

# GAT
python scripts/train.py model=gat
```

### 4. Experimenting with Datasets

```bash
# Cora
python scripts/train.py dataset=cora

# PubMed
python scripts/train.py dataset=pubmed
```

### 5. Hyperparameter Tuning

```bash
# Adjust sparsification ratio
python scripts/train.py sparsifier.sparsification_ratio=0.3

# Adjust learning rate and hidden dimensions
python scripts/train.py training.learning_rate=0.001 model.architecture.hidden_dim=128

# Combine multiple overrides
python scripts/train.py model=gat dataset=pubmed sparsifier=jaccard \
    sparsifier.sparsification_ratio=0.4 \
    training.learning_rate=0.005 \
    model.architecture.hidden_dim=128 \
    model.architecture.heads=4
```

## Using Experiment Presets

Experiment presets combine multiple configuration choices:

```bash
# Baseline experiment
python scripts/train.py experiment=baseline

# Jaccard on Cora
python scripts/train.py experiment=jaccard_cora
```

## Project Structure Overview

```
├── configs/              # All experiment configurations
│   ├── config.yaml       # Main config (entry point)
│   ├── model/            # Model architectures (gcn.yaml, gat.yaml)
│   ├── dataset/          # Dataset configs (cora.yaml, pubmed.yaml)
│   ├── sparsifier/       # Sparsification methods
│   └── experiment/       # Complete experiment presets
│
├── scripts/
│   └── train.py          # Main training script
│
├── src/
│   ├── sparsification/   # Your core research module
│   │   ├── base.py       # Abstract base class
│   │   ├── random.py     # Random baseline
│   │   ├── spectral.py   # Spectral methods
│   │   └── metric.py     # Similarity-based methods
│   ├── models/           # GNN architectures (TODO: implement)
│   ├── data/             # Data loading utilities
│   └── utils/            # Metrics, logging, etc.
│
└── notebooks/
    ├── exploratory/      # Your research notebooks
    └── tutorials/        # Learning materials
```

## Next Steps

1. **Implement Models:** Complete `src/models/gcn.py` and `src/models/gat.py`
2. **Complete Training:** Uncomment training loop in `scripts/train.py`
3. **Run Tests:** `pytest tests/` to verify sparsification methods
4. **Enable W&B:** Set up Weights & Biases for experiment tracking

## Testing Sparsification Methods (Works Now!)

You can test the sparsification methods immediately:

```python
from torch_geometric.datasets import Planetoid
from src.sparsification.random import RandomSparsifier
from src.sparsification.metric import MetricSparsifier

# Load dataset
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# Test random sparsification
sparsifier = RandomSparsifier(sparsification_ratio=0.5, seed=42)
sparse_data = sparsifier.sparsify(data)

# View statistics
stats = sparsifier.get_sparsification_stats(data, sparse_data)
print(stats)

# Test Jaccard sparsification
sparsifier = MetricSparsifier(sparsification_ratio=0.5, metric='jaccard')
sparse_data = sparsifier.sparsify(data)
```

## Getting Help

- Check `docs/MIGRATION.md` for detailed migration information
- See `configs/config.yaml` for all configuration options
- Read docstrings in `src/sparsification/base.py` for API details
- Refer to notebooks in `notebooks/tutorials/` for PyG basics
