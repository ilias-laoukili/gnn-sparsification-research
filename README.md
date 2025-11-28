# Graph Sparsification for Graph Neural Networks

![Status](https://img.shields.io/badge/Status-In%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## Introduction

This repository contains the research work for the **Projet TREMPLIN RECHERCHE 2025/2026** at **Université Gustave Eiffel / LAMA**.

**Tutor:** Maximilien Dreveton

The primary goal of this project is to investigate the trade-off between Graph Neural Network (GNN) performance and computational efficiency by sparsifying graphs using various methods including metric backbones. We aim to reduce the number of edges in the graph while maintaining high classification accuracy.

## 🎯 Research Objectives

- **Compare** different graph sparsification methods (Random, Spectral, Metric-based)
- **Analyze** the impact of sparsification on GNN accuracy and training time
- **Identify** optimal sparsification ratios for different datasets
- **Understand** which graph properties are most important to preserve

## 📐 Methodology

We investigate multiple sparsification approaches:

### 1. Metric Backbone (Jaccard-based)

Edge weights based on the Jaccard Distance of node neighborhoods:

$$ d(u,v) = 1 - \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|} $$

Edges that don't preserve shortest path metrics are removed.

### 2. Spectral Methods

Preserve spectral properties of the graph Laplacian (under development).

### 3. Random Baseline

Random edge removal for comparison baseline.

## 🏗️ Project Structure

```
gnn-sparsification-research/
├── configs/              # Hydra configuration files
│   ├── experiment/       # Experiment presets
│   ├── model/            # Model configs (GCN, GAT)
│   ├── dataset/          # Dataset configs (Cora, PubMed)
│   └── sparsifier/       # Sparsification methods
├── src/                  # Main source code
│   ├── sparsification/   # Core sparsification module
│   ├── models/           # GNN implementations
│   ├── data/             # Data loading utilities
│   ├── training/         # Training loops
│   └── utils/            # Metrics and utilities
├── scripts/              # Entry point scripts
│   ├── train.py          # Main training script
│   ├── demo_sparsification.py
│   └── verify_setup.py
├── notebooks/            # Jupyter notebooks
│   ├── exploratory/      # Research experiments
│   └── tutorials/        # Learning materials
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/ilias-laoukili/gnn-sparsification-research.git
cd gnn-sparsification-research

# Run automated setup
./setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Installation

```bash
python scripts/verify_setup.py
```

### 3. Test Sparsification Methods

```bash
# Run demonstration
python scripts/demo_sparsification.py

# Run unit tests
pytest tests/
```

### 4. Run Experiments (TODO: Complete model implementations first)

```bash
# Run with default configuration
python scripts/train.py

# Override components
python scripts/train.py model=gat dataset=pubmed sparsifier=jaccard

# Use experiment preset
python scripts/train.py experiment=baseline

# Adjust parameters
python scripts/train.py sparsifier.sparsification_ratio=0.3 training.learning_rate=0.001
```

### 5. Using Makefile

```bash
make help       # Show available commands
make test       # Run tests
make format     # Format code
make train      # Run training
```

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get up and running quickly
- **[Migration Guide](docs/MIGRATION.md)** - Understand the refactoring
- **[Architecture](docs/ARCHITECTURE.md)** - Deep dive into system design
- **[Refactoring Summary](REFACTORING_SUMMARY.md)** - Complete refactoring details

## 🧪 Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** PyTorch 2.0+, PyTorch Geometric
- **Configuration:** Hydra, OmegaConf
- **Experiment Tracking:** Weights & Biases
- **Testing:** pytest
- **Datasets:** Planetoid (Cora, CiteSeer, PubMed), extensible to OGB

## 🔬 Sparsification Methods

### Currently Implemented:

- ✅ **RandomSparsifier** - Baseline random edge removal
- ✅ **MetricSparsifier** - Jaccard, Cosine, Euclidean similarity
- 🚧 **SpectralSparsifier** - Spectral graph theory (placeholder)

### API Example:

```python
from torch_geometric.datasets import Planetoid
from src.sparsification.metric import MetricSparsifier

# Load dataset
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# Apply sparsification
sparsifier = MetricSparsifier(
    sparsification_ratio=0.5,
    metric='jaccard'
)
sparse_data = sparsifier.sparsify(data)

# View statistics
stats = sparsifier.get_sparsification_stats(data, sparse_data)
print(stats)
```

## 📊 Preliminary Results

- **Baseline (Full Graph):** ~80% Accuracy on Cora (GCN)
- **Sparsified Graphs:** Experiments ongoing

Full results will be updated as experiments complete.

## 🛠️ Development Status

### ✅ Completed:
- Core sparsification module with abstract base class
- Three sparsification methods (Random, Metric, Spectral placeholder)
- Hydra configuration system
- Project structure refactoring
- Unit tests for sparsification
- Documentation (Quick Start, Architecture, Migration)

### 🚧 In Progress:
- GNN model implementations (GCN, GAT)
- Training loop completion
- Weights & Biases integration
- Comprehensive experiments

### 📋 Planned:
- Additional sparsification methods
- Link prediction tasks
- Graph-level classification
- Hyperparameter optimization
- Extended benchmarks on OGB datasets

## 🤝 Contributing

This is an academic research project. For questions or suggestions:

1. Check existing documentation in `docs/`
2. Review code docstrings for API details
3. Run tests before submitting changes

## 📄 License

This project is part of an academic research initiative at Université Gustave Eiffel.

## 🙏 Acknowledgments

- **Tutor:** Maximilien Dreveton
- **Institution:** Université Gustave Eiffel / LAMA
- **Program:** Projet TREMPLIN RECHERCHE 2025/2026

## 📞 Contact

For questions about this research:
- Open an issue on GitHub
- Contact through the university

---

**Note:** This project recently underwent a major refactoring to adopt professional MLOps standards. See [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) for details.
