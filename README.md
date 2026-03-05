# Graph Sparsification for Graph Neural Networks

![Status](https://img.shields.io/badge/Status-In%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

This project investigates graph sparsification methods for improving GNN efficiency while preserving accuracy, as part of **Projet TREMPLIN RECHERCHE 2025/2026** at **Université Gustave Eiffel / LAMA**.

**Tutor:** Maximilien Dreveton

The central question: *can we remove a large fraction of edges from a graph before training a GNN, without significantly hurting accuracy — and can smart sparsification methods outperform random edge removal?*

We evaluate multiple sparsification strategies on both homophilic (citation) and heterophilic (Roman-empire) graphs, using GCN* (Luo et al., NeurIPS 2024) as the backbone model.

---

## Project Structure

```
gnn-sparsification-research/
├── src/                          # Core library
│   ├── data/                     # Dataset loading (26+ datasets via PyG)
│   ├── models/                   # GCN, GraphSAGE, GAT, GCNStar
│   ├── sparsification/           # Sparsification engine + metrics
│   ├── training/                 # GNNTrainer with early stopping
│   ├── experiments/              # Ablation study framework
│   └── utils/                    # Seeds, analysis, reporting
├── scripts/
│   ├── run_ablation.py           # Ablation study — single dataset (CPU)
│   ├── run_ablation_parallel.sh  # Ablation study — all datasets in parallel
│   └── roman_empire_gpu.py       # Roman-empire experiment (GPU / Kaggle)
├── notebooks/exploratory/
│   ├── 01_Dataset_Loading_and_Preprocessing.ipynb
│   ├── 02_Sparsification_Algorithm_Implementation.ipynb
│   ├── 03_Effect_of_Sparsification_on_Topology.ipynb
│   ├── 04_Ablation_Study_and_Visualization.ipynb
│   └── 05_Roman_Empire_Analysis.ipynb
├── tests/                        # Unit tests
├── presentation/                 # LaTeX slides + figures
└── report/                       # LaTeX research report
```

---

## Sparsification Methods

### Threshold-based (keep top-k edges by score)
| Method | Score | Notes |
|--------|-------|-------|
| **Jaccard-T** | Neighbourhood overlap | Good for homophilic graphs |
| **AA-T** | Adamic-Adar (log-weighted overlap) | Penalises high-degree hubs |
| **FeatCos-T** | Feature cosine similarity | Feature-aware, not topology-based |
| **ApproxER-T** | Approximate effective resistance | Spectral, O(m log n) via JLT |
| **Random** | Uniform random | Baseline lower bound |

Each method also has an **inverse** variant (`-IT`: keep lowest-scoring edges) and a **weighted** variant (`-W`: use scores as edge weights during training).

Additional variants: **sampled** (probabilistic proportional to score) and **degree-aware** (per-node minimum edge guarantee).

### Metric Backbone
Global edge pruning via the Relaxed Triangle Inequality — preserves geodesic distances. Implemented in `src/sparsification/metric_backbone.py`.

---

## Model: GCN* (NeurIPS 2024)

We replicate **GCN\*** from *"Classic GNNs are Strong Baselines"* (Luo et al., NeurIPS 2024) exactly:

```
Input → Linear(300, 512) → 9 × [GCNConv + learned residual + BatchNorm + ReLU + Dropout(0.5)] → Linear(512, 18)
```

Hyperparameters (from paper): `hidden=512, layers=9, dropout=0.5, lr=1e-3, wd=0.0, epochs=2500`
Paper result on Roman-empire: **91.27 ± 0.20%**

---

## Experiments

### Notebooks (01–04): Understanding Sparsification

The first four notebooks build up understanding of sparsification methods in general:

1. **Dataset loading** — inspect graph statistics across 26+ datasets
2. **Algorithm implementation** — implement and test Jaccard/AA metrics
3. **Topological effects** — degree distribution, connectivity, clustering under sparsification
4. **Ablation study** — four scenarios (Full+Binary, Sparse+Binary, Full+Weighted, Sparse+Weighted) on homophilic datasets

### Notebook 05 + Script: Roman-empire Application

Applies sparsification to Roman-empire, a challenging **heterophilic** graph (22,662 nodes, 65,854 edges, homophily ≈ 0.047, 18 classes).

Run on GPU (Kaggle T4×2, ~3.5h for non-ApproxER methods):
```bash
# Session 1 — fast methods (~3.5h on T4×2)
python scripts/roman_empire_gpu.py --skip-approxer

# Session 2 — ApproxER methods (~8h on T4×2), after uploading session 1 results
python scripts/roman_empire_gpu.py --approxer-only --resume
```

### Ablation Study (multi-dataset)

```bash
# Single dataset
python scripts/run_ablation.py --dataset cora

# All datasets in parallel waves
bash scripts/run_ablation_parallel.sh
```

---

## Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"
```

Dependencies are declared in `pyproject.toml`. Key packages: `torch`, `torch-geometric`, `scikit-learn`, `numpy`, `pandas`, `scipy`.

---

## Tests

```bash
pytest tests/
```

---

## License

MIT. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Université Gustave Eiffel / LAMA
- Projet TREMPLIN RECHERCHE 2025/2026
- Tutor: Maximilien Dreveton
- Luo et al., *"Classic GNNs are Strong Baselines"*, NeurIPS 2024
- Platonov et al., *"A Critical Look at the State of Self-Supervised Learning for Graphs"* (Roman-empire dataset)
