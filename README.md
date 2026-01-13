# Graph Sparsification for Graph Neural Networks

![Status](https://img.shields.io/badge/Status-In%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT%20%2B%20CC--BY-lightgrey)

## 🚦 Project Status

**Phase:** Research Validation (Pre-Publication)

### What Works ✅
- ✅ Exploratory notebooks with validated results (see [`notebooks/exploratory/`](notebooks/exploratory))
- ✅ Jaccard/Adamic-Adar similarity-based sparsification
- ✅ Experiments on Cora, PubMed, Flickr datasets
- ✅ Comprehensive unit tests for all sparsification methods
- ✅ Statistical analysis tools (confidence intervals, significance tests)

### What's Being Fixed 🚧
- 🚧 True metric backbone implementation (RTI filtering) - **COMPLETED**
- 🚧 Effective resistance approximation performance - **COMPLETED**
- 🚧 Baseline comparison methods (degree, betweenness, DropEdge) - **COMPLETED**
- 🚧 Unit tests and CI/CD pipeline - **IN PROGRESS**
- 🚧 Statistical significance testing - **COMPLETED**

### How to Reproduce Results

```bash
# Option 1: Run notebooks (recommended for exploration)
jupyter notebook notebooks/exploratory/

# Option 2: Run automated experiments (reproducibility)
bash scripts/reproduce_experiments.sh

# Option 3: Run specific dataset/method
bash scripts/reproduce_experiments.sh --dataset cora --method jaccard
```

## Introduction

This repository contains the research work for the **Projet TREMPLIN RECHERCHE 2025/2026** at **Université Gustave Eiffel / LAMA**.

**Tutor:** Maximilien Dreveton

The primary goal of this project is to investigate the trade-off between Graph Neural Network (GNN) performance and computational efficiency by sparsifying graphs using various methods, specifically focusing on **Metric Backbones**. We aim to reduce the number of edges in the graph (compression) while maintaining high classification accuracy (performance).

## 🎯 Research Objectives

- **Compare** different graph sparsification methods (Metric-based vs. Random baseline).
- **Analyze** the impact of sparsification on GNN accuracy and training time.
- **Identify** optimal sparsification ratios for different datasets (Citation vs. Social).
- **Understand** the topological properties preserved by different metrics (Jaccard vs. Adamic-Adar).

## 📐 Methodology

We investigate multiple sparsification approaches based on the **Metric Backbone** framework (Relaxed Triangle Inequality):

### 1. Jaccard Metric Backbone
Used for citation networks (homophily-based). Edge weights are based on neighborhood overlap:
$$d(u,v) = 1 - \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

### 2. Adamic-Adar Metric Backbone
Used for social networks. Weights common neighbors by their rarity (inverse log degree):
$$AA(u,v) = \sum_{z \in N(u) \cap N(v)} \frac{1}{\log(|N(z)|)}$$

### 3. Effective Resistance (Spectral)
Measures edge criticality based on electrical network analogy. High R_eff = bottleneck edge:
$$R_{eff}(u,v) = L^+_{uu} + L^+_{vv} - 2L^+_{uv}$$

**Two implementations available:**
- **Exact** (`effective_resistance`): Dense O(N³) for small benchmarks (<5K nodes)
- **Approximate** (`approx_er`): JLT-based O(m log n) for large graphs (>10K nodes)

### 4. Random Baseline
Random edge removal to establish a performance lower bound.

## 🏗️ Project Structure

```text
gnn-sparsification-research/
├── notebooks/
│   └── exploratory/      # ✅ CORE RESEARCH EXPERIMENTS
│       ├── 01_exp_jaccard_sparsification_cora.ipynb
│       ├── 02_exp_jaccard_sparsification_pubmed.ipynb
│       └── 03_exp_adamic_adar_sparsification_flickr.ipynb
├── src/                  # (In Progress) Modular source code
├── scripts/              # (In Progress) Training scripts
├── configs/              # (In Progress) Hydra configs
├── docs/                 # Documentation & Reports
└── tests/                # Unit tests
```

## 🚀 Reproducing Research Results

The research experiments are available as standalone Jupyter Notebooks. To replicate the findings:

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

Launch Jupyter Lab or Notebook:

```bash
jupyter notebook notebooks/exploratory/
```

#### **Experiment 1: Small Scale Baseline (Cora)**

- **Notebook:** `01_exp_jaccard_sparsification_cora.ipynb`
- **Method:** Jaccard Metric Backbone (Dense Matrix Implementation).
- **Finding:** High compressibility. We can remove **~40%** of edges with negligible accuracy loss.

#### **Experiment 2: Scalability & Saturation (PubMed)**

- **Notebook:** `02_exp_jaccard_sparsification_pubmed.ipynb`
- **Method:** Sparse Tensor Algebra ($O(E)$ complexity).
- **Finding:** Discovery of the **"Saturation Effect"**. Edge retention stabilizes at **~65%** regardless of the stretch factor $\alpha$, suggesting a rigid topological "Hard Core" in citation graphs.

#### **Experiment 3: Social Networks & Optimization (Flickr)**

- **Notebook:** `03_exp_adamic_adar_sparsification_flickr.ipynb`
- **Method:** Adamic-Adar Metric Backbone.
- **Engineering:** Achieved **1000x speedup** in metric computation using CPU-vectorized sparse operations (`scipy.sparse`) to enable processing 900k edges on consumer hardware (Mac M4) without OOM.

## 📊 Key Research Findings

| Dataset | Type | Nodes | Edges | Metric | Key Insight |
|---------|------|-------|-------|--------|-------------|
| **Cora** | Citation | 2.7K | 5K | Jaccard | **High Redundancy:** 40% edges removed with <1% acc drop. |
| **PubMed** | Citation | 19.7K | 44K | Jaccard | **Hard Core:** Sparsification saturates at 65% retention. |
| **Flickr** | Social | 89K | 900K | Adamic-Adar | **Structure:** Adamic-Adar effectively prunes high-degree hubs. |

### Effective Resistance Approximation Quality

The JLT-based approximation preserves rankings with Spearman ρ ≈ 0.80:

| Metric | Exact ER | Approx ER (ε=0.3) |
|--------|----------|-------------------|
| Time (Cora) | 2.6s | 40s |
| Time (Flickr) | ∞ (OOM) | ~5 min |
| Spearman ρ | 1.0 | 0.80 |
| Top-10% overlap | 100% | 77% |

**Note:** Approx ER is slower on small graphs but essential for large graphs where O(N³) is infeasible.

## 🛠️ Development Status

**Current Phase:** Transitioning from *Exploratory Notebooks* to *Modular Python Package*.

### ✅ Completed

- [x] Validated Jaccard Metric Backbone on small dense graphs.
- [x] Implemented Sparse Tensor algebra for medium graphs.
- [x] Implemented Adamic-Adar Metric Backbone for social graphs.
- [x] Optimized memory usage for Apple Silicon (MPS/M4).

### 🚧 In Progress

- [ ] Refactoring notebook logic into reusable `src/` modules.
- [ ] Finalizing `Hydra` configuration integration.
- [ ] Standardizing the `train.py` CLI entry point.

### 📋 Future Work

- **Spectral Methods:** Preserving graph Laplacian eigenvalues.
- **OGB Datasets:** Scaling to millions of nodes (Arxiv, Products).

## 📄 License

This repository uses a **Dual License** model:

1. **Code:** The source code and notebooks are licensed under the **MIT License**.
2. **Report & Figures:** The written report (`docs/`) and generated figures are licensed under **CC-BY 4.0**.

See [LICENSE](LICENSE) for details.

## 🤝 Acknowledgments

- **University:** Université Gustave Eiffel / LAMA
- **Program:** Projet TREMPLIN RECHERCHE 2025/2026
- **Tutor:** Maximilien Dreveton
