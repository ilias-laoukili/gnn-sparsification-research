# Graph Sparsification for Graph Neural Networks

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

## Introduction

This repository contains the research work for the **Projet TREMPLIN RECHERCHE 2025/2026** at **University Gustave Eiffel / LAMA**.

**Tutor:** Maximilien Dreveton

The primary goal of this project is to investigate the trade-off between Graph Neural Network (GNN) performance (Accuracy) and computational efficiency by sparsifying graphs using metric backbones. We aim to reduce the number of edges in the graph while maintaining high classification accuracy.

## Methodology

We propose a sparsification method based on the **Metric Backbone** intuition. The process involves:

1.  **Defining Edge Weights:** We assign weights to edges based on the Jaccard Distance of the neighborhoods of the connected nodes $u$ and $v$:

    $$ d(u,v) = 1 - \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|} $$

2.  **Filtering Edges:** We filter the edges to preserve the shortest path metric of the weighted graph. Edges that are not essential for maintaining the shortest path distances between nodes are considered redundant and removed.

## Tech Stack

*   **Language:** Python
*   **Libraries:** PyTorch Geometric (PyG), NetworkX, PyTorch
*   **Datasets:** Cora (Current), Citeseer/PubMed (Planned)

## Setup & Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gnn-sparsification-research.git
    cd gnn-sparsification-research
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For `torch-scatter` and `torch-sparse`, you may need to install specific wheels compatible with your CUDA version. See [PyG Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details.*

## Usage

To run the main analysis notebook:

```bash
jupyter notebook Notes.ipynb
```

Or if you have a python script:

```python
# Example usage in a script
from model import GCN
from sparsification import sparsify_graph

# Load data, sparsify, and train
data = load_cora()
data = sparsify_graph(data)
train_model(data)
```

## Preliminary Results

*   **Baseline (Full Graph):** ~80% Accuracy on Cora (GCN)
*   **Sparsified Graph:** Experiments are ongoing to determine the impact of Jaccard-based sparsification on accuracy and training time.

---
*This project is part of an academic research initiative.*
