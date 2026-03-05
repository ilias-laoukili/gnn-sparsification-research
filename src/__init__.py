"""GNN Sparsification Research Package.

A modular PyTorch Geometric framework for researching graph sparsification methods
and their impact on Graph Neural Network performance.

Public API
----------
Data loading:
    DatasetLoader

Models:
    GCN, GCNStar, GraphSAGE, GAT, get_model

Sparsification:
    GraphSparsifier
    calculate_jaccard_scores
    calculate_adamic_adar_scores
    calculate_approx_effective_resistance_scores
    calculate_effective_resistance_scores
    calculate_feature_cosine_scores

Training:
    GNNTrainer, EarlyStopper

Utilities:
    set_global_seed, compute_graph_stats, print_text_table

Internal submodules (not re-exported here):
    src.experiments  — AblationStudy, ExperimentScenario
    src.utils        — compute_effects, run_ablation_config, retention_to_numeric
"""

from .data import DatasetLoader
from .models import GAT, GCN, GCNStar, GraphSAGE, get_model
from .sparsification import (
    GraphSparsifier,
    calculate_adamic_adar_scores,
    calculate_approx_effective_resistance_scores,
    calculate_effective_resistance_scores,
    calculate_feature_cosine_scores,
    calculate_jaccard_scores,
)
from .training import EarlyStopper, GNNTrainer
from .utils import compute_graph_stats, print_text_table, set_global_seed

__version__ = "0.1.0"

__all__ = [
    # Data
    "DatasetLoader",
    # Models
    "GCN",
    "GCNStar",
    "GraphSAGE",
    "GAT",
    "get_model",
    # Sparsification
    "GraphSparsifier",
    "calculate_jaccard_scores",
    "calculate_adamic_adar_scores",
    "calculate_approx_effective_resistance_scores",
    "calculate_effective_resistance_scores",
    "calculate_feature_cosine_scores",
    # Training
    "GNNTrainer",
    "EarlyStopper",
    # Utilities
    "set_global_seed",
    "compute_graph_stats",
    "print_text_table",
]
