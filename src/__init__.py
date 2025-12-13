"""GNN Sparsification Research Package.

A modular PyTorch Geometric framework for researching graph sparsification methods
and their impact on Graph Neural Network performance.
"""

from .data import DatasetLoader
from .experiments import AblationStudy, ExperimentScenario
from .models import GAT, GCN, GraphSAGE, get_model
from .sparsification import GraphSparsifier
from .sparsification.metrics import (
    calculate_adamic_adar_scores,
    calculate_jaccard_scores,
)
from .training import EarlyStopper, GNNTrainer
from .utils import (
    set_global_seed,
    compute_effects,
    compute_graph_stats,
    retention_to_numeric,
    run_ablation_config,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetLoader",
    "GCN",
    "GraphSAGE",
    "GAT",
    "get_model",
    "GraphSparsifier",
    "calculate_jaccard_scores",
    "calculate_adamic_adar_scores",
    "GNNTrainer",
    "EarlyStopper",
    "AblationStudy",
    "ExperimentScenario",
    "set_global_seed",
    "compute_effects",
    "compute_graph_stats",
    "retention_to_numeric",
    "run_ablation_config",
]
