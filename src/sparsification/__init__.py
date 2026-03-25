"""Graph sparsification methods and utilities."""

from .core import GraphSparsifier
from .random import precompute_random_scores, random_sparsify
from .metrics import (
    calculate_adamic_adar_scores,
    calculate_approx_effective_resistance_scores,
    calculate_effective_resistance_scores,
    calculate_feature_cosine_scores,
    calculate_jaccard_scores,
    compute_geodesic_preservation,
    compute_topology_metrics,
    compute_topology_preservation,
)

__all__ = [
    "GraphSparsifier",
    "calculate_jaccard_scores",
    "calculate_adamic_adar_scores",
    "calculate_effective_resistance_scores",
    "calculate_approx_effective_resistance_scores",
    "calculate_feature_cosine_scores",
    "compute_geodesic_preservation",
    "compute_topology_metrics",
    "compute_topology_preservation",
    "precompute_random_scores",
    "random_sparsify",
]
