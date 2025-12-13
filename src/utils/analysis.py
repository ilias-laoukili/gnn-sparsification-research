"""Analysis helpers shared between notebooks and scripts.

This module centralizes lightweight data-processing helpers so notebooks
can stay focused on visualization and reporting.
"""

from collections import Counter
import os
from typing import Dict, TYPE_CHECKING

import pandas as pd
import torch
from torch_geometric.data import Data

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from ..experiments import AblationStudy
    from ..data import DatasetLoader


def compute_effects(df_subset: pd.DataFrame) -> Dict[str, float]:
    """Decompose structure/weighting/interaction effects for a scenario row set."""
    a = df_subset[df_subset["Scenario"] == "A: Full + Binary"]["Accuracy"].values[0]
    b = df_subset[df_subset["Scenario"] == "B: Sparse + Binary"]["Accuracy"].values[0]
    c = df_subset[df_subset["Scenario"] == "C: Full + Weighted"]["Accuracy"].values[0]
    d = df_subset[df_subset["Scenario"] == "D: Sparse + Weighted"]["Accuracy"].values[0]
    return {
        "Baseline (A)": a,
        "Structure Effect (B-A)": b - a,
        "Weighting Effect (C-A)": c - a,
        "Combined Effect (D-A)": d - a,
        "Interaction (D-B-C+A)": d - b - c + a,
    }


def compute_graph_stats(data: Data) -> Dict[str, float]:
    """Compute basic graph topology statistics for a PyG ``Data`` object."""
    edge_index = data.edge_index.cpu().numpy()
    degrees = Counter(edge_index[0])
    degree_values = list(degrees.values())

    return {
        "num_nodes": data.num_nodes,
        "num_edges": edge_index.shape[1],
        "avg_degree": float(pd.Series(degree_values).mean()),
        "median_degree": float(pd.Series(degree_values).median()),
        "max_degree": int(pd.Series(degree_values).max()),
        "min_degree": int(pd.Series(degree_values).min()),
        "std_degree": float(pd.Series(degree_values).std()),
        "density": edge_index.shape[1] / (data.num_nodes * (data.num_nodes - 1)),
    }


def retention_to_numeric(retention_str: str) -> float:
    """Convert percentage strings like ``"90%"`` to numeric ratios (0.9)."""
    return float(retention_str.rstrip("%")) / 100.0


def run_ablation_config(
    ds: str,
    metric: str,
    ratio: float,
    loader: "DatasetLoader",
    device: str,
    worker_threads: int = 1,
    model_name: str = "gcn",
    hidden_channels: int = 64,
    epochs: int = 200,
    patience: int = 20,
) -> pd.DataFrame:
    """Run a single (dataset, metric, retention) ablation config and return results."""
    # Import here to avoid circular import at module load time
    from ..experiments import AblationStudy
    
    # Scope thread limits to the worker process
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(worker_threads)
    torch.set_num_threads(worker_threads)

    data_ds, nf, nc = loader.get_dataset(ds, device)
    study_ds = AblationStudy(data=data_ds, num_features=nf, num_classes=nc, device=device)
    results_df = study_ds.run_multi_config_study(
        model_names=[model_name],
        metrics=[metric],
        retention_ratios=[ratio],
        hidden_channels=hidden_channels,
        epochs=epochs,
        patience=patience,
    )
    results_df["Dataset"] = ds.title()
    return results_df
