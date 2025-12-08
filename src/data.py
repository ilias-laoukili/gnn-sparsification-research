"""Dataset loading utilities for graph neural network experiments."""

from typing import Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


class DatasetLoader:
    """Unified interface for loading graph datasets.

    Provides a consistent API for loading various graph datasets with
    proper train/val/test splits and automatic device placement.

    Args:
        root: Root directory where datasets will be downloaded/stored.

    Attributes:
        root: Dataset root directory path.

    Example:
        >>> loader = DatasetLoader(root="data")
        >>> data, num_features, num_classes = loader.get_dataset("cora", "cuda")
        >>> print(f"Loaded graph with {data.num_nodes} nodes")
    """

    def __init__(self, root: str = "data") -> None:
        self.root = root

    def get_dataset(
        self, name: str, device: str = "cpu"
    ) -> Tuple[Data, int, int]:
        """Load a dataset by name and move to specified device.

        Args:
            name: Dataset name (e.g., 'cora', 'pubmed', 'citeseer').
            device: Target device for the data ('cpu', 'cuda', or 'mps').

        Returns:
            Tuple of (data, num_features, num_classes) where:
                - data: PyG Data object with node features and edges
                - num_features: Dimensionality of node features
                - num_classes: Number of target classes

        Raises:
            ValueError: If dataset name is not supported.

        Example:
            >>> loader = DatasetLoader()
            >>> data, n_features, n_classes = loader.get_dataset("cora")
        """
        name_lower = name.lower()

        if name_lower in ["cora", "pubmed", "citeseer"]:
            dataset = Planetoid(
                root=self.root,
                name=name_lower.capitalize(),
            )
            data = dataset[0].to(device)
            num_features = dataset.num_features
            num_classes = dataset.num_classes
        else:
            raise ValueError(
                f"Unsupported dataset: {name}. "
                f"Supported datasets: cora, pubmed, citeseer"
            )

        return data, num_features, num_classes
