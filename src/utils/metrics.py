"""Evaluation metrics for graph neural networks."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@torch.no_grad()
def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor
) -> float:
    """Compute classification accuracy.
    
    Args:
        logits: Model predictions (N, num_classes)
        labels: Ground truth labels (N,)
        mask: Boolean mask for which nodes to evaluate
        
    Returns:
        Accuracy as a float
    """
    pred = logits.argmax(dim=1)
    correct = (pred[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_node_classification(
    model: torch.nn.Module,
    data: Data
) -> Dict[str, float]:
    """Evaluate node classification model on all splits.
    
    Args:
        model: PyTorch GNN model
        data: PyTorch Geometric Data object
        
    Returns:
        Dictionary with train/val/test accuracies
    """
    model.eval()
    out = model(data.x, data.edge_index)
    
    results = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        results[f'{split}_acc'] = compute_accuracy(out, data.y, mask)
    
    return results
