"""Training loops, trainers, and optimization utilities."""

from .trainer import EarlyStopper, GNNTrainer

# hpo_helpers is NOT imported here to avoid circular imports
# (hpo_helpers → hpo.config → hpo/__init__ chain).
# Import directly: from src.training.hpo_helpers import train_val_acc, evaluate_hp

__all__ = ["GNNTrainer", "EarlyStopper"]
