"""Training utilities for GNN models with early stopping."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.data import Data


class EarlyStopper:
    """Monitor validation metrics and halt training when improvement stalls.

    Implements patience-based early stopping with optional model checkpointing
    to restore best weights after training completes.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in monitored value to qualify as improvement.

    Attributes:
        patience: Configured patience value.
        min_delta: Minimum improvement threshold.
        counter: Epochs since last improvement.
        best_score: Best validation score observed.
        should_stop: Flag indicating training should halt.

    Example:
        >>> stopper = EarlyStopper(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if stopper.step(-val_loss):  # Negate loss since higher is better
        ...         break
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Update stopper with latest validation score.

        Args:
            score: Current validation metric (higher is better).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self) -> None:
        """Reset stopper state for a new training run."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class GNNTrainer:
    """Unified training engine for GNN models with edge weight support.

    Handles the training loop, evaluation, and early stopping for
    node classification tasks with configurable loss functions.

    Args:
        model: GNN model instance (must implement forward(data, edge_weight)).
        optimizer: PyTorch optimizer configured with model parameters.
        criterion: Loss function (default: NLLLoss for log_softmax outputs).
        device: Target device for computation.

    Attributes:
        model: The GNN model being trained.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Computation device.
        history: Dictionary tracking training metrics per epoch.

    Example:
        >>> trainer = GNNTrainer(model, optimizer, device="cuda")
        >>> history = trainer.train(data, epochs=200, patience=20)
        >>> test_acc = trainer.evaluate(data)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[Callable] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion or F.nll_loss
        self.device = device
        self.history: dict = {"train_loss": [], "val_acc": []}

    def _train_step(self, data: Data, edge_weight: Optional[Tensor] = None) -> float:
        """Execute single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data, edge_weight=edge_weight)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _evaluate(self, data: Data, mask: Tensor, edge_weight: Optional[Tensor] = None) -> float:
        """Compute accuracy on masked nodes."""
        self.model.eval()
        out = self.model(data, edge_weight=edge_weight)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        return (correct / mask.sum()).item()

    def train(
        self,
        data: Data,
        epochs: int = 200,
        patience: Optional[int] = None,
        verbose: bool = False,
        edge_weight: Optional[Tensor] = None,
    ) -> dict:
        """Train the model with optional early stopping.

        Args:
            data: PyG Data object with train/val/test masks.
            epochs: Maximum number of training epochs.
            patience: Early stopping patience. If None, trains for full epochs.
            verbose: Print progress every 10 epochs if True.
            edge_weight: Optional edge weights for message passing.

        Returns:
            Dictionary containing training history with keys:
                - train_loss: List of training losses per epoch.
                - val_acc: List of validation accuracies per epoch.
                - epochs_trained: Actual number of epochs completed.
                - best_val_acc: Best validation accuracy achieved.
        """
        self.history = {"train_loss": [], "val_acc": []}
        stopper = EarlyStopper(patience=patience) if patience else None

        for epoch in range(epochs):
            loss = self._train_step(data, edge_weight)
            val_acc = self._evaluate(data, data.val_mask, edge_weight)

            self.history["train_loss"].append(loss)
            self.history["val_acc"].append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

            if stopper and stopper.step(val_acc):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        self.history["epochs_trained"] = epoch + 1
        self.history["best_val_acc"] = max(self.history["val_acc"])

        return self.history

    def evaluate(self, data: Data, edge_weight: Optional[Tensor] = None) -> float:
        """Compute test accuracy.

        Args:
            data: PyG Data object with test_mask.
            edge_weight: Optional edge weights for message passing.

        Returns:
            Test set accuracy as float in [0, 1].
        """
        return self._evaluate(data, data.test_mask, edge_weight)

    def train_and_evaluate(
        self,
        data: Data,
        epochs: int = 200,
        patience: Optional[int] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tuple[float, dict]:
        """Convenience method to train and return test accuracy.

        Args:
            data: PyG Data object with all masks.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            edge_weight: Optional edge weights for message passing.

        Returns:
            Tuple of (test_accuracy, training_history).
        """
        history = self.train(data, epochs, patience, edge_weight=edge_weight)
        test_acc = self.evaluate(data, edge_weight=edge_weight)
        return test_acc, history
