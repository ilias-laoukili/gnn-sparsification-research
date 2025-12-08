"""Reproducibility utilities for stochastic experiments."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    Sets deterministic behavior for all random number generators to ensure
    experiment reproducibility. Also configures cuDNN for deterministic ops.

    Args:
        seed: Integer seed value.

    Example:
        >>> set_global_seed(42)
        >>> torch.rand(3)  # Now reproducible across runs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for hash stability
    os.environ["PYTHONHASHSEED"] = str(seed)
