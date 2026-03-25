"""Device detection utility."""

import torch


def get_device() -> str:
    """Return the best available PyTorch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
