"""Training helpers for HPO / transfer experiments.

These wrap GNNTrainer with a FlexibleGCN (or any model_cls) for single-run
and multi-seed evaluation of a given HP dict.
"""

import time

import numpy as np
import torch

from src.hpo.config import DEFAULT_EPOCHS, DEFAULT_PATIENCE
from src.models.flexible import FlexibleGCN
from src.training.trainer import GNNTrainer


def train_val_acc(graph, n_feat, n_class, hp, device, seed,
                  model_cls=FlexibleGCN, epochs=DEFAULT_EPOCHS,
                  patience=DEFAULT_PATIENCE):
    """Train one model and return best validation accuracy (float)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(n_feat, hp["hidden_channels"], n_class,
                      hp["num_layers"], hp["dropout"]).to(device)
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                           weight_decay=hp["weight_decay"])
    trainer = GNNTrainer(model=model, optimizer=opt, device=device)
    history = trainer.train(graph, epochs=epochs, patience=patience)
    return history["best_val_acc"]


def evaluate_hp(graph, n_feat, n_class, hp, device, seeds,
                model_cls=FlexibleGCN, epochs=DEFAULT_EPOCHS,
                patience=DEFAULT_PATIENCE):
    """Multi-seed evaluation: train + test for each seed.

    Returns dict with acc_mean, acc_std, f1_mean, train_time_s.
    """
    accs, f1s, times = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_cls(n_feat, hp["hidden_channels"], n_class,
                          hp["num_layers"], hp["dropout"]).to(device)
        model.reset_parameters()
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                               weight_decay=hp["weight_decay"])
        trainer = GNNTrainer(model=model, optimizer=opt, device=device)
        t0 = time.time()
        _, hist = trainer.train_and_evaluate(graph, epochs=epochs,
                                             patience=patience)
        accs.append(hist["test_metrics"]["accuracy"])
        f1s.append(hist["test_metrics"]["macro_f1"])
        times.append(time.time() - t0)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_mean":  float(np.mean(f1s)),
        "train_time_s": float(np.mean(times)),
    }
