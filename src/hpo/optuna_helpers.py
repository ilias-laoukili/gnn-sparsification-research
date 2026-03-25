"""Optuna TPE study helpers for HPO transfer experiments."""

import time

import optuna

from src.hpo.config import (
    DO_MAX, DO_MIN, HIDDEN_CHOICES, LAYER_CHOICES,
    LR_MAX, LR_MIN, WD_MAX, WD_MIN,
    DEFAULT_EPOCHS, DEFAULT_PATIENCE,
)
from src.models.flexible import FlexibleGCN
from src.training.hpo_helpers import train_val_acc

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(graph, n_feat, n_class, device,
                   model_cls=FlexibleGCN, epochs=DEFAULT_EPOCHS,
                   patience=DEFAULT_PATIENCE):
    """Return an Optuna objective function for val-acc maximisation."""
    def objective(trial):
        hp = {
            "lr":              trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", WD_MIN, WD_MAX),
            "dropout":         trial.suggest_float("dropout", DO_MIN, DO_MAX),
            "hidden_channels": trial.suggest_categorical("hidden_channels", HIDDEN_CHOICES),
            "num_layers":      trial.suggest_categorical("num_layers", LAYER_CHOICES),
        }
        try:
            return train_val_acc(graph, n_feat, n_class, hp, device,
                                 seed=trial.number % 10,
                                 model_cls=model_cls, epochs=epochs,
                                 patience=patience)
        except Exception:
            return 0.0
    return objective


def run_study(graph, n_feat, n_class, device, n_trials, sampler,
              model_cls=FlexibleGCN, epochs=DEFAULT_EPOCHS,
              patience=DEFAULT_PATIENCE):
    """Create and run an Optuna study.

    Returns (best_params, best_value, elapsed_seconds).
    """
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(make_objective(graph, n_feat, n_class, device,
                                  model_cls=model_cls, epochs=epochs,
                                  patience=patience),
                   n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value), float(time.time() - t0)
