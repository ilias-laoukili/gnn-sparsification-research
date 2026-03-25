"""HPO experiment utilities: config, sampling, Optuna helpers, and probing."""

from .config import (
    ALL_METRICS,
    BACKBONE_METRICS,
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    DEGREE_AWARE_METRICS,
    DO_MAX,
    DO_MIN,
    HIDDEN_CHOICES,
    INVERSE_METRICS,
    LAYER_CHOICES,
    LR_MAX,
    LR_MIN,
    RANDOM_SCORE_SEED,
    SAMPLED_METRICS,
    SPARSIFIER_METRICS,
    WD_MAX,
    WD_MIN,
)
# optuna_helpers and probe are NOT imported here to avoid circular imports
# (training.hpo_helpers → hpo.config → hpo/__init__ → optuna_helpers → training.hpo_helpers).
# Import them directly: from src.hpo.optuna_helpers import make_objective, run_study
#                        from src.hpo.probe import generate_probe_configs, run_probe
