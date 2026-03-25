"""Shared HP search-space bounds and metric lists for HPO experiments.

RETENTION_RATES are NOT defined here — each experiment uses its own set.
"""

# ── HP search space bounds ──────────────────────────────────────────────────
LR_MIN, LR_MAX = 1e-4, 1e-1
WD_MIN, WD_MAX = 0.0, 5e-2
DO_MIN, DO_MAX = 0.0, 0.9
HIDDEN_CHOICES = [8, 16, 32, 64, 128, 256]
LAYER_CHOICES = [1, 2, 3, 4]

# ── Training constants ──────────────────────────────────────────────────────
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 50

# ── Random sparsification seed ──────────────────────────────────────────────
RANDOM_SCORE_SEED = 42

# ── Sparsification metric lists ─────────────────────────────────────────────
# Standard threshold metrics (keep highest-scoring edges)
SPARSIFIER_METRICS = ["jaccard", "adamic_adar", "approx_er", "feature_cosine"]
# Inverse threshold metrics (keep lowest-scoring edges)
INVERSE_METRICS = [m + "_inv" for m in SPARSIFIER_METRICS]
# Metric backbone — retention determined by graph structure
BACKBONE_METRICS = ["metric_backbone_jaccard", "metric_backbone_adamic_adar",
                    "metric_backbone_approx_er"]
# Degree-aware threshold: min edges per node before global threshold
DEGREE_AWARE_METRICS = ["degree_aware_jaccard", "degree_aware_adamic_adar",
                        "degree_aware_approx_er"]
# Probabilistic sampling proportional to edge scores
SAMPLED_METRICS = ["sampled_jaccard", "sampled_adamic_adar", "sampled_approx_er"]

ALL_METRICS = (SPARSIFIER_METRICS + INVERSE_METRICS + BACKBONE_METRICS
               + DEGREE_AWARE_METRICS + SAMPLED_METRICS + ["random"])
