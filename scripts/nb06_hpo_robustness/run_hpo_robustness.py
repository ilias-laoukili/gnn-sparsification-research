#!/usr/bin/env python3
"""HPO Robustness experiment.

Tests the hypothesis: graph sparsification acts as a structural regularizer
that flattens the optimization landscape, making GCN training robust to
poor hyperparameter choices.

For each (dataset, retention_ratio), evaluates N_HP_SAMPLES random HP
configurations sampled via Latin Hypercube Sampling and measures:
  - Accuracy distribution over the HP grid (mean, std, IQR, best, worst)
  - yield_rate_within: fraction of configs at >= 90% of best-at-this-r
    (correct robustness metric — independent of accuracy level)
  - yield_rate_vs_dense: fraction reaching 90% of the best dense-graph config
    (answers: "can bad HPs still match the best dense model?")
  - robustness_gain: std(dense) / std(sparse)

Nine sparsification methods are supported via --metric:
  jaccard            — keep top-r% of edges by Jaccard neighborhood overlap.
  adamic_adar        — keep top-r% by Adamic-Adar (log-weighted common neighbors).
  approx_er          — keep top-r% by approximate effective resistance.
  feature_cosine     — keep top-r% by cosine similarity of node feature vectors.
  jaccard_inv        — keep BOTTOM-r% by Jaccard (InverseThreshold).
  adamic_adar_inv    — keep BOTTOM-r% by Adamic-Adar (InverseThreshold).
  approx_er_inv      — keep BOTTOM-r% by approx effective resistance (InverseThreshold).
  feature_cosine_inv — keep BOTTOM-r% by feature cosine (InverseThreshold).
  random             — keep a uniformly random r% of edges (density-only control).

Output: results/hpo/<dataset>_hpo_<metric>.json
        Written incrementally after each retention rate (safe against crashes).

Usage:
    python scripts/nb06_hpo_robustness/run_hpo_robustness.py --dataset cora
    python scripts/nb06_hpo_robustness/run_hpo_robustness.py --dataset cora --metric random
    python scripts/nb06_hpo_robustness/run_hpo_robustness.py --dataset cora --resume
    python scripts/nb06_hpo_robustness/run_hpo_robustness.py --all
    python scripts/nb06_hpo_robustness/run_hpo_robustness.py --list
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import SAFE_DATASETS, DatasetLoader
from src.hpo.config import (
    DO_MAX, DO_MIN, HIDDEN_CHOICES, INVERSE_METRICS,
    LAYER_CHOICES, LR_MAX, LR_MIN, SPARSIFIER_METRICS,
    WD_MAX, WD_MIN,
)
from src.models.flexible import FlexibleGCN
from src.sparsification.core import GraphSparsifier
from src.sparsification.random import precompute_random_scores, random_sparsify
from src.training.trainer import GNNTrainer
from src.utils.device import get_device

# ── Experiment configuration ──────────────────────────────────────────────────

RETENTION_RATES   = [1.0, 0.8, 0.6, 0.4, 0.2]
N_HP_SAMPLES      = 100         # LHC samples — same set across all retention rates
N_SEEDS           = 3           # Seeds per HP config
EPOCHS            = 500
PATIENCE          = 50
YIELD_THRESHOLD   = 0.90        # Used for both within-r and vs-dense yield rates
RANDOM_SCORE_SEED = 42          # Seed for random sparsification edge ranking

RESULTS_DIR = REPO_ROOT / "results" / "hpo"

# Datasets to skip (no node features or too large for CPU/MPS)
_SKIP = {"polblogs", "flickr", "physics", "cs", "corafull", "ppi"}
VALID_DATASETS = [d for d in SAFE_DATASETS if d not in _SKIP]


# ── HP Sampling ───────────────────────────────────────────────────────────────

def sample_hp_configs(n: int, seed: int = 42) -> list:
    """Sample n HP configurations via Latin Hypercube Sampling over 5 dimensions."""
    sampler = LatinHypercube(d=5, seed=seed)
    samples = sampler.random(n=n)

    configs = []
    for u_lr, u_wd, u_do, u_hid, u_lay in samples:
        lr     = 10 ** (np.log10(LR_MIN) + u_lr * (np.log10(LR_MAX) - np.log10(LR_MIN)))
        wd     = WD_MIN + u_wd * (WD_MAX - WD_MIN)
        do     = DO_MIN + u_do * (DO_MAX - DO_MIN)
        hidden = HIDDEN_CHOICES[min(int(u_hid * len(HIDDEN_CHOICES)), len(HIDDEN_CHOICES) - 1)]
        layers = LAYER_CHOICES[min(int(u_lay * len(LAYER_CHOICES)), len(LAYER_CHOICES) - 1)]
        configs.append({
            "lr":              float(round(lr, 7)),
            "weight_decay":    float(round(wd, 6)),
            "dropout":         float(round(do, 4)),
            "hidden_channels": int(hidden),
            "num_layers":      int(layers),
        })
    return configs


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_config(graph, num_features, num_classes, hp, device, seeds):
    """Train FlexibleGCN with one HP config across multiple seeds."""
    accs, f1s, times = [], [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FlexibleGCN(
            in_channels=num_features,
            hidden_channels=hp["hidden_channels"],
            out_channels=num_classes,
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
        )
        trainer = GNNTrainer(model=model, optimizer=optimizer, device=device)
        t0 = time.time()
        _, history = trainer.train_and_evaluate(graph, epochs=EPOCHS, patience=PATIENCE)
        accs.append(history["test_metrics"]["accuracy"])
        f1s.append(history["test_metrics"]["macro_f1"])
        times.append(time.time() - t0)
    return {
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
        "f1_mean":      float(np.mean(f1s)),
        "f1_std":       float(np.std(f1s)),
        "train_time_s": float(np.mean(times)),
    }


# ── Robustness statistics ─────────────────────────────────────────────────────

def compute_robustness_stats(hp_results, best_dense_acc, dense_std):
    accs = np.array([r["acc_mean"] for r in hp_results])
    q25, q75 = float(np.percentile(accs, 25)), float(np.percentile(accs, 75))

    best_at_r   = float(accs.max())
    thr_within  = YIELD_THRESHOLD * best_at_r
    thr_dense   = YIELD_THRESHOLD * best_dense_acc

    std = float(np.std(accs))
    robustness_gain = (
        float(dense_std / std) if (dense_std is not None and std > 1e-9) else 1.0
    )

    return {
        "acc_mean":            float(np.mean(accs)),
        "acc_std":             std,
        "acc_iqr":             float(q75 - q25),
        "acc_best":            best_at_r,
        "acc_worst":           float(accs.min()),
        "yield_rate_within":   float((accs >= thr_within).mean()),
        "yield_rate_vs_dense": float((accs >= thr_dense).mean()),
        "robustness_gain":     robustness_gain,
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _checkpoint(path, dataset, metric, num_features, num_classes,
                hp_configs, results, homophily):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset":          dataset,
            "homophily":        homophily,
            "n_hp_configs":     N_HP_SAMPLES,
            "n_seeds":          N_SEEDS,
            "metric":           metric,
            "retention_rates":  RETENTION_RATES,
            "epochs":           EPOCHS,
            "patience":         PATIENCE,
            "yield_threshold":  YIELD_THRESHOLD,
            "hp_space": {
                "lr":          [LR_MIN, LR_MAX, "log"],
                "weight_decay":[WD_MIN, WD_MAX, "linear"],
                "dropout":     [DO_MIN, DO_MAX, "linear"],
                "hidden":      HIDDEN_CHOICES,
                "num_layers":  LAYER_CHOICES,
            },
            "num_features": num_features,
            "num_classes":  num_classes,
        },
        "hp_configs": hp_configs,
        "results":    results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ── Main experiment loop ──────────────────────────────────────────────────────

def run_dataset(dataset_name, device, resume, metric):
    output_path = RESULTS_DIR / f"{dataset_name}_hpo_{metric}.json"

    existing_results = {}
    completed_keys   = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_results = existing.get("results", {})
        completed_keys   = set(existing_results.keys())
        print(f"  Resuming — {len(completed_keys)} retention rates done: "
              f"{sorted(completed_keys, key=float)}")

    print(f"\n{'#'*70}")
    print(f"# DATASET: {dataset_name.upper()}  |  Metric: {metric}  |  Device: {device}")
    print(f"{'#'*70}")

    loader = DatasetLoader(root=str(REPO_ROOT / "data"))
    data, num_features, num_classes = loader.get_dataset(dataset_name, device)

    src, dst = data.edge_index
    homophily = float((data.y[src] == data.y[dst]).float().mean().item())

    print(f"  {data.num_nodes:,} nodes  {data.edge_index.size(1):,} edges  "
          f"{num_features} features  {num_classes} classes  h={homophily:.3f}")

    hp_configs = sample_hp_configs(N_HP_SAMPLES, seed=42)
    print(f"  Sampled {N_HP_SAMPLES} HP configs (LHC, seed=42, 5D wider space)")

    if metric in SPARSIFIER_METRICS:
        sparsifier = GraphSparsifier(data, device)
        print(f"  Computing {metric} edge scores...")
        sparsifier.compute_scores(metric)
        def _sparsify(r):
            return sparsifier.sparsify(metric, r)
    elif metric in INVERSE_METRICS:
        base_metric = metric[:-4]
        sparsifier = GraphSparsifier(data, device)
        print(f"  Computing {base_metric} edge scores (inverse threshold)...")
        sparsifier.compute_scores(base_metric)
        def _sparsify(r):
            return sparsifier.sparsify(base_metric, r, keep_lowest=True)
    else:
        print(f"  Precomputing symmetric random edge ranking (seed={RANDOM_SCORE_SEED})...")
        undirected_scores, inverse_idx = precompute_random_scores(data, seed=RANDOM_SCORE_SEED)
        def _sparsify(r):
            return random_sparsify(data, undirected_scores, inverse_idx, r, device)

    print(f"  Done. Starting HP evaluation loop.")

    results   = existing_results.copy()
    seeds     = list(range(N_SEEDS))
    dense_std = None

    if "1.0" in results:
        dense_std = results["1.0"]["robustness_stats"]["acc_std"]

    for r in sorted(RETENTION_RATES, reverse=True):
        r_key = str(r)

        if r_key in completed_keys:
            print(f"  r={r:.1f}  SKIPPED (already done)")
            if r == 1.0 and dense_std is None:
                dense_std = results[r_key]["robustness_stats"]["acc_std"]
            continue

        sparse_data = _sparsify(r)
        n_edges = int(sparse_data.edge_index.size(1))

        s_src = sparse_data.edge_index[0].cpu()
        s_dst = sparse_data.edge_index[1].cpu()
        y_cpu = data.y.cpu()
        effective_h = float((y_cpu[s_src] == y_cpu[s_dst]).float().mean().item())

        print(f"\n  r={r:.1f}  ({n_edges:,} edges, eff_h={effective_h:.3f})  —  "
              f"{N_HP_SAMPLES} configs × {N_SEEDS} seeds")

        hp_results = []
        t_start = time.time()
        for i, hp in enumerate(hp_configs):
            try:
                res = train_one_config(
                    sparse_data, num_features, num_classes, hp, device, seeds
                )
            except Exception as e:
                print(f"    [SKIP hp_idx={i}] {type(e).__name__}: {e}")
                continue
            hp_results.append({"hp_idx": i, **res})

            if (i + 1) % 10 == 0:
                mean_so_far = np.mean([x["acc_mean"] for x in hp_results])
                elapsed = (time.time() - t_start) / 60
                print(f"    [{i+1:3d}/{N_HP_SAMPLES}]  "
                      f"running mean acc={mean_so_far:.4f}  "
                      f"elapsed={elapsed:.1f}m")

        best_dense_acc = (
            float(max(x["acc_mean"] for x in hp_results))
            if r == 1.0
            else float(results["1.0"]["meta"]["best_dense_acc"])
        )

        stats = compute_robustness_stats(hp_results, best_dense_acc, dense_std)

        if r == 1.0:
            dense_std = stats["acc_std"]

        results[r_key] = {
            "meta": {
                "retention_ratio": r,
                "n_edges":         n_edges,
                "best_dense_acc":  best_dense_acc,
                "effective_h":     effective_h,
            },
            "hp_results":       hp_results,
            "robustness_stats": stats,
        }

        _checkpoint(output_path, dataset_name, metric, num_features, num_classes,
                    hp_configs, results, homophily)

        print(f"  r={r:.1f}  DONE  "
              f"std={stats['acc_std']:.4f}  "
              f"yield_within={stats['yield_rate_within']:.1%}  "
              f"yield_vs_dense={stats['yield_rate_vs_dense']:.1%}  "
              f"gain={stats['robustness_gain']:.2f}x  "
              f"[{(time.time()-t_start)/60:.1f}m]")

    print(f"\n  Saved → {output_path}")
    del data
    if metric in SPARSIFIER_METRICS or metric in INVERSE_METRICS:
        del sparsifier
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HPO Robustness: measure GCN sensitivity to HP choices "
                    "on dense vs sparsified graphs."
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--metric",  type=str, default="jaccard",
                        choices=SPARSIFIER_METRICS + INVERSE_METRICS + ["random"],
                        help="Sparsification method. '_inv' suffix = InverseThreshold "
                             "(keep lowest-scoring edges). 'random' = density-only control.")
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list",   action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Valid datasets:")
        for d in VALID_DATASETS:
            print(f"  {d}")
        return

    if not args.dataset and not args.all:
        parser.print_help()
        return

    device = get_device()
    print(f"Device: {device}  |  Metric: {args.metric}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = VALID_DATASETS if args.all else [args.dataset.lower()]
    t0 = time.time()

    for ds in datasets:
        if ds not in VALID_DATASETS:
            print(f"ERROR: '{ds}' not valid. Run --list to see options.")
            continue
        try:
            run_dataset(ds, device, resume=args.resume, metric=args.metric)
        except Exception as exc:
            import traceback
            print(f"ERROR on {ds}: {exc}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"COMPLETED in {(time.time()-t0)/60:.1f} minutes")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
