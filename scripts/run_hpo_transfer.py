#!/usr/bin/env python3
"""HPO Transfer experiment — 3-condition self-contained design.

Research question:
  If we run a full hyperparameter search on the *sparsified* graph and select
  the optimal HP set H*_sparse (FlexibleGCN, val-acc objective), does it yield
  near-optimal performance when the model is retrained on the *complete* graph?

3-condition design (all equal budget, all val-acc objective, no test leakage):
  oracle   — Optuna TPE on full graph  (r=1.0), 20 trials, run once per (ds, metric)
  proxy    — Optuna TPE on sparse graph (r<1.0), 20 trials per retention rate
  baseline — Optuna RandomSampler on sparse graph, 20 trials per retention rate

All three: retrain best HP config on the full graph (3 seeds), record test acc.

Primary metric:
  normalized_proxy = (tpe_acc - rnd_acc) / max(oracle_acc - rnd_acc, 1e-4)
    =  1.0  → proxy recovers oracle-level performance
    =  0.0  → proxy no better than random
    < 0.0  → proxy worse than random (noisy landscape)

Model: FlexibleGCN (GCNConv, binary adjacency, same as run_hpo_robustness.py).
HP space: lr, weight_decay, dropout, hidden_channels, num_layers.
Early stopping on val acc via GNNTrainer (patience=50, max 500 epochs).

Output: results/hpo_transfer/<dataset>_transfer_<metric>.json
        Written incrementally after each retention rate.

Usage:
  python scripts/run_hpo_transfer.py --dataset cora --metric jaccard
  python scripts/run_hpo_transfer.py --dataset chameleon --metric random --resume
  python scripts/run_hpo_transfer.py --all --metric jaccard
  python scripts/run_hpo_transfer.py --list
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise ImportError("optuna is required: pip install optuna") from e

from torch import nn
from torch_geometric.nn import GCNConv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data import SAFE_DATASETS, DatasetLoader
from src.sparsification.core import GraphSparsifier
from src.training.trainer import GNNTrainer

# ── Experiment configuration ──────────────────────────────────────────────────

RETENTION_RATES     = [0.8, 0.6, 0.4, 0.2]
N_ORACLE_TRIALS     = 20   # TPE on full graph  (run once per dataset/metric)
N_PROXY_TRIALS      = 20   # TPE on sparse graph (per retention rate)
N_RANDOM_TRIALS     = 20   # RandomSampler on sparse graph (per retention rate)
N_SEEDS_FINAL       = 3    # seeds for final evaluation of each H* on full graph
EPOCHS              = 500
PATIENCE            = 50
RANDOM_SCORE_SEED   = 42

# HP search space — identical to run_hpo_robustness.py
LR_MIN,  LR_MAX  = 1e-4, 1e-1
WD_MIN,  WD_MAX  = 0.0,  5e-2
DO_MIN,  DO_MAX  = 0.0,  0.9
HIDDEN_CHOICES   = [8, 16, 32, 64, 128, 256]
LAYER_CHOICES    = [1, 2, 3, 4]

SPARSIFIER_METRICS = ["jaccard", "adamic_adar", "approx_er", "feature_cosine"]
INVERSE_METRICS    = ["jaccard_inv", "adamic_adar_inv", "approx_er_inv", "feature_cosine_inv"]
ALL_METRICS        = SPARSIFIER_METRICS + INVERSE_METRICS + ["random"]

_SKIP = {"polblogs", "flickr", "physics", "cs", "corafull", "ppi"}
VALID_DATASETS = [d for d in SAFE_DATASETS if d not in _SKIP]

TRANSFER_RESULTS_DIR = REPO_ROOT / "results" / "hpo_transfer"


# ── FlexibleGCN — identical to run_hpo_robustness.py ─────────────────────────

class FlexibleGCN(nn.Module):
    """Variable-depth GCN (GCNConv + ReLU + Dropout). Binary adjacency only."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(
            self.convs[-1](x, edge_index, edge_weight=edge_weight), dim=1
        )


# ── Symmetric random sparsification — identical to run_hpo_robustness.py ─────

def _precompute_random_scores(data):
    ei  = data.edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    n   = data.num_nodes
    u   = np.minimum(src, dst)
    v   = np.maximum(src, dst)
    keys = u.astype(np.int64) * (n + 1) + v.astype(np.int64)
    _, inverse_idx = np.unique(keys, return_inverse=True)
    n_undirected   = int(inverse_idx.max()) + 1
    rng = np.random.default_rng(RANDOM_SCORE_SEED)
    undirected_scores = rng.random(n_undirected)
    return undirected_scores, inverse_idx


def _random_sparsify(data, undirected_scores, inverse_idx, retention_ratio, device):
    if retention_ratio == 1.0:
        return data.clone()
    n_undirected = len(undirected_scores)
    n_keep       = max(1, int(n_undirected * retention_ratio))
    keep_undir   = np.zeros(n_undirected, dtype=bool)
    keep_undir[np.argsort(undirected_scores)[-n_keep:]] = True
    mask = keep_undir[inverse_idx]
    sparse = data.clone()
    sparse.edge_index = data.edge_index[:, mask].to(device)
    return sparse


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_val_acc(graph, num_features, num_classes, hp, device, seed):
    """Train one HP config (1 seed); return best_val_acc — Optuna objective."""
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
    opt = torch.optim.Adam(model.parameters(),
                           lr=hp["lr"], weight_decay=hp["weight_decay"])
    trainer = GNNTrainer(model=model, optimizer=opt, device=device)
    history = trainer.train(graph, epochs=EPOCHS, patience=PATIENCE)
    return history["best_val_acc"]


def _evaluate_hp(graph, num_features, num_classes, hp, device, seeds):
    """Evaluate HP config across seeds on graph; return test metrics."""
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
        opt = torch.optim.Adam(model.parameters(),
                               lr=hp["lr"], weight_decay=hp["weight_decay"])
        trainer = GNNTrainer(model=model, optimizer=opt, device=device)
        t0 = time.time()
        _, hist = trainer.train_and_evaluate(graph, epochs=EPOCHS, patience=PATIENCE)
        accs.append(hist["test_metrics"]["accuracy"])
        f1s.append(hist["test_metrics"]["macro_f1"])
        times.append(time.time() - t0)
    return {
        "acc_mean":     float(np.mean(accs)),
        "acc_std":      float(np.std(accs)),
        "f1_mean":      float(np.mean(f1s)),
        "train_time_s": float(np.mean(times)),
    }


# ── Optuna study helpers ───────────────────────────────────────────────────────

def _make_objective(graph, num_features, num_classes, device):
    """Return an Optuna objective bound to graph. Uses val acc — no test leakage."""
    def objective(trial):
        hp = {
            "lr":              trial.suggest_float("lr", LR_MIN, LR_MAX, log=True),
            "weight_decay":    trial.suggest_float("weight_decay", WD_MIN, WD_MAX),
            "dropout":         trial.suggest_float("dropout", DO_MIN, DO_MAX),
            "hidden_channels": trial.suggest_categorical("hidden_channels", HIDDEN_CHOICES),
            "num_layers":      trial.suggest_categorical("num_layers", LAYER_CHOICES),
        }
        try:
            return _train_val_acc(graph, num_features, num_classes, hp, device,
                                  seed=trial.number % 10)
        except Exception:
            return 0.0
    return objective


def _run_study(graph, num_features, num_classes, device,
               n_trials, sampler, study_seed=42):
    """Run an Optuna study; return (best_params, best_val_acc, study_time_s)."""
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )
    t0 = time.time()
    study.optimize(
        _make_objective(graph, num_features, num_classes, device),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params, float(study.best_value), float(time.time() - t0)


# ── Main experiment ───────────────────────────────────────────────────────────

def run_dataset(dataset_name, device, resume, metric):
    output_path = TRANSFER_RESULTS_DIR / f"{dataset_name}_transfer_{metric}.json"

    existing_results = {}
    completed_keys   = set()
    oracle_block     = None   # reuse oracle across retention rates

    if resume and output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        existing_results = existing.get("results", {})
        completed_keys   = set(existing_results.keys())
        oracle_block     = existing.get("oracle_condition")
        print(f"  Resuming — {len(completed_keys)} retention rates done: "
              f"{sorted(completed_keys, key=float)}")
        if oracle_block:
            print(f"  Reusing oracle from checkpoint  "
                  f"(acc={oracle_block['acc_mean']:.4f})")

    print(f"\n{'#'*70}")
    print(f"# DATASET: {dataset_name.upper()}  |  Metric: {metric}  |  Device: {device}")
    print(f"{'#'*70}")

    loader = DatasetLoader(root=str(REPO_ROOT / "data"))
    full_data, num_features, num_classes = loader.get_dataset(dataset_name, device)

    src, dst = full_data.edge_index
    homophily = float((full_data.y[src] == full_data.y[dst]).float().mean().item())
    print(f"  {full_data.num_nodes:,} nodes  {full_data.edge_index.size(1):,} edges  "
          f"{num_features} features  {num_classes} classes  h={homophily:.3f}")

    # ── Oracle: TPE on full graph (once per dataset/metric) ───────────────────
    seeds_final = list(range(N_SEEDS_FINAL))

    if oracle_block is None:
        print(f"  [Oracle] Optuna TPE on FULL graph — {N_ORACLE_TRIALS} trials")
        tpe_sampler_oracle = optuna.samplers.TPESampler(seed=42)
        hp_oracle, best_val_oracle, t_oracle = _run_study(
            full_data, num_features, num_classes, device,
            n_trials=N_ORACLE_TRIALS, sampler=tpe_sampler_oracle,
        )
        oracle_eval = _evaluate_hp(full_data, num_features, num_classes,
                                   hp_oracle, device, seeds_final)
        oracle_block = {
            "best_params":         hp_oracle,
            "best_val_acc":        best_val_oracle,
            "n_trials":            N_ORACLE_TRIALS,
            "study_time_s":        t_oracle,
            **oracle_eval,
        }
        print(f"  Oracle → test acc: {oracle_block['acc_mean']:.4f}  "
              f"(val acc on full: {best_val_oracle:.4f})")
    oracle_acc = oracle_block["acc_mean"]

    # ── Sparsifier setup ──────────────────────────────────────────────────────
    if metric in SPARSIFIER_METRICS:
        sparsifier = GraphSparsifier(full_data, device)
        print(f"  Computing {metric} edge scores...")
        sparsifier.compute_scores(metric)
        def _sparsify(r):
            return sparsifier.sparsify(metric, r)
    elif metric in INVERSE_METRICS:
        base_metric = metric[:-4]
        sparsifier = GraphSparsifier(full_data, device)
        print(f"  Computing {base_metric} edge scores (inverse threshold)...")
        sparsifier.compute_scores(base_metric)
        def _sparsify(r):
            return sparsifier.sparsify(base_metric, r, keep_lowest=True)
    else:
        print(f"  Precomputing symmetric random scores (seed={RANDOM_SCORE_SEED})...")
        undirected_scores, inverse_idx = _precompute_random_scores(full_data)
        def _sparsify(r):
            return _random_sparsify(full_data, undirected_scores, inverse_idx, r, device)

    results = existing_results.copy()

    for r in sorted(RETENTION_RATES, reverse=True):
        r_key = str(r)
        if r_key in completed_keys:
            print(f"  r={r:.1f}  SKIPPED (already done)")
            continue

        sparse_data = _sparsify(r)
        n_edges     = int(sparse_data.edge_index.size(1))
        s_src = sparse_data.edge_index[0].cpu()
        s_dst = sparse_data.edge_index[1].cpu()
        y_cpu = full_data.y.cpu()
        effective_h = float((y_cpu[s_src] == y_cpu[s_dst]).float().mean().item())
        print(f"\n  r={r:.1f}  ({n_edges:,} edges, eff_h={effective_h:.3f})")

        # ── Proxy: TPE on sparse graph ─────────────────────────────────────────
        print(f"  [Proxy ]  Optuna TPE    on sparse graph — {N_PROXY_TRIALS} trials")
        tpe_sampler = optuna.samplers.TPESampler(seed=42)
        hp_tpe, best_val_tpe, t_tpe = _run_study(
            sparse_data, num_features, num_classes, device,
            n_trials=N_PROXY_TRIALS, sampler=tpe_sampler,
        )
        tpe_eval = _evaluate_hp(full_data, num_features, num_classes,
                                hp_tpe, device, seeds_final)
        tpe_acc = tpe_eval["acc_mean"]

        # ── Baseline: RandomSampler on sparse graph ────────────────────────────
        print(f"  [Random]  Optuna Random on sparse graph — {N_RANDOM_TRIALS} trials")
        rnd_sampler = optuna.samplers.RandomSampler(seed=42)
        hp_rnd, best_val_rnd, t_rnd = _run_study(
            sparse_data, num_features, num_classes, device,
            n_trials=N_RANDOM_TRIALS, sampler=rnd_sampler,
        )
        rnd_eval = _evaluate_hp(full_data, num_features, num_classes,
                                hp_rnd, device, seeds_final)
        rnd_acc = rnd_eval["acc_mean"]

        # ── Summary ────────────────────────────────────────────────────────────
        denom = max(oracle_acc - rnd_acc, 1e-4)
        tpe_ratio        = float(tpe_acc / oracle_acc) if oracle_acc > 1e-9 else float("nan")
        rnd_ratio        = float(rnd_acc / oracle_acc) if oracle_acc > 1e-9 else float("nan")
        normalized_proxy = float((tpe_acc - rnd_acc) / denom)
        proxy_time_s     = float(t_tpe + tpe_eval["train_time_s"] * N_SEEDS_FINAL)

        print(f"  oracle={oracle_acc:.4f}  tpe={tpe_acc:.4f}  rnd={rnd_acc:.4f}")
        print(f"  tpe_ratio={tpe_ratio:.4f}  normalized_proxy={normalized_proxy:.4f}  "
              f"({'above' if tpe_acc > rnd_acc else 'below'} random)")

        results[r_key] = {
            "meta": {
                "retention_ratio": r,
                "n_edges":         n_edges,
                "effective_h":     effective_h,
            },
            "proxy_tpe": {
                "best_params":        hp_tpe,
                "best_val_acc_on_sparse": best_val_tpe,
                "n_trials":           N_PROXY_TRIALS,
                "study_time_s":       float(t_tpe),
                **tpe_eval,
            },
            "baseline_random": {
                "best_params":        hp_rnd,
                "best_val_acc_on_sparse": best_val_rnd,
                "n_trials":           N_RANDOM_TRIALS,
                "study_time_s":       float(t_rnd),
                **rnd_eval,
            },
            "summary": {
                "oracle_acc":        float(oracle_acc),
                "tpe_acc":           float(tpe_acc),
                "rnd_acc":           float(rnd_acc),
                "tpe_ratio":         tpe_ratio,
                "rnd_ratio":         rnd_ratio,
                "normalized_proxy":  normalized_proxy,
                "above_random":      bool(tpe_acc > rnd_acc),
                "is_near_optimal_95": bool(tpe_ratio >= 0.95),
                "is_near_optimal_99": bool(tpe_ratio >= 0.99),
                "proxy_time_s":      proxy_time_s,
            },
        }

        _checkpoint(output_path, dataset_name, metric, num_features,
                    num_classes, homophily, oracle_block, results)
        print(f"  Saved → {output_path}")

    print(f"\n  Finished {dataset_name} [{metric}]")
    del full_data
    if metric in SPARSIFIER_METRICS or metric in INVERSE_METRICS:
        del sparsifier
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _checkpoint(path, dataset, metric, num_features, num_classes,
                homophily, oracle_block, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset":           dataset,
            "metric":            metric,
            "homophily":         homophily,
            "n_oracle_trials":   N_ORACLE_TRIALS,
            "n_proxy_trials":    N_PROXY_TRIALS,
            "n_random_trials":   N_RANDOM_TRIALS,
            "n_seeds_final":     N_SEEDS_FINAL,
            "epochs":            EPOCHS,
            "patience":          PATIENCE,
            "retention_rates":   RETENTION_RATES,
            "design": (
                "3-condition self-contained: oracle TPE (full graph, val-acc), "
                "proxy TPE (sparse graph, val-acc), baseline Random (sparse graph, val-acc). "
                "All equal budget. primary_metric=normalized_proxy."
            ),
            "num_features": num_features,
            "num_classes":  num_classes,
        },
        "oracle_condition": oracle_block,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "HPO Transfer v2: 3-condition design. "
            "Oracle TPE (full), Proxy TPE (sparse), Baseline Random (sparse). "
            "Primary metric: normalized_proxy."
        )
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--metric", type=str, default="jaccard",
        choices=ALL_METRICS,
    )
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip retention rates already in output JSON")
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
    TRANSFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Metric: {args.metric}  |  "
          f"Oracle/Proxy/Random trials: {N_ORACLE_TRIALS}/{N_PROXY_TRIALS}/{N_RANDOM_TRIALS}  |  "
          f"Final seeds: {N_SEEDS_FINAL}")

    datasets = VALID_DATASETS if args.all else [args.dataset.lower()]
    t0 = time.time()

    for ds in datasets:
        if ds not in VALID_DATASETS:
            print(f"ERROR: '{ds}' not in valid datasets. Run --list to see options.")
            continue
        try:
            run_dataset(ds, device, resume=args.resume, metric=args.metric)
        except Exception as exc:
            import traceback
            print(f"ERROR on {ds}: {exc}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"COMPLETED in {(time.time()-t0)/60:.1f} minutes")
    print(f"Results in: {TRANSFER_RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
