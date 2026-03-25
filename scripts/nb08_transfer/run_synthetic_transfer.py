#!/usr/bin/env python3
"""Synthetic HPO Transfer experiment on cSBM graphs.

2-condition transfer + HP probe for flatness/rank correlation.

Transfer (C2):
  oracle — Optuna TPE on full graph  (r=1.0), 50 trials
  proxy  — Optuna TPE on sparse graph (r<1.0), 50 trials per retention rate
  Both: val-acc selection, retrain best HP on full graph, 3-seed eval.

HP probe (C1 + rank correlation):
  20 fixed random HP configs evaluated at r=1.0 and every sparse r.
  Provides σ(acc) for flatness and Spearman ρ for rank correlation.

Primary metrics:
  transfer_loss = |acc_oracle - acc_proxy| / acc_oracle
  acc_ratio     = acc_proxy / acc_oracle

Output: results/hpo_transfer/synth_h{h}_mu{mu}_d{d}_c{c}_n{n}_gs{gs}_transfer_{metric}.json

Usage:
    python scripts/nb08_transfer/run_synthetic_transfer.py --h 0.5 --metric jaccard
    python scripts/nb08_transfer/run_synthetic_transfer.py --h 0.5 --metric random --graph_seed 1
    python scripts/nb08_transfer/run_synthetic_transfer.py --h 0.5 --metric jaccard --resume
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.synthetic import build_csbm_graph, dataset_name as _synth_ds_name
from src.hpo.config import (
    ALL_METRICS, DEFAULT_EPOCHS, DEFAULT_PATIENCE,
    INVERSE_METRICS, RANDOM_SCORE_SEED, SPARSIFIER_METRICS,
)
from src.hpo.optuna_helpers import run_study
from src.hpo.probe import generate_probe_configs, run_probe
from src.sparsification.core import GraphSparsifier
from src.sparsification.random import precompute_random_scores, random_sparsify
from src.training.hpo_helpers import evaluate_hp
from src.utils.device import get_device

# ── Configuration ────────────────────────────────────────────────────────────

RETENTION_RATES   = [0.8, 0.6, 0.4, 0.2]
N_ORACLE_TRIALS   = 50
N_PROXY_TRIALS    = 50
N_SEEDS_FINAL     = 3
N_PROBE_CONFIGS   = 20   # fixed random HP configs for flatness + rank correlation
EPOCHS            = DEFAULT_EPOCHS
PATIENCE          = DEFAULT_PATIENCE

DEFAULT_MU = 1.0
DEFAULT_D  = 10.0
DEFAULT_N  = 1000
DEFAULT_C  = 5
DEFAULT_F  = 64

RESULTS_DIR = REPO_ROOT / "results" / "hpo_transfer"


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(args, device):
    h, mu, d, n, c, f, gs = args.h, args.mu, args.d, args.n, args.c, args.f, args.graph_seed
    metric = args.metric

    ds_name = _synth_ds_name(h, mu, d, c, n, gs, theta_exp=None, arch="gcn")
    output_path = RESULTS_DIR / f"{ds_name}_transfer_{metric}.json"

    # ── Resume ───────────────────────────────────────────────────────────
    existing_results = {}
    completed_keys = set()
    oracle_block = None
    existing = {}

    if args.resume and output_path.exists():
        with open(output_path) as fh:
            existing = json.load(fh)
        existing_results = existing.get("results", {})
        completed_keys = set(existing_results.keys())
        oracle_block = existing.get("oracle_condition")
        print(f"  Resuming — {len(completed_keys)} retention rates done: "
              f"{sorted(completed_keys, key=float)}")

    print(f"\n{'#'*70}")
    print(f"# SYNTHETIC TRANSFER: h={h}  mu={mu}  d={d}  n={n}  c={c}  gs={gs}")
    print(f"# Metric: {metric}  |  Device: {device}")
    print(f"{'#'*70}")

    # ── Generate graph ───────────────────────────────────────────────────
    full_data, effective_h = build_csbm_graph(
        h=h, mu=mu, d=d, n=n, c=c, f=f,
        graph_seed=gs, theta_exp=None, device=device,
    )
    n_feat = int(full_data.x.size(1))
    n_class = int(full_data.y.max().item()) + 1

    print(f"  {full_data.num_nodes:,} nodes  {full_data.edge_index.size(1):,} edges  "
          f"{n_feat} features  {n_class} classes  h_eff={effective_h:.3f}")

    # ── Oracle: TPE on full graph (run once) ─────────────────────────────
    seeds_final = list(range(N_SEEDS_FINAL))

    if oracle_block is None:
        print(f"  [Oracle] Optuna TPE on FULL graph — {N_ORACLE_TRIALS} trials")
        sampler = optuna.samplers.TPESampler(seed=42)
        hp_oracle, val_oracle, t_oracle = run_study(
            full_data, n_feat, n_class, device,
            n_trials=N_ORACLE_TRIALS, sampler=sampler,
            epochs=EPOCHS, patience=PATIENCE,
        )
        oracle_eval = evaluate_hp(full_data, n_feat, n_class,
                                  hp_oracle, device, seeds_final,
                                  epochs=EPOCHS, patience=PATIENCE)
        oracle_block = {
            "best_params":  hp_oracle,
            "best_val_acc": val_oracle,
            "n_trials":     N_ORACLE_TRIALS,
            "study_time_s": t_oracle,
            **oracle_eval,
        }
        print(f"  Oracle → test acc: {oracle_block['acc_mean']:.4f}  "
              f"(val: {val_oracle:.4f})")
    else:
        print(f"  Oracle from checkpoint → acc: {oracle_block['acc_mean']:.4f}")

    oracle_acc = oracle_block["acc_mean"]

    # ── HP Probe on dense graph (r=1.0) ──────────────────────────────────
    probe_configs = generate_probe_configs(N_PROBE_CONFIGS)
    probe_dense = existing.get("probe_dense") if args.resume and output_path.exists() else None

    if probe_dense is None:
        print(f"  [Probe] {N_PROBE_CONFIGS} fixed HP configs on DENSE graph...")
        probe_dense = run_probe(full_data, n_feat, n_class, device, probe_configs,
                                epochs=EPOCHS, patience=PATIENCE)
        dense_accs = [p["val_acc"] for p in probe_dense]
        print(f"  Probe dense: mean={np.mean(dense_accs):.4f}  "
              f"std={np.std(dense_accs):.4f}")
    else:
        print(f"  Probe dense from checkpoint")

    # ── Sparsifier setup ─────────────────────────────────────────────────
    if metric in SPARSIFIER_METRICS:
        sparsifier = GraphSparsifier(full_data, device)
        print(f"  Computing {metric} edge scores...")
        sparsifier.compute_scores(metric)
        def _sparsify(r):
            return sparsifier.sparsify(metric, r)
    elif metric in INVERSE_METRICS:
        base = metric[:-4]
        sparsifier = GraphSparsifier(full_data, device)
        print(f"  Computing {base} edge scores (inverse)...")
        sparsifier.compute_scores(base)
        def _sparsify(r):
            return sparsifier.sparsify(base, r, keep_lowest=True)
    else:
        print(f"  Random sparsification (seed={RANDOM_SCORE_SEED})")
        rnd_scores, rnd_inv = precompute_random_scores(full_data, seed=RANDOM_SCORE_SEED)
        def _sparsify(r):
            return random_sparsify(full_data, rnd_scores, rnd_inv, r, device)

    results = existing_results.copy()

    for r in sorted(RETENTION_RATES, reverse=True):
        r_key = str(r)
        if r_key in completed_keys:
            print(f"\n  r={r:.1f}  SKIPPED (done)")
            continue

        sparse_data = _sparsify(r)
        n_edges = int(sparse_data.edge_index.size(1))
        s_src = sparse_data.edge_index[0].cpu()
        s_dst = sparse_data.edge_index[1].cpu()
        y_cpu = full_data.y.cpu()
        eff_h_sparse = float((y_cpu[s_src] == y_cpu[s_dst]).float().mean())
        print(f"\n  r={r:.1f}  ({n_edges:,} edges, eff_h={eff_h_sparse:.3f})")

        # ── Probe on sparse graph (also eval on full for flattening) ─────
        print(f"  [Probe] {N_PROBE_CONFIGS} fixed HP configs on sparse graph...")
        probe_sparse = run_probe(sparse_data, n_feat, n_class, device,
                                 probe_configs, eval_graph=full_data,
                                 epochs=EPOCHS, patience=PATIENCE)
        sparse_accs = [p["val_acc"] for p in probe_sparse]
        full_accs = [p["val_acc_on_full"] for p in probe_sparse]
        print(f"  Probe sparse: mean={np.mean(sparse_accs):.4f}  "
              f"std={np.std(sparse_accs):.4f}")
        print(f"  Probe on-full: mean={np.mean(full_accs):.4f}  "
              f"std={np.std(full_accs):.4f}")

        # ── Proxy: TPE on sparse graph ───────────────────────────────────
        print(f"  [Proxy] Optuna TPE on sparse graph — {N_PROXY_TRIALS} trials")
        sampler_proxy = optuna.samplers.TPESampler(seed=42)
        hp_proxy, val_proxy, t_proxy = run_study(
            sparse_data, n_feat, n_class, device,
            n_trials=N_PROXY_TRIALS, sampler=sampler_proxy,
            epochs=EPOCHS, patience=PATIENCE,
        )
        # Retrain best proxy HP on FULL graph
        proxy_eval = evaluate_hp(full_data, n_feat, n_class,
                                 hp_proxy, device, seeds_final,
                                 epochs=EPOCHS, patience=PATIENCE)
        proxy_acc = proxy_eval["acc_mean"]

        # ── Metrics ──────────────────────────────────────────────────────
        acc_ratio = float(proxy_acc / oracle_acc) if oracle_acc > 1e-9 else float("nan")
        transfer_loss = float(abs(oracle_acc - proxy_acc) / oracle_acc) if oracle_acc > 1e-9 else float("nan")

        print(f"  oracle={oracle_acc:.4f}  proxy={proxy_acc:.4f}  "
              f"ratio={acc_ratio:.4f}  loss={transfer_loss:.4f}")

        results[r_key] = {
            "meta": {
                "retention_ratio": r,
                "n_edges":         n_edges,
                "effective_h":     eff_h_sparse,
            },
            "proxy": {
                "best_params":           hp_proxy,
                "best_val_acc_on_sparse": val_proxy,
                "n_trials":              N_PROXY_TRIALS,
                "study_time_s":          float(t_proxy),
                **proxy_eval,
            },
            "probe": probe_sparse,
            "summary": {
                "oracle_acc":     float(oracle_acc),
                "proxy_acc":      float(proxy_acc),
                "acc_ratio":      acc_ratio,
                "transfer_loss":  transfer_loss,
                "proxy_time_s":   float(t_proxy + proxy_eval["train_time_s"] * N_SEEDS_FINAL),
            },
        }

        _checkpoint(output_path, ds_name, metric, n_feat, n_class,
                    effective_h, oracle_block, results,
                    h, mu, d, n, c, f, gs, effective_h,
                    probe_configs=probe_configs, probe_dense=probe_dense)
        print(f"  Saved → {output_path}")

    print(f"\n  Finished {ds_name} [{metric}]")
    del full_data
    if metric in SPARSIFIER_METRICS or metric in INVERSE_METRICS:
        del sparsifier
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ── Checkpoint ───────────────────────────────────────────────────────────────

def _checkpoint(path, ds_name, metric, n_feat, n_class,
                homophily, oracle_block, results,
                h, mu, d, n, c, f, gs, effective_h,
                probe_configs=None, probe_dense=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "dataset":         ds_name,
            "metric":          metric,
            "homophily":       homophily,
            "n_oracle_trials": N_ORACLE_TRIALS,
            "n_proxy_trials":  N_PROXY_TRIALS,
            "n_seeds_final":   N_SEEDS_FINAL,
            "n_probe_configs": N_PROBE_CONFIGS,
            "epochs":          EPOCHS,
            "patience":        PATIENCE,
            "retention_rates": RETENTION_RATES,
            "design": (
                "2-condition: oracle TPE (full graph, val-acc, 50 trials) vs "
                "proxy TPE (sparse graph, val-acc, 50 trials). "
                "Both retrain best HP on full graph (3 seeds). "
                "HP probe: 20 fixed configs at each r for flatness/rank. "
                "primary_metric=acc_ratio."
            ),
            "num_features": n_feat,
            "num_classes":  n_class,
            "synth": {
                "requested_h":  h,
                "effective_h":  effective_h,
                "mu": mu, "d": d, "n": n, "c": c, "f": f,
                "graph_seed":   gs,
            },
        },
        "oracle_condition": oracle_block,
        "probe_configs":   probe_configs,
        "probe_dense":     probe_dense,
        "results": results,
    }
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic HPO Transfer: 2-condition (oracle vs proxy) on cSBM."
    )
    parser.add_argument("--h", type=float, required=True)
    parser.add_argument("--metric", type=str, default="jaccard", choices=ALL_METRICS)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument("--d", type=float, default=DEFAULT_D)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--c", type=int, default=DEFAULT_C)
    parser.add_argument("--f", type=int, default=DEFAULT_F)
    parser.add_argument("--graph_seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  Oracle/Proxy trials: {N_ORACLE_TRIALS}/{N_PROXY_TRIALS}  |  "
          f"Final seeds: {N_SEEDS_FINAL}")

    t0 = time.time()
    run_experiment(args, device)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
