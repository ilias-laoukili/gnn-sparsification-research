#!/usr/bin/env python3
"""MLP Control Sweep — synthetic validation of NB04's central claim.

Tests whether GNN(sparse, approx_er) outperforms MLP (no edges) on cSBM
graphs across a range of homophily values, with proper controls.

Fixes over v1:
  - f=512 (matches sweeps 1-3; v1 used f=64)
  - N_HP_SAMPLES=20 (balanced coverage vs runtime)
  - RETENTION_RATIOS=[0.2, 0.4, 0.6] sweep (v1 fixed r=0.4)
  - gcn_random condition: density-matched random baseline (same r, random edges)
    Separates "fewer edges" from "better edges" effects
  - Resume key includes retention_ratio to avoid cross-r collisions
  - --arch flag: GCN (default), GraphSAGE, GAT
  - Output: results/mlp_control/mlp_control.csv (arch column; old rows default arch=gcn)

Hypotheses from NB04:
  H1: MLP > GNN(full) for low-h  (harmful message passing)
  H2: GNN(sparse) > MLP for low-h  (retained edges carry genuine signal)  <- central
  H3: GNN(full) > GNN(sparse) for high-h  (sparsification hurts homophilous graphs)
  H4: GNN(sparse) - MLP gap is monotone decreasing in h

Model conditions per (h, graph_seed, arch):
  - {arch}_full       : Flexible{Arch} on full graph
  - {arch}_sparse_r*  : Flexible{Arch} on approx_er sparsified graph (r in [0.2, 0.4, 0.6])
  - {arch}_random_r*  : Flexible{Arch} on randomly sparsified graph (density control)
  - mlp               : FlexibleMLP ignoring edge_index entirely

Output: results/mlp_control/mlp_control.csv
        One row per (h_requested, graph_seed, arch, model_type, retention_ratio, hp_idx).
        Append-safe for parallel invocations (file-locked).

Usage:
    python scripts/run_mlp_control_sweep.py --h 0.1 --graph_seed 0
    python scripts/run_mlp_control_sweep.py --h 0.5 --graph_seed 1 --resume
    python scripts/run_mlp_control_sweep.py --h 0.1 --graph_seed 0 --arch graphsage
    python scripts/run_mlp_control_sweep.py --h 0.1 --graph_seed 0 --arch gat
"""

import argparse
import csv
import fcntl
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats.qmc import LatinHypercube
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.sparsification.core import GraphSparsifier
from src.training.trainer import GNNTrainer

# ── Configuration ──────────────────────────────────────────────────────────────

N_HP_SAMPLES     = 20              # balanced HPO coverage
N_SEEDS          = 3               # training seeds per HP config
EPOCHS           = 500
PATIENCE         = 50
RETENTION_RATIOS = [0.2, 0.4, 0.6] # r sweep: separates density from quality
SPARSIFIER       = "approx_er"     # best metric from NB04

# cSBM defaults — match Sweep 1 parameters exactly
N_NODES    = 1000
N_CLASSES  = 5
AVG_DEGREE = 10.0
MU         = 1.0
N_FEATURES = 512   # matches sweeps 1-3 (v1 erroneously used 64)

LR_MIN,  LR_MAX = 1e-4, 1e-1
WD_MIN,  WD_MAX = 0.0,  5e-2
DO_MIN,  DO_MAX = 0.0,  0.9
HIDDEN_CHOICES  = [8, 16, 32, 64, 128, 256]
LAYER_CHOICES   = [1, 2, 3, 4]

RESULTS_DIR = REPO_ROOT / "results" / "mlp_control"
# Allow an env-variable override so the shell script can redirect writes to
# /tmp/ during concurrent runs (avoids cloud-sync race conditions).
_default_csv = RESULTS_DIR / "mlp_control.csv"
CSV_PATH     = Path(os.environ.get("MLP_CSV_PATH", str(_default_csv)))

CSV_FIELDNAMES = [
    "h_requested", "h_effective", "mu", "d", "n", "c", "graph_seed",
    "arch", "model_type", "retention_ratio", "sparsifier_metric",
    "hp_idx", "lr", "weight_decay", "dropout", "hidden_channels", "num_layers",
    "acc_mean", "acc_std", "f1_mean", "f1_std",
]


# ── Models ─────────────────────────────────────────────────────────────────────

class FlexibleGCN(nn.Module):
    """Variable-depth GCN (matches run_synthetic_hpo.py)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
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
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class FlexibleSAGE(nn.Module):
    """Variable-depth GraphSAGE with mean aggregation."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class FlexibleGAT(nn.Module):
    """Variable-depth GAT (heads=2 on hidden layers, heads=1 on output)."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        heads = 2
        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, out_channels, heads=1, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


ARCH_REGISTRY = {
    "gcn":        FlexibleGCN,
    "graphsage":  FlexibleSAGE,
    "gat":        FlexibleGAT,
}


class FlexibleMLP(nn.Module):
    """Pure MLP — identical HP space as FlexibleGCN, ignores edge_index.

    Zero-edge baseline: can only use node features.
    GCN(sparse) beating this proves retained edges carry structural signal.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.layers  = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, edge_weight=None):
        x = data.x  # edge_index intentionally ignored
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


# ── Graph generation ───────────────────────────────────────────────────────────

def build_csbm_graph(h, mu, d, n, c, f, graph_seed, device):
    """Generate a cSBM graph (mirrors run_synthetic_hpo.py)."""
    try:
        import synth_graph_rs
    except ImportError:
        raise ImportError(
            "synth-graph-rs not installed. "
            "Run: cd /path/to/synth-graph-rs && maturin develop --release"
        )

    edge_index_np, x_np, y_np, effective_h = synth_graph_rs.generate_csbm(
        n=n, c=c, h=h, d=d, f=f, mu=mu,
        ensure_connected=True, seed=graph_seed,
    )

    edge_index = torch.from_numpy(edge_index_np).long()
    x          = torch.from_numpy(x_np).float()
    y          = torch.from_numpy(y_np).long()

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    rng = np.random.default_rng(graph_seed + 1000)
    for cls in range(c):
        idx = np.where(y_np == cls)[0]
        rng.shuffle(idx)
        n_train = min(20, len(idx) // 3)
        n_rest  = len(idx) - n_train
        n_val   = n_rest // 2
        train_mask[idx[:n_train]]              = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]]       = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data.to(device), float(effective_h)


# ── HP sampling ────────────────────────────────────────────────────────────────

def sample_hp_configs(n, seed=42):
    """Latin Hypercube sample over the HP space (matches run_synthetic_hpo.py)."""
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


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_config(graph, model_cls, hp, num_classes, device, seeds):
    """Train model_cls with hp across multiple seeds; return mean/std metrics."""
    accs, f1s = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_cls(
            in_channels=graph.num_node_features,
            hidden_channels=hp["hidden_channels"],
            out_channels=num_classes,
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        ).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
        )
        trainer = GNNTrainer(model=model, optimizer=optimizer, device=device)
        _, history = trainer.train_and_evaluate(graph, epochs=EPOCHS, patience=PATIENCE)
        accs.append(history["test_metrics"]["accuracy"])
        f1s.append(history["test_metrics"]["macro_f1"])
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
    }


# ── CSV helpers ────────────────────────────────────────────────────────────────

def load_completed_keys(csv_path):
    """Return set of (h_requested, graph_seed, arch, model_type, retention_ratio, hp_idx)."""
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((
                    float(row["h_requested"]),
                    int(row["graph_seed"]),
                    row.get("arch", "gcn"),   # backward compat: old rows default to gcn
                    row["model_type"],
                    float(row["retention_ratio"]),
                    int(row["hp_idx"]),
                ))
            except (KeyError, ValueError):
                pass  # skip malformed rows
    return completed


def append_row(csv_path, row: dict):
    """Append one row to the shared CSV. File-locked; header written only once."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Check file size *inside* the lock to avoid double-header race condition
        write_header = os.fstat(f.fileno()).st_size == 0
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


# ── Main sweep ─────────────────────────────────────────────────────────────────

def build_model_conditions(data, graph_seed, arch, device):
    """Build the list of conditions for one (h, graph_seed, arch) combination.

    Conditions:
      - {arch}_full    : Flexible{Arch} on full graph
      - {arch}_sparse  : Flexible{Arch} on approx_er sparsified graph (each r)
      - {arch}_random  : Flexible{Arch} on randomly sparsified graph (density control)
      - mlp            : FlexibleMLP ignoring edge_index entirely

    The {arch}_random condition uses a deterministic seed derived from
    (graph_seed, r) so the random subgraph is reproducible across runs.
    """
    model_cls = ARCH_REGISTRY[arch]

    sparsifier = GraphSparsifier(data, device)
    # Pre-compute approx_er scores once (cached for all r values)
    sparsifier.compute_scores("approx_er")

    conditions = []

    # {arch}_full — full graph, no sparsification
    conditions.append({
        "model_type":        f"{arch}_full",
        "graph":             data,
        "model_cls":         model_cls,
        "retention_ratio":   1.0,
        "sparsifier_metric": "none",
    })

    for r in RETENTION_RATIOS:
        # {arch}_sparse — top-r fraction by approx_er score
        sparse_approx = sparsifier.sparsify("approx_er", r)
        conditions.append({
            "model_type":        f"{arch}_sparse",
            "graph":             sparse_approx,
            "model_cls":         model_cls,
            "retention_ratio":   r,
            "sparsifier_metric": "approx_er",
        })

        # {arch}_random — random r fraction (density-matched control)
        rand_spar = GraphSparsifier(data, device)
        np.random.seed(graph_seed * 10000 + int(round(r * 1000)))
        sparse_random = rand_spar.sparsify("random", r)
        conditions.append({
            "model_type":        f"{arch}_random",
            "graph":             sparse_random,
            "model_cls":         model_cls,
            "retention_ratio":   r,
            "sparsifier_metric": "random",
        })

    # mlp — full graph passed in but edges ignored by FlexibleMLP.forward()
    conditions.append({
        "model_type":        "mlp",
        "graph":             data,
        "model_cls":         FlexibleMLP,
        "retention_ratio":   1.0,
        "sparsifier_metric": "none",
    })

    return conditions


def run(h, graph_seed, arch, device, resume):
    print(f"\n{'#'*65}")
    print(f"# MLP Control: h={h:.2f}  graph_seed={graph_seed}  arch={arch}  device={device}")
    print(f"{'#'*65}")

    completed = load_completed_keys(CSV_PATH) if resume else set()
    if completed:
        n_done = sum(1 for k in completed if k[0] == h and k[1] == graph_seed and k[2] == arch)
        print(f"  Resuming — {n_done} rows already done for this (h, seed, arch)")

    print(f"  Generating cSBM graph (f={N_FEATURES})...")
    data, effective_h = build_csbm_graph(
        h=h, mu=MU, d=AVG_DEGREE, n=N_NODES, c=N_CLASSES,
        f=N_FEATURES, graph_seed=graph_seed, device=device,
    )
    num_classes = int(data.y.max().item()) + 1
    print(f"  {data.num_nodes} nodes  {data.edge_index.size(1)} edges  "
          f"requested_h={h:.3f}  effective_h={effective_h:.3f}")

    print(f"  Building model conditions (arch={arch}, r sweep: {RETENTION_RATIOS})...")
    conditions = build_model_conditions(data, graph_seed, arch, device)
    n_conditions = len(conditions)
    print(f"  {n_conditions} conditions × {N_HP_SAMPLES} HP configs × {N_SEEDS} seeds "
          f"= {n_conditions * N_HP_SAMPLES * N_SEEDS} training runs total")

    hp_configs = sample_hp_configs(N_HP_SAMPLES, seed=42)
    seeds      = list(range(N_SEEDS))

    base_row = dict(
        h_requested=h, h_effective=round(effective_h, 6),
        mu=MU, d=AVG_DEGREE, n=N_NODES, c=N_CLASSES,
        graph_seed=graph_seed, arch=arch,
    )

    for cond in conditions:
        model_type  = cond["model_type"]
        graph       = cond["graph"]
        model_cls   = cond["model_cls"]
        ret_ratio   = cond["retention_ratio"]
        spar_metric = cond["sparsifier_metric"]

        n_edges = graph.edge_index.size(1)
        label   = f"{model_type}  r={ret_ratio:.1f}  metric={spar_metric}  edges={n_edges}"
        print(f"\n  [{label}]")
        t_start  = time.time()
        n_done   = 0
        n_skip   = 0

        for hp_idx, hp in enumerate(hp_configs):
            key = (h, graph_seed, arch, model_type, ret_ratio, hp_idx)
            if key in completed:
                n_skip += 1
                continue
            try:
                metrics = train_one_config(graph, model_cls, hp, num_classes, device, seeds)
            except Exception as e:
                print(f"    [SKIP hp_idx={hp_idx}] {type(e).__name__}: {e}")
                continue

            row = {
                **base_row,
                "model_type":        model_type,
                "retention_ratio":   ret_ratio,
                "sparsifier_metric": spar_metric,
                "hp_idx":            hp_idx,
                "lr":                hp["lr"],
                "weight_decay":      hp["weight_decay"],
                "dropout":           hp["dropout"],
                "hidden_channels":   hp["hidden_channels"],
                "num_layers":        hp["num_layers"],
                **metrics,
            }
            append_row(CSV_PATH, row)
            n_done += 1

            if n_done % 5 == 0:
                elapsed = (time.time() - t_start) / 60
                print(f"    [{n_done:2d} written]  elapsed={elapsed:.1f}m")

        elapsed = (time.time() - t_start) / 60
        print(f"  done — {n_done} written, {n_skip} skipped  [{elapsed:.1f}m]")

    print(f"\n  Saved → {CSV_PATH}")
    del data
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ── CLI ────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="MLP Control Sweep: validate that GNN(sparse) > MLP "
                    "on heterophilous cSBM (NB04 central claim)."
    )
    parser.add_argument("--h",          type=float, required=True,
                        help="Requested edge homophily [0, 1]")
    parser.add_argument("--graph_seed", type=int,   default=0,
                        help="RNG seed for graph generation (default: 0)")
    parser.add_argument("--arch",       type=str,   default="gcn",
                        choices=["gcn", "graphsage", "gat"],
                        help="GNN architecture for sparse/full conditions (default: gcn)")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip already-completed (arch, model_type, retention_ratio, hp_idx) rows")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run(h=args.h, graph_seed=args.graph_seed, arch=args.arch, device=device, resume=args.resume)
    print(f"\nCompleted in {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
