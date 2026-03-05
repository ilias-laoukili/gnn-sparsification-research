#!/usr/bin/env python3
"""Roman-empire sparsification experiment — GPU version (Kaggle / cloud).

Replicates GCN* from "Classic GNNs are Strong Baselines" (Luo et al., NeurIPS 2024)
using paper hyperparameters, then evaluates sparsification methods on top.

Key design choices:
  - N_SPLITS=3  (same as tunedGNN paper)
  - Scores computed once per method, thresholds applied per retention rate
  - Main loop parallelises across all available GPUs (one method per GPU)
  - Results written to /kaggle/working/ when on Kaggle, else results/ locally
  - Supports --resume to skip already-completed configs (safe across sessions)

Estimated runtime on Kaggle T4×2:
  --skip-approxer   ~3.5 h  (45 configs, fits in one 12 h session)
  --approxer-only   ~8 h    (20 configs, second session after downloading CSV)
  Full run          ~12 h   (65 configs, risky for one session)

Usage:
  # On Kaggle — session 1 (fast methods):
  !python scripts/roman_empire_gpu.py --skip-approxer

  # On Kaggle — session 2 (upload CSV from session 1 first):
  !python scripts/roman_empire_gpu.py --approxer-only --resume

  # Locally with a GPU:
  python scripts/roman_empire_gpu.py --skip-approxer

  # Kaggle upload: zip src/ scripts/roman_empire_gpu.py results/roman_empire_results.csv
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch_geometric.data import Data

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import DatasetLoader, GNNTrainer, GraphSparsifier, get_model, set_global_seed

# ── Paths ─────────────────────────────────────────────────────────────────────
# Write directly to /kaggle/working so files appear in Kaggle's output panel
_KAGGLE_WORKING = Path('/kaggle/working')
OUT_DIR        = _KAGGLE_WORKING if _KAGGLE_WORKING.exists() else REPO_ROOT / 'results'
RESULTS_CSV    = OUT_DIR / 'roman_empire_results.csv'
BASELINES_JSON = OUT_DIR / 'roman_empire_baselines.json'

# ── Paper hyperparameters (Luo et al., NeurIPS 2024) ─────────────────────────
BEST_HP = {
    'hidden_channels': 512,
    'num_layers':      9,
    'dropout':         0.5,
    'lr':              1e-3,
    'weight_decay':    0.0,
}

GLOBAL_SEED     = 42
N_SPLITS        = 3      # same as tunedGNN paper (3 fixed splits)
EPOCHS          = 2500
PATIENCE        = 200
DATASET_NAME    = 'roman_empire'
DATA_ROOT       = str(REPO_ROOT / 'data')

RETENTION_RATES = [0.9, 0.8, 0.6, 0.4, 0.2]

SPARSIFICATION_CONFIGS = [
    ('Random',          'random',         False, False, 'threshold'),
    ('Jaccard-T',       'jaccard',        False, False, 'threshold'),
    ('AA-T',            'adamic_adar',    False, False, 'threshold'),
    ('ApproxER-T',      'approx_er',      False, False, 'threshold'),
    ('FeatCos-T',       'feature_cosine', False, False, 'threshold'),
    ('Jaccard-IT',      'jaccard',        True,  False, 'threshold'),
    ('AA-IT',           'adamic_adar',    True,  False, 'threshold'),
    ('ApproxER-IT',     'approx_er',      True,  False, 'threshold'),
    ('FeatCos-IT',      'feature_cosine', True,  False, 'threshold'),
    ('ApproxER-T-W',    'approx_er',      False, True,  'threshold'),
    ('ApproxER-IT-W',   'approx_er',      True,  True,  'threshold'),
    ('Jaccard-Samp',    'jaccard',        False, False, 'sampled'),
    ('Jaccard-DegA',    'jaccard',        False, False, 'degree_aware'),
]


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(resume: bool) -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('kaggle_exp')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(OUT_DIR / 'roman_empire_experiment.log',
                             mode='a' if resume else 'w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ── Checkpointing ─────────────────────────────────────────────────────────────
def _checkpoint(record: dict):
    pd.DataFrame([record]).to_csv(
        RESULTS_CSV, mode='a', header=not RESULTS_CSV.exists(), index=False)


def load_completed_keys(resume: bool) -> set:
    if not resume or not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(zip(df['Method'], df['Model'], df['TargetRetention']))


# ── Baselines ─────────────────────────────────────────────────────────────────
def run_baselines(data, all_split_data, num_features, num_classes,
                  device, logger) -> dict:
    """Run GCN* baseline on N_SPLITS splits sequentially on `device`."""
    logger.info(f'Running GCN* baseline × {N_SPLITS} splits on {device} ...')
    dev = torch.device(device)
    records = []
    for split_idx, sd in enumerate(all_split_data):
        graph = Data(
            x=data.x.to(dev), edge_index=data.edge_index.to(dev), y=data.y.to(dev),
            train_mask=sd.train_mask.to(dev), val_mask=sd.val_mask.to(dev),
            test_mask=sd.test_mask.to(dev), num_nodes=data.num_nodes,
        )
        set_global_seed(GLOBAL_SEED + split_idx)
        model     = get_model('gcn_star', num_features, BEST_HP['hidden_channels'],
                              num_classes, dropout=BEST_HP['dropout'],
                              num_layers=BEST_HP['num_layers'])
        optimiser = Adam(model.parameters(),
                         lr=BEST_HP['lr'], weight_decay=BEST_HP['weight_decay'])
        trainer   = GNNTrainer(model, optimiser, device=device)

        t0      = time.perf_counter()
        history = trainer.train(graph, epochs=EPOCHS, patience=PATIENCE)
        t1      = time.perf_counter()
        metrics = trainer.compute_metrics(graph, graph.test_mask)
        records.append({
            'split_idx':      split_idx,
            'train_time_s':   t1 - t0,
            'test_accuracy':  metrics['accuracy'],
            'macro_f1':       metrics['macro_f1'],
            'epochs_trained': history['epochs_trained'],
        })
        logger.info(f'  split {split_idx}: acc={metrics["accuracy"]*100:.2f}%  '
                    f'epochs={history["epochs_trained"]}  t={t1-t0:.0f}s')

    agg = {}
    for key in ('train_time_s', 'test_accuracy', 'macro_f1', 'epochs_trained'):
        vals = [r[key] for r in records]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std']  = float(np.std(vals))
    agg['edge_count_mean'] = float(data.edge_index.size(1))
    agg['edge_count_std']  = 0.0
    agg['test_accuracy_seeds'] = json.dumps([r['test_accuracy'] for r in records])
    agg['macro_f1_seeds']      = json.dumps([r['macro_f1']      for r in records])
    logger.info(f'  GCN* baseline: {agg["test_accuracy_mean"]*100:.2f}% '
                f'± {agg["test_accuracy_std"]*100:.2f}%  (paper: 91.27 ± 0.20%)')
    return agg


# ── Main-loop worker (subprocess — one method × ALL retentions on one GPU) ────
def _main_worker(args: tuple):
    """Run one method across ALL retention rates on a given GPU.

    Scores are computed once, then thresholds are applied per retention rate.
    This avoids recomputing expensive metrics (e.g. ApproxER) 5× per method.
    Returns a list of (retention, result_dict | None, error | None).
    """
    (label, metric, keep_lowest, weighted, variant, retentions,
     x_cpu, edge_index_cpu, y_cpu, num_nodes, num_features, num_classes,
     split_masks, best_hp, global_seed, epochs, patience,
     device_str, repo_root_str) = args

    import sys as _sys
    if repo_root_str not in _sys.path:
        _sys.path.insert(0, repo_root_str)

    import json as _json
    import time as _time
    import torch as _torch
    import numpy as _np
    from torch.optim import Adam as _Adam
    from torch_geometric.data import Data as _Data
    from src import GNNTrainer, GraphSparsifier, get_model, set_global_seed

    _torch.cuda.set_device(_torch.device(device_str))

    full_graph = _Data(x=x_cpu, edge_index=edge_index_cpu,
                       y=y_cpu, num_nodes=num_nodes)
    sp = GraphSparsifier(full_graph, device='cpu')

    # ── Compute scores once for all retentions ────────────────────────────────
    all_scores = None
    if metric != 'random' and variant == 'threshold':
        try:
            all_scores = sp.compute_scores(metric)  # shape: [num_edges]
        except Exception as e:
            # Return failure for all retentions
            return [(r, None, str(e)) for r in retentions]

    dev   = _torch.device(device_str)
    y_dev = y_cpu.to(dev)
    results = []

    for retention in retentions:
        # ── Sparsify ──────────────────────────────────────────────────────────
        try:
            if variant == 'sampled':
                sparse, mask = sp.sparsify_sampled(
                    metric, retention, seed=global_seed, return_mask=True)
            elif variant == 'degree_aware':
                sparse, mask = sp.sparsify_degree_aware(
                    metric, retention, return_mask=True)
            else:
                sparse, mask = sp.sparsify(
                    metric, retention, return_mask=True, keep_lowest=keep_lowest)
            mask_np = mask.numpy()
        except Exception as e:
            results.append((retention, None, str(e)))
            continue

        actual_ret = float(sparse.edge_index.size(1)) / float(edge_index_cpu.size(1))

        # ── Edge weight (reuse pre-computed scores) ───────────────────────────
        if weighted and metric != 'random' and all_scores is not None:
            scores = all_scores[mask_np]
            mn, mx = scores.min(), scores.max()
            norm   = (scores - mn) / (mx - mn + 1e-8)
            if keep_lowest:
                norm = 1.0 - norm
            edge_weight = _torch.tensor(norm, dtype=_torch.float32).to(dev)
        else:
            edge_weight = None

        sparse_x          = sparse.x.to(dev)
        sparse_edge_index = sparse.edge_index.to(dev)

        # ── Train across all splits ───────────────────────────────────────────
        records = []
        for split_idx, (train_mask, val_mask, test_mask) in enumerate(split_masks):
            graph = _Data(
                x=sparse_x, y=y_dev, edge_index=sparse_edge_index,
                train_mask=train_mask.to(dev),
                val_mask=val_mask.to(dev),
                test_mask=test_mask.to(dev),
                num_nodes=num_nodes,
            )
            set_global_seed(global_seed + split_idx)
            model     = get_model('gcn_star', num_features, best_hp['hidden_channels'],
                                  num_classes, dropout=best_hp['dropout'],
                                  num_layers=best_hp['num_layers'])
            optimiser = _Adam(model.parameters(),
                              lr=best_hp['lr'], weight_decay=best_hp['weight_decay'])
            trainer   = GNNTrainer(model, optimiser, device=device_str)

            t0      = _time.perf_counter()
            history = trainer.train(graph, epochs=epochs, patience=patience,
                                    edge_weight=edge_weight)
            t1      = _time.perf_counter()
            metrics = trainer.compute_metrics(graph, graph.test_mask,
                                              edge_weight=edge_weight)
            records.append({
                'train_time_s':   t1 - t0,
                'test_accuracy':  metrics['accuracy'],
                'macro_f1':       metrics['macro_f1'],
                'epochs_trained': history['epochs_trained'],
                'edge_count':     int(sparse.edge_index.size(1)),
            })

        # ── Aggregate ─────────────────────────────────────────────────────────
        agg = {}
        for key in records[0]:
            vals = [r[key] for r in records]
            agg[f'{key}_mean'] = float(_np.mean(vals))
            agg[f'{key}_std']  = float(_np.std(vals))
        agg['test_accuracy_seeds'] = _json.dumps([r['test_accuracy'] for r in records])
        agg['macro_f1_seeds']      = _json.dumps([r['macro_f1']      for r in records])
        results.append((retention, {'actual_ret': actual_ret, **agg}, None))

    return results


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_main_loop(data, all_split_data, num_features, num_classes,
                  base_acc, base_time, n_gpus, completed, active_configs, logger):
    total_configs = len(active_configs) * len(RETENTION_RATES)

    # Group by method — one worker per method handles all its retention rates.
    # Scores are computed once per method, not once per (method, retention).
    method_jobs = []
    for label, metric, keep_lowest, weighted, variant in active_configs:
        pending = [r for r in RETENTION_RATES
                   if (label, 'gcn_star', r) not in completed]
        if pending:
            method_jobs.append((label, metric, keep_lowest, weighted,
                                variant, pending))

    n_completed_configs = total_configs - sum(len(j[5]) for j in method_jobs)
    skipped_configs     = sum(len(RETENTION_RATES) - len(j[5]) for j in method_jobs)
    if skipped_configs:
        logger.info(f'  Skipping {skipped_configs} already-completed configs.')
    logger.info(f'Main loop: {sum(len(j[5]) for j in method_jobs)} configs remaining  '
                f'({len(method_jobs)} methods × up to {len(RETENTION_RATES)} retentions, '
                f'{n_gpus} GPU workers)')

    cpu            = torch.device('cpu')
    x_cpu          = data.x.to(cpu)
    edge_index_cpu = data.edge_index.to(cpu)
    y_cpu          = data.y.to(cpu)
    split_masks    = [(sd.train_mask.to(cpu), sd.val_mask.to(cpu),
                       sd.test_mask.to(cpu)) for sd in all_split_data]

    worker_args = [
        (label, metric, keep_lowest, weighted, variant, pending_retentions,
         x_cpu, edge_index_cpu, y_cpu, data.num_nodes, num_features, num_classes,
         split_masks, BEST_HP, GLOBAL_SEED, EPOCHS, PATIENCE,
         f'cuda:{i % n_gpus}', str(REPO_ROOT))
        for i, (label, metric, keep_lowest, weighted, variant, pending_retentions)
        in enumerate(method_jobs)
    ]

    done    = 0
    t_start = time.perf_counter()
    mp_ctx  = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=mp_ctx) as pool:
        future_to_job = {pool.submit(_main_worker, arg): job
                         for arg, job in zip(worker_args, method_jobs)}
        for future in as_completed(future_to_job):
            label, metric, keep_lowest, weighted, variant, _ = \
                future_to_job[future]
            retention_results = future.result()  # list of (retention, res, err)

            for retention, res, err in retention_results:
                if err:
                    logger.warning(f'FAIL {label} r={retention}: {err}')
                    continue

                done  += 1
                acc_d  = (res['test_accuracy_mean'] - base_acc) * 100
                speedup = base_time / max(res['train_time_s_mean'], 1e-9)
                record  = dict(
                    Method=label, Metric=metric, KeepLowest=keep_lowest,
                    Weighted=weighted, Variant=variant, Model='gcn_star',
                    TargetRetention=retention, ActualRetention=res['actual_ret'],
                    Speedup=speedup, AccDelta=acc_d,
                    **{k: v for k, v in res.items() if k != 'actual_ret'},
                )
                _checkpoint(record)

                elapsed = (time.perf_counter() - t_start) / 60
                n_remaining = total_configs - n_completed_configs - done
                eta = elapsed / done * n_remaining if done > 0 else 0
                logger.info(
                    f'[{n_completed_configs+done:3d}/{total_configs}] '
                    f'{label:<16s} r={retention:.0%}  '
                    f'acc={res["test_accuracy_mean"]*100:.2f}%'
                    f'±{res["test_accuracy_std"]*100:.2f}%  '
                    f'Δ={acc_d:+.2f}pp  {speedup:.2f}×  '
                    f'elapsed={elapsed:.0f}m  eta≈{eta:.0f}m'
                )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Roman-empire experiment (Kaggle/GPU)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed main-loop configs in results.csv')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')
    parser.add_argument('--skip-approxer', action='store_true',
                        help='Exclude ApproxER configs (~3.5 h, fits in one session)')
    parser.add_argument('--approxer-only', action='store_true',
                        help='Run only ApproxER configs (~8 h, use in a second session)')
    args = parser.parse_args()

    logger = setup_logging(args.resume)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.resume and RESULTS_CSV.exists():
        RESULTS_CSV.unlink()
        logger.info(f'Fresh run: deleted {RESULTS_CSV.name}')

    if args.cpu or not torch.cuda.is_available():
        device  = 'cpu'
        n_gpus  = 1
    else:
        n_gpus  = torch.cuda.device_count()
        device  = 'cuda:0'
        for i in range(n_gpus):
            logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}  '
                        f'({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')

    logger.info('=' * 65)
    # ── Filter configs based on flags ─────────────────────────────────────────
    active_configs = [c for c in SPARSIFICATION_CONFIGS
                      if not (args.skip_approxer and c[1] == 'approx_er')
                      and not (args.approxer_only and c[1] != 'approx_er')]

    logger.info('Roman-empire sparsification experiment (GCN* — NeurIPS 2024)')
    logger.info(f'Device  : {device}  ({n_gpus} GPU(s))')
    logger.info(f'Resume  : {args.resume}')
    logger.info(f'BEST_HP : {BEST_HP}')
    logger.info(f'Epochs  : {EPOCHS}  (patience={PATIENCE})')
    logger.info(f'Splits  : {N_SPLITS}')
    if args.skip_approxer:
        logger.info('Mode    : skip-approxer  (9 methods × 5 rates = 45 configs)')
    elif args.approxer_only:
        logger.info('Mode    : approxer-only  (4 methods × 5 rates = 20 configs)')
    logger.info('=' * 65)

    # ── Load dataset ──────────────────────────────────────────────────────────
    set_global_seed(GLOBAL_SEED)
    loader = DatasetLoader(root=DATA_ROOT)
    data, num_features, num_classes = loader.get_dataset(
        DATASET_NAME, device='cpu', split_idx=0)
    logger.info(f'Dataset: {data.num_nodes:,} nodes, '
                f'{data.edge_index.size(1):,} edges, '
                f'{num_features} features, {num_classes} classes')

    logger.info(f'Pre-loading {N_SPLITS} splits ...')
    all_split_data = []
    for i in range(N_SPLITS):
        sd, _, _ = loader.get_dataset(DATASET_NAME, device='cpu', split_idx=i)
        all_split_data.append(sd)
    logger.info(f'Splits 0–{N_SPLITS-1} loaded.')

    # ── Baselines on cuda:0 ───────────────────────────────────────────────────
    if args.resume and BASELINES_JSON.exists():
        with open(BASELINES_JSON) as f:
            bl_data = json.load(f)
        bl = bl_data['gcn_star']
        logger.info(f'Baselines loaded from {BASELINES_JSON.name} (skipping re-run)')
    else:
        bl = run_baselines(data, all_split_data, num_features, num_classes,
                           device, logger)
        with open(BASELINES_JSON, 'w') as f:
            json.dump({'_best_hp': BEST_HP, '_device': device, '_n_splits': N_SPLITS,
                       'gcn_star': {
                           k: (json.loads(v) if isinstance(v, str) and k.endswith('_seeds') else v)
                           for k, v in bl.items()
                       }}, f, indent=2)
        logger.info(f'Baselines saved → {BASELINES_JSON.name}')

    base_acc  = bl['test_accuracy_mean']
    base_time = bl['train_time_s_mean']

    # ── Main loop — parallel across all GPUs ─────────────────────────────────
    completed = load_completed_keys(args.resume)
    run_main_loop(data, all_split_data, num_features, num_classes,
                  base_acc, base_time, n_gpus, completed, active_configs, logger)

    total_rows = pd.read_csv(RESULTS_CSV).shape[0] if RESULTS_CSV.exists() else 0
    logger.info('=' * 65)
    logger.info(f'DONE  —  {total_rows} configs in {RESULTS_CSV.name}')
    logger.info('=' * 65)


if __name__ == '__main__':
    main()
